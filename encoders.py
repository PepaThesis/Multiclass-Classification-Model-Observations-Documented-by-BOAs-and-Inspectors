"""
Dutch Multi-Class Text Classifier — Multi-Model Edition
=========================================================
Supports:
  1. pdelobelle/robbert-v2-dutch-base          (RobBERT v2)
  2. DTAI-KULeuven/robbert-2023-dutch-base      (RobBERT 2023 base)
  3. DTAI-KULeuven/robbert-2023-dutch-large     (RobBERT 2023 large)
  4. GroNLP/bert-base-dutch-cased               (BERTje)
  5. bert-base-multilingual-cased               (mBERT)
  6. microsoft/mdeberta-v3-base                 (mDeBERTa v3)
  7. microsoft/deberta-v3-large                 (DeBERTa-XLM / v3 large)
  8. Geotrend/bert-base-nl-cased                (Geotrend NL BERT)

Inputs : train.xlsx  (columns: text, class)
         dev.xlsx    (columns: text, class)
Outputs: - classification_report.txt
         - confusion_matrix.png
         - misclassifications.xlsx
         - low_confidence_predictions.xlsx
         - training_history.png
         - best_model/  (checkpoint)

Set MODEL_KEY below to one of the keys in MODEL_CONFIGS.

STABILITY CHANGES (silent-kernel-death fixes):
  - num_workers=0 and pin_memory=False (worker processes duplicate the
    dataset in RAM on Kaggle and are the most common silent-hang source)
  - dynamic per-batch padding via DataCollatorWithPadding instead of
    padding="max_length" (less RAM, and faster: median text ~74 tokens)
  - RAM/VRAM probes after each stage and each epoch, so an impending
    out-of-memory kill is visible in the log before it happens
  - outputs written to /kaggle/working when available, so checkpoints
    survive "Save & Run All" background execution
"""

# ── Install missing packages (Kaggle / Colab) ─────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("transformers", "datasets", "accelerate", "openpyxl", "scikit-learn", "seaborn", "psutil")

# ── Imports ───────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import psutil  # CHANGED: memory probes

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,  # CHANGED: dynamic padding
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
"

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
#  Each entry contains recommended settings tuned for Dutch NLP fine-tuning.
#  All values can be overridden in the USER CONFIG section below.
# ═══════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {

    # ── 1. RobBERT v2 ────────────────────────────────────────────────────
    "robbert_v2": dict(
        model_name         = "pdelobelle/robbert-v2-dutch-base",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 5,
        lr                 = 2e-5,
        warmup_ratio       = 0.06,
        weight_decay       = 0.1,
        classifier_dropout = 0.1,
    ),

    # ── 2. RobBERT 2023 base ─────────────────────────────────────────────
    "robbert_2023_base": dict(
        model_name         = "DTAI-KULeuven/robbert-2023-dutch-base",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 5,
        lr                 = 3e-5,
        warmup_ratio       = 0.06,
        weight_decay       = 0.1,
        classifier_dropout = 0.1,
    ),

    # ── 3. RobBERT 2023 large ────────────────────────────────────────────
    "robbert_2023_large": dict(
        model_name         = "DTAI-KULeuven/robbert-2023-dutch-large",
        max_len            = 128,
        batch_size         = 8,
        accumulation_steps = 2,
        epochs             = 4,
        lr                 = 2e-5,
        warmup_ratio       = 0.06,
        weight_decay       = 0.1,
        classifier_dropout = 0.1,
    ),

    # ── 4. BERTje ────────────────────────────────────────────────────────
    "bertje": dict(
        model_name         = "GroNLP/bert-base-dutch-cased",
        max_len            = 265,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 5,
        lr                 = 2e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),

    # ── 5. mBERT ─────────────────────────────────────────────────────────
    "mbert": dict(
        model_name         = "bert-base-multilingual-cased",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 4,
        lr                 = 3e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),

    # ── 6. mDeBERTa v3 base ──────────────────────────────────────────────
    "mdeberta_v3_base": dict(
        model_name         = "microsoft/mdeberta-v3-base",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 5,
        lr                 = 2e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),

    # ── 7. DeBERTa v3 large (DeBERTa-XLM) ───────────────────────────────
    "deberta_v3_large": dict(
        model_name         = "microsoft/deberta-v3-large",
        max_len            = 128,
        batch_size         = 4,
        accumulation_steps = 4,
        epochs             = 4,
        lr                 = 1e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),

    # ── 8. Geotrend NL BERT ──────────────────────────────────────────────
    "geotrend_nl": dict(
        model_name         = "Geotrend/bert-base-nl-cased",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 4,
        lr                 = 3e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),

    # ── 9. XLM-RoBERTa base ──────────────────────────────────────────────
    "XLM": dict(
        model_name         = "xlm-roberta-base",
        max_len            = 128,
        batch_size         = 16,
        accumulation_steps = 1,
        epochs             = 4,
        lr                 = 3e-5,
        warmup_ratio       = 0.1,
        weight_decay       = 0.01,
        classifier_dropout = 0.1,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  USER CONFIG  ← change these two lines to switch model
# ═══════════════════════════════════════════════════════════════════════════
MODEL_KEY   = "mdeberta_v3_base"  # one of the keys in MODEL_CONFIGS above
TRAIN_FILE  = "/kaggle/input/datasets/hannekehuls/thesis-final/train.xlsx"
DEV_FILE    = "/kaggle/input/datasets/hannekehuls/thesis-final/test9.xlsx"

# CHANGED: write everything to /kaggle/working when on Kaggle, so outputs
# and checkpoints survive background ("Save & Run All") execution.
WORK_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR  = os.path.join(WORK_DIR, "best_model")
CONFIDENCE_THRESH = 0.70
SEED        = 42

# Pull settings from registry (override any value here if needed)
_cfg               = MODEL_CONFIGS[MODEL_KEY]
MODEL_NAME         = _cfg["model_name"]
MAX_LEN            = _cfg["max_len"]
BATCH_SIZE         = _cfg["batch_size"]
ACCUMULATION_STEPS = _cfg["accumulation_steps"]
EPOCHS             = _cfg["epochs"]
LR                 = _cfg["lr"]
WARMUP_RATIO       = _cfg["warmup_ratio"]
WEIGHT_DECAY       = _cfg["weight_decay"]
CLASSIFIER_DROPOUT = _cfg["classifier_dropout"]


# ═══════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY, DEVICE & MEMORY PROBE
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHANGED: memory probe — call at every stage; if RAM% climbs toward 100
# in the log, the silent kernel death is an out-of-memory kill.
def mem_report(tag: str):
    ram = psutil.virtual_memory()
    line = f"  [mem] {tag}: RAM {ram.percent:.0f}% ({ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB)"
    if device.type == "cuda":
        line += f"  VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB"
    print(line)

print(f"▶ Model       : {MODEL_NAME}  [{MODEL_KEY}]")
print(f"▶ Using device: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Effective batch size: {BATCH_SIZE} × {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")
print(f"  LR={LR}  WD={WEIGHT_DECAY}  EPOCHS={EPOCHS}  MAX_LEN={MAX_LEN}")
print(f"  Output dir  : {OUTPUT_DIR}")
mem_report("start")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("\n▶ Loading data …")
train_df = pd.read_excel(TRAIN_FILE)
dev_df   = pd.read_excel(DEV_FILE)

train_df.columns = [c.strip().lower() for c in train_df.columns]
dev_df.columns   = [c.strip().lower() for c in dev_df.columns]

assert "text"  in train_df.columns, "Column 'text' not found in train file"
assert "class" in train_df.columns, "Column 'class' not found in train file"
assert "text"  in dev_df.columns,   "Column 'text' not found in dev file"
assert "class" in dev_df.columns,   "Column 'class' not found in dev file"

train_df = train_df.dropna(subset=["text", "class"]).reset_index(drop=True)
dev_df   = dev_df.dropna(subset=["text", "class"]).reset_index(drop=True)

le = LabelEncoder()
le.fit(pd.concat([train_df["class"], dev_df["class"]], ignore_index=True))
class_names = list(le.classes_)
num_labels  = len(class_names)

train_df["label"] = le.transform(train_df["class"])
dev_df["label"]   = le.transform(dev_df["class"])

print(f"  Train samples : {len(train_df)}")
print(f"  Dev   samples : {len(dev_df)}")
print(f"  Classes ({num_labels}): {class_names}")

train_classes = set(train_df["class"].unique())
dev_only      = set(dev_df["class"].unique()) - train_classes
if dev_only:
    print(f"  ⚠  Classes in dev but NOT in train: {sorted(dev_only)}")
mem_report("after data loading")


# ═══════════════════════════════════════════════════════════════════════════
#  TOKEN-LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n▶ Analysing token lengths (train set) …")
_tok_check = AutoTokenizer.from_pretrained(MODEL_NAME)
_lengths   = [
    len(_tok_check(str(t), truncation=False)["input_ids"])
    for t in train_df["text"]
]
_p50, _p90, _p95, _p99 = (int(np.percentile(_lengths, q)) for q in [50, 90, 95, 99])
_pct_trunc = float(np.mean([l > MAX_LEN for l in _lengths])) * 100
print(f"  p50={_p50}  p90={_p90}  p95={_p95}  p99={_p99}  tokens")
print(f"  Truncated at MAX_LEN={MAX_LEN}: {_pct_trunc:.1f}% of train samples")
if _p95 > MAX_LEN:
    print(f"  ⚠  p95 ({_p95}) > MAX_LEN ({MAX_LEN}) → consider raising MAX_LEN to 256")
else:
    print(f"  ✓ MAX_LEN={MAX_LEN} covers ≥95% of samples")
del _tok_check, _lengths


# ═══════════════════════════════════════════════════════════════════════════
#  TOKENISER & DATASET
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n▶ Loading tokeniser for '{MODEL_NAME}' …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# CHANGED: no padding here — the collator pads each batch to its own
# longest sequence (dynamic padding). Less RAM, less compute.
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            truncation=True,
        )
        enc["labels"] = self.labels[idx]
        return enc

train_dataset = TextDataset(train_df["text"], train_df["label"], tokenizer, MAX_LEN)
dev_dataset   = TextDataset(dev_df["text"],   dev_df["label"],   tokenizer, MAX_LEN)

# CHANGED: dynamic per-batch padding (handles token_type_ids automatically
# for BERT-style models; RoBERTa/DeBERTa tokenizers simply don't emit them)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# CHANGED: num_workers=0 and pin_memory=False — worker processes duplicate
# the dataset in RAM on Kaggle/Colab and are the most common cause of
# silent hangs/kernel deaths there.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False, collate_fn=collator)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False, collate_fn=collator)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n▶ Loading model …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)
if CLASSIFIER_DROPOUT is not None:
    model.config.classifier_dropout = CLASSIFIER_DROPOUT
    print(f"  classifier_dropout set to {CLASSIFIER_DROPOUT}")
model = model.to(device)
mem_report("after model loading")


# ═══════════════════════════════════════════════════════════════════════════
#  OPTIMISER & SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
total_steps  = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"  Optimiser steps: {total_steps}  (warmup: {warmup_steps})")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_confs, all_probs = [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            probs      = torch.softmax(outputs.logits, dim=1)
            top_probs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_confs.extend(top_probs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confs),
        np.array(all_probs),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n▶ Training …\n" + "─" * 60)
best_dev_loss = float("inf")
os.makedirs(OUTPUT_DIR, exist_ok=True)
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, 1):
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss / ACCUMULATION_STEPS
        loss.backward()
        total_train_loss += outputs.loss.item()

        if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 20 == 0 or step == len(train_loader):
            print(f"  Epoch {epoch}/{EPOCHS}  Step {step:>4}/{len(train_loader)}  "
                  f"Train loss: {total_train_loss / step:.4f}", end="\r")

    avg_train = total_train_loss / len(train_loader)
    dev_loss, preds, labels_arr, confs, _ = evaluate(model, dev_loader)
    accuracy  = (preds == labels_arr).mean()

    history.append({"epoch": epoch, "train_loss": avg_train, "dev_loss": dev_loss, "accuracy": accuracy})
    print(f"\n  Epoch {epoch}/{EPOCHS}  train_loss={avg_train:.4f}  "
          f"dev_loss={dev_loss:.4f}  acc={accuracy:.4f}")
    mem_report(f"after epoch {epoch}")

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ✓ Best model saved (dev_loss={dev_loss:.4f})")

print("\n" + "─" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n▶ Loading best model for final evaluation …")
best_model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR, num_labels=num_labels, ignore_mismatched_sizes=True
).to(device)
_, final_preds, final_labels, final_confs, final_probs = evaluate(best_model, dev_loader)

present_ids   = sorted(set(final_labels.tolist()) | set(final_preds.tolist()))
present_names = [class_names[i] for i in present_ids]

print("\n▶ Classification report (dev set):\n")
report = classification_report(
    final_labels, final_preds,
    labels=present_ids, target_names=present_names,
    digits=4, zero_division=0,
)
print(report)

with open(os.path.join(WORK_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME}  [{MODEL_KEY}]\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# ── Confusion matrix ──────────────────────────────────────────────────────
print("▶ Saving confusion matrix …")
cm = confusion_matrix(final_labels, final_preds, labels=present_ids)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names,
            ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)
ax.set_title(f"Confusion Matrix — {MODEL_KEY}", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrix.png")

# ── Misclassifications ────────────────────────────────────────────────────
print("▶ Saving misclassifications …")
mis_mask        = final_preds != final_labels
mis_probs       = final_probs[mis_mask]
second_ids      = np.argsort(mis_probs, axis=1)[:, -2]
second_names    = [class_names[i] for i in second_ids]
second_probs    = mis_probs[np.arange(len(mis_probs)), second_ids]

misclass_df = pd.DataFrame({
    "text":            dev_df["text"].values[mis_mask],
    "true_class":      le.inverse_transform(final_labels[mis_mask]),
    "predicted_class": le.inverse_transform(final_preds[mis_mask]),
    "confidence":      final_confs[mis_mask].round(4),
    "2nd_best_class":  second_names,
    "2nd_best_prob":   second_probs.round(4),
    "uncertain":       final_confs[mis_mask] < CONFIDENCE_THRESH,
})
misclass_df = misclass_df.sort_values(["uncertain", "confidence"], ascending=[False, True])
misclass_df.to_excel(os.path.join(WORK_DIR, "misclassifications.xlsx"), index=False)
print(f"  Saved: misclassifications.xlsx  ({len(misclass_df)} samples)")

# ── Low-confidence predictions ────────────────────────────────────────────
print(f"▶ Saving low-confidence predictions (threshold={CONFIDENCE_THRESH}) …")
lc_mask     = final_confs < CONFIDENCE_THRESH
lc_probs    = final_probs[lc_mask]
lc_sec_ids  = np.argsort(lc_probs, axis=1)[:, -2]
lc_sec_p    = lc_probs[np.arange(len(lc_probs)), lc_sec_ids]

lowconf_df = pd.DataFrame({
    "text":            dev_df["text"].values[lc_mask],
    "true_class":      le.inverse_transform(final_labels[lc_mask]),
    "predicted_class": le.inverse_transform(final_preds[lc_mask]),
    "confidence":      final_confs[lc_mask].round(4),
    "2nd_best_class":  [class_names[i] for i in lc_sec_ids],
    "2nd_best_prob":   lc_sec_p.round(4),
    "correct":         (final_preds == final_labels)[lc_mask],
})
lowconf_df = lowconf_df.sort_values("confidence")
lowconf_df.to_excel(os.path.join(WORK_DIR, "low_confidence_predictions.xlsx"), index=False)
n_unc = lc_mask.sum()
print(f"  Saved: low_confidence_predictions.xlsx  ({n_unc} samples)")

# ── Training history ──────────────────────────────────────────────────────
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
axes[0].plot(history_df["epoch"], history_df["dev_loss"],   marker="o", label="Dev")
axes[0].set_title(f"Loss — {MODEL_KEY}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].plot(history_df["epoch"], history_df["accuracy"], marker="o", color="green")
axes[1].set_title(f"Dev accuracy — {MODEL_KEY}")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "training_history.png"), dpi=150)
plt.close()
print("  Saved: training_history.png")

# ── Summary ───────────────────────────────────────────────────────────────
overall_acc  = (final_preds == final_labels).mean()
n_mis        = mis_mask.sum()
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Model               : {MODEL_NAME}")
print(f"  Dev samples         : {len(dev_df)}")
print(f"  Overall accuracy    : {overall_acc:.4f}")
print(f"  Misclassified       : {n_mis} ({n_mis/len(dev_df)*100:.1f}%)")
print(f"  Low-confidence (<{CONFIDENCE_THRESH}): {n_unc} ({n_unc/len(dev_df)*100:.1f}%)")
print(f"  Best dev loss       : {best_dev_loss:.4f}")
mem_report("end")
print("\nOutput files (in " + WORK_DIR + "):")
print("  • classification_report.txt")
print("  • confusion_matrix.png")
print("  • misclassifications.xlsx")
print("  • low_confidence_predictions.xlsx")
print("  • training_history.png")
print(f"  • {OUTPUT_DIR}/  (model checkpoint)")
print("=" * 60)