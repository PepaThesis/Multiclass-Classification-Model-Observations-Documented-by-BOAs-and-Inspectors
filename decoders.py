"""
Multi-Class Text Classifier — Decoder models (QLoRA)
========================================================
Supports all five decoder models out of the box. Change MODEL_NAME and run.

Decoder models (this script):
    "GroNLP/gpt2-small-dutch"          ~117M  Dutch GPT-2
    "Qwen/Qwen2.5-0.5B"                ~500M  multilingual
    "facebook/opt-1.3b"                ~1.3B  multilingual baseline
    "BramVanroy/fietje-2b"             ~2.0B  Dutch  (reduce BATCH_SIZE to 2 if OOM)

THIS VERSION: gradient checkpointing fully DISABLED (both the explicit
enable call and the prepare_model_for_kbit_training flag) — the Phi
remote-code path used by Fietje is incompatible with checkpointing on
current torch. use_cache is still disabled during training.
If a model OOMs without checkpointing: set batch_size=1 and
accumulation=16 in its registry entry (same effective batch of 16).

Inputs : train.xlsx  (columns: text, class)
         dev.xlsx    (columns: text, class)
Outputs: classification_report.txt  |  confusion_matrix.png
         misclassifications.xlsx    |  low_confidence_predictions.xlsx
         training_history.png       |  best_adapter/  (LoRA weights)

CHANGES vs previous version (all marked with "# CHANGED"):
  - HF token no longer hardcoded: read from Kaggle Secrets / env var
    (REVOKE the old leaked token at huggingface.co -> Settings -> Tokens!)
  - final evaluation rebuilt: frees the training model first (prevents the
    silent OOM kill at "Loading best checkpoint"), then loads the saved
    adapter via PeftModel.from_pretrained instead of the broken
    get_peft_model + load_adapter("default") combination
  - epochs_no_improve initialised before the training loop
  - num_workers=0, pin_memory=False (silent-kernel-death fix on Kaggle)
  - all outputs written to /kaggle/working when available
  - RAM/VRAM probes after each stage and each epoch (psutil)
  - evaluation log/report header states the actual eval file name
"""

# ── Must be set BEFORE any torch import ──────────────────────────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("transformers>=4.42.0", "accelerate", "openpyxl",
    "scikit-learn", "seaborn", "sentencepiece", "bitsandbytes", "peft", "psutil")

import gc
import warnings
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
    AutoConfig,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── CHANGED: HF token from Kaggle Secrets or environment, never hardcoded ────
# Kaggle: Add-ons -> Secrets -> add a secret named HF_TOKEN.
# The old hardcoded token was exposed: revoke it on huggingface.co!
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")  # falls back to env var, else None
# None is fine: none of the registry models are gated.

# ─────────────────────────────────────────────────────────────────────────────
#  ▶  CHANGE THIS LINE TO SWITCH MODELS — everything else auto-configures
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "BramVanroy/fietje-2b"

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {

    # ── Dutch GPT-2 (117M) ───────────────────────────────────────────────────
    "GroNLP/gpt2-small-dutch": dict(
        padding_side   = "left",
        max_len        = 128,
        batch_size     = 8,
        lr             = 2e-4,
        lora_r         = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        # GPT-2 uses Conv1D projections — target the attention projections
        target_modules = ["c_attn", "c_proj"],
        use_qlora      = True,
        accumulation   = 4,
        epochs         = 7,
    ),

    # ── Qwen 0.5B ────────────────────────────────────────────────────────────
    "Qwen/Qwen2.5-0.5B": dict(
        padding_side   = "left",
        max_len        = 128,
        batch_size     = 4,
        lr             = 2e-4,
        lora_r         = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        use_qlora      = True,
        accumulation   = 4,
        epochs         = 7,
    ),

    # ── OPT 1.3B (note: 1.3 billion — not 13B) ───────────────────────────────
    "facebook/opt-1.3b": dict(
        padding_side   = "left",
        max_len        = 128,
        batch_size     = 4,
        lr             = 1e-4,
        lora_r         = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "v_proj"],
        use_qlora      = True,
        accumulation   = 4,
        epochs         = 7,
    ),

    # ── Fietje 2B (Dutch) ────────────────────────────────────────────────────
    "BramVanroy/fietje-2b": dict(
        padding_side   = "left",
        max_len        = 128,
        batch_size     = 2,
        lr             = 1e-4,
        lora_r         = 8,       # smaller r to save VRAM
        lora_alpha     = 16,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        use_qlora      = True,
        accumulation   = 8,       # keep effective batch = 16
        epochs         = 4,
    ),
}

assert MODEL_NAME in MODEL_REGISTRY, (
    f"'{MODEL_NAME}' not in MODEL_REGISTRY.\n"
    f"Available: {list(MODEL_REGISTRY.keys())}"
)

cfg = MODEL_REGISTRY[MODEL_NAME]

# ── Unpack config ─────────────────────────────────────────────────────────────
PADDING_SIDE       = cfg["padding_side"]
MAX_LEN            = cfg["max_len"]
BATCH_SIZE         = cfg["batch_size"]
LR                 = cfg["lr"]
LORA_R             = cfg["lora_r"]
LORA_ALPHA         = cfg["lora_alpha"]
LORA_DROPOUT       = cfg["lora_dropout"]
TARGET_MODULES     = cfg["target_modules"]
USE_QLORA          = cfg["use_qlora"]
ACCUMULATION_STEPS = cfg["accumulation"]
EPOCHS             = cfg["epochs"]

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
TRAIN_FILE        = "/kaggle/input/datasets/train.xlsx"
DEV_FILE          = "/kaggle/input/datasets/dev.xlsx"

# CHANGED: write everything to /kaggle/working when on Kaggle, so outputs
# and the adapter survive background ("Save & Run All") execution.
WORK_DIR          = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR        = os.path.join(WORK_DIR, "best_adapter")
WARMUP_RATIO      = 0.15
WEIGHT_DECAY      = 0.01
CONFIDENCE_THRESH = 0.70
SEED              = 42
PATIENCE          = 2   # stop if dev_loss doesn't improve for 2 epochs

EVAL_NAME = os.path.basename(DEV_FILE)  # CHANGED: honest name in logs/reports

# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY, DEVICE & MEMORY PROBE
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CHANGED: memory probe — if RAM% climbs toward 100 in the log before a
# silent stop, the kernel was OOM-killed.
def mem_report(tag: str):
    ram = psutil.virtual_memory()
    line = f"  [mem] {tag}: RAM {ram.percent:.0f}% ({ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB)"
    if device.type == "cuda":
        line += f"  VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB"
    print(line)

print(f"▶ Model   : {MODEL_NAME}")
print(f"  QLoRA   : {USE_QLORA}  |  MAX_LEN: {MAX_LEN}  |  "
      f"Batch: {BATCH_SIZE}×{ACCUMULATION_STEPS}={BATCH_SIZE*ACCUMULATION_STEPS}")
print(f"  Eval on : {EVAL_NAME}")
print(f"  Output  : {WORK_DIR}")
if device.type == "cuda":
    print(f"  GPU     : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
mem_report("start")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ Loading data …")
train_df = pd.read_excel(TRAIN_FILE)
dev_df   = pd.read_excel(DEV_FILE)
train_df.columns = [c.strip().lower() for c in train_df.columns]
dev_df.columns   = [c.strip().lower() for c in dev_df.columns]

for df, name in [(train_df, "train"), (dev_df, "eval")]:
    assert "text"  in df.columns, f"'text' column missing in {name}"
    assert "class" in df.columns, f"'class' column missing in {name}"

train_df = train_df.dropna(subset=["text", "class"]).reset_index(drop=True)
dev_df   = dev_df.dropna(subset=["text", "class"]).reset_index(drop=True)

le = LabelEncoder()
le.fit(pd.concat([train_df["class"], dev_df["class"]], ignore_index=True))
class_names = list(le.classes_)
num_labels  = len(class_names)

train_df["label"] = le.transform(train_df["class"])
dev_df["label"]   = le.transform(dev_df["class"])

_counts       = np.bincount(train_df["label"].values, minlength=num_labels).astype(float)
_weights      = _counts.sum() / (_counts * num_labels)
class_weights = torch.tensor(_weights, dtype=torch.float32).to(device)

print(f"  Train: {len(train_df)}  Eval ({EVAL_NAME}): {len(dev_df)}  "
      f"Classes ({num_labels}): {class_names}")
print(f"  Weights: { {class_names[i]: round(_weights[i], 3) for i in range(num_labels)} }")

dev_only = set(dev_df["class"].unique()) - set(train_df["class"].unique())
if dev_only:
    print(f"  ⚠  Classes in eval but NOT in train: {sorted(dev_only)}")
mem_report("after data loading")

# ─────────────────────────────────────────────────────────────────────────────
#  TOKENISER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n▶ Loading tokeniser …")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, token=HF_TOKEN
)
tokenizer.padding_side = PADDING_SIDE

# Ensure pad token exists (GPT-style models often lack one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token set to eos_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
else:
    print(f"  pad_token='{tokenizer.pad_token}'  (id={tokenizer.pad_token_id})")

# Token-length stats
_lengths = [len(tokenizer(str(t), truncation=False)["input_ids"]) for t in train_df["text"]]
_p95 = int(np.percentile(_lengths, 95))
_pct = float(np.mean([l > MAX_LEN for l in _lengths])) * 100
print(f"  p50={int(np.percentile(_lengths, 50))}  p90={int(np.percentile(_lengths, 90))}  "
      f"p95={_p95}  p99={int(np.percentile(_lengths, 99))}")
print(f"  Truncated at MAX_LEN={MAX_LEN}: {_pct:.1f}% of train samples")
if _p95 > MAX_LEN:
    print(f"  ⚠  p95 ({_p95}) > MAX_LEN ({MAX_LEN}) → consider raising MAX_LEN")
del _lengths

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts.tolist(), labels.tolist()
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# CHANGED: num_workers=0, pin_memory=False — worker processes duplicate the
# dataset in RAM on Kaggle/Colab and are the most common silent-hang source.
train_loader = DataLoader(
    TextDataset(train_df["text"], train_df["label"], tokenizer, MAX_LEN),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False,
)
dev_loader = DataLoader(
    TextDataset(dev_df["text"], dev_df["label"], tokenizer, MAX_LEN),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n▶ Loading model ({'4-bit QLoRA' if USE_QLORA else 'full precision'}) …")

model_config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    trust_remote_code=True,
    token=HF_TOKEN,
)
model_config.pad_token_id = tokenizer.pad_token_id

bnb_config = None
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

load_kwargs = dict(
    config=model_config,
    device_map={"": 0},
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
    token=HF_TOKEN,
)
if bnb_config:
    load_kwargs["quantization_config"] = bnb_config

base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **load_kwargs)

# The KV cache is useless during training and was implicated in the
# gradient checkpointing — it changes how many tensors the recomputed
# forward pass saves ("A different number of tensors was saved ...").
# The cache is useless during training anyway.
base_model.config.use_cache = False

if USE_QLORA:
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=False
    )

# Sync pad token id into model config
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

total_steps  = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"  Optimiser steps: {total_steps}  (warmup: {warmup_steps})")

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)

# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_confs, all_probs = [], [], [], []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss    += loss_fn(outputs.logits.float(), labels).item()
            probs          = torch.softmax(outputs.logits.float(), dim=1)
            top_probs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(top_probs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confs),
        np.array(all_probs),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ Training …\n" + "─" * 60)
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

best_dev_loss     = float("inf")
epochs_no_improve = 0            # CHANGED: initialised before the loop
history           = []
loss_fn_train     = torch.nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs       = model(input_ids=input_ids, attention_mask=attention_mask)
        weighted_loss = loss_fn_train(outputs.logits.float(), labels)
        loss          = weighted_loss / ACCUMULATION_STEPS

        if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
            raise RuntimeError(
                f"NaN/Inf loss at epoch {epoch} step {step}. "
                f"Logits: min={outputs.logits.min().item():.2f} "
                f"max={outputs.logits.max().item():.2f}"
            )

        loss.backward()
        total_train_loss += weighted_loss.item()

        if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 20 == 0 or step == len(train_loader):
            print(f"  Epoch {epoch}/{EPOCHS}  Step {step:>4}/{len(train_loader)}  "
                  f"loss={total_train_loss/step:.4f}", end="\r")

    avg_train = total_train_loss / len(train_loader)
    dev_loss, preds, labels_arr, _, _ = evaluate(model, dev_loader)
    accuracy = (preds == labels_arr).mean()

    history.append({"epoch": epoch, "train_loss": avg_train,
                    "dev_loss": dev_loss, "accuracy": accuracy})
    print(f"\n  Epoch {epoch}/{EPOCHS}  train={avg_train:.4f}  "
          f"dev={dev_loss:.4f}  acc={accuracy:.4f}")
    mem_report(f"after epoch {epoch}")

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        epochs_no_improve = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ✓ Saved (dev_loss={dev_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch}")
            break

    torch.cuda.empty_cache()

print("\n" + "─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
# CHANGED: free the training model FIRST (prevents holding two copies of a
# 2B model in memory — the silent OOM kill at exactly this point), then load
# the saved adapter the canonical way with PeftModel.from_pretrained.
print("\n▶ Loading best checkpoint for final evaluation …")
del model, base_model, optimizer, scheduler
gc.collect()
torch.cuda.empty_cache()
mem_report("after freeing training model")

base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **load_kwargs)
base.config.pad_token_id = tokenizer.pad_token_id
best_model = PeftModel.from_pretrained(base, OUTPUT_DIR, is_trainable=False)
best_model.eval()
mem_report("after loading best checkpoint")

_, final_preds, final_labels, final_confs, final_probs = evaluate(best_model, dev_loader)

# ── Classification report ─────────────────────────────────────────────────────
print(f"\n▶ Classification report ({EVAL_NAME}):\n")
present_ids   = sorted(set(final_labels.tolist()) | set(final_preds.tolist()))
present_names = [class_names[i] for i in present_ids]
report = classification_report(
    final_labels, final_preds,
    labels=present_ids, target_names=present_names, digits=4, zero_division=0,
)
print(report)
with open(os.path.join(WORK_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Classification Report — {MODEL_NAME}  (eval file: {EVAL_NAME})\n"
            + "=" * 60 + "\n\n" + report)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(final_labels, final_preds, labels=present_ids)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names,
            ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)
ax.set_title(f"Confusion Matrix — {MODEL_NAME.split('/')[-1]}", fontsize=14)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ── Misclassifications ────────────────────────────────────────────────────────
mis_mask    = final_preds != final_labels
mis_probs   = final_probs[mis_mask]
sec_ids     = np.argsort(mis_probs, axis=1)[:, -2]
misclass_df = pd.DataFrame({
    "text":            dev_df["text"].values[mis_mask],
    "true_class":      le.inverse_transform(final_labels[mis_mask]),
    "predicted_class": le.inverse_transform(final_preds[mis_mask]),
    "confidence":      final_confs[mis_mask].round(4),
    "2nd_best_class":  [class_names[i] for i in sec_ids],
    "2nd_best_prob":   mis_probs[np.arange(len(mis_probs)), sec_ids].round(4),
    "uncertain":       final_confs[mis_mask] < CONFIDENCE_THRESH,
})
misclass_df.sort_values(["uncertain", "confidence"], ascending=[False, True]) \
           .to_excel(os.path.join(WORK_DIR, "misclassifications.xlsx"), index=False)

# ── Low-confidence predictions ────────────────────────────────────────────────
lc_mask   = final_confs < CONFIDENCE_THRESH
lc_probs  = final_probs[lc_mask]
lc_sec    = np.argsort(lc_probs, axis=1)[:, -2]
lowconf_df = pd.DataFrame({
    "text":            dev_df["text"].values[lc_mask],
    "true_class":      le.inverse_transform(final_labels[lc_mask]),
    "predicted_class": le.inverse_transform(final_preds[lc_mask]),
    "confidence":      final_confs[lc_mask].round(4),
    "2nd_best_class":  [class_names[i] for i in lc_sec],
    "2nd_best_prob":   lc_probs[np.arange(len(lc_probs)), lc_sec].round(4),
    "correct":         (final_preds == final_labels)[lc_mask],
})
lowconf_df.sort_values("confidence") \
          .to_excel(os.path.join(WORK_DIR, "low_confidence_predictions.xlsx"), index=False)
n_unc = lc_mask.sum()

# ── Training history ──────────────────────────────────────────────────────────
hdf = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hdf["epoch"], hdf["train_loss"], marker="o", label="Train")
axes[0].plot(hdf["epoch"], hdf["dev_loss"],   marker="o", label="Eval")
axes[0].set_title("Loss per epoch"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].plot(hdf["epoch"], hdf["accuracy"], marker="o", color="green")
axes[1].set_title(f"Accuracy ({EVAL_NAME})"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "training_history.png"), dpi=150)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
overall_acc = (final_preds == final_labels).mean()
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Model            : {MODEL_NAME}")
print(f"  Mode             : {'QLoRA 4-bit' if USE_QLORA else 'Full fine-tune'}")
print(f"  Eval file        : {EVAL_NAME}  ({len(dev_df)} samples)")
print(f"  Overall accuracy : {overall_acc:.4f}")
print(f"  Misclassified    : {mis_mask.sum()} ({mis_mask.sum()/len(dev_df)*100:.1f}%)")
print(f"  Low-confidence   : {n_unc} ({n_unc/len(dev_df)*100:.1f}%)")
print(f"  Best dev loss    : {best_dev_loss:.4f}")
mem_report("end")
print(f"\nOutput files (in {WORK_DIR}):")
for f in ["classification_report.txt", "confusion_matrix.png",
          "misclassifications.xlsx", "low_confidence_predictions.xlsx",
          "training_history.png", OUTPUT_DIR + "/"]:
    print(f"  • {f}")
print("=" * 60)