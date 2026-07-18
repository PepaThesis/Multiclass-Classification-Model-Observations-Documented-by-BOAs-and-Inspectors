# Train an SVM to classify texts into:
# ["Acting","Assisting","Communicating","Cooperating","Coping",
#  "Informing","Investigating","Networking","Observing"]
#
# No sklearn Pipeline â steps are explicit:
#   1. TF-IDF vectorization
#   2. Chi-squared feature selection (multiple k values compared)
#   3. LinearSVC training & evaluation

import os
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

RANDOM_STATE = 42

# K values to compare; "all" keeps every feature
K_VALUES = [10_000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dutch_stopwords():
    try:
        return stopwords.words("dutch")
    except LookupError:
        nltk.download("stopwords")
        return stopwords.words("dutch")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'")

    df = pd.read_excel(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    if "text" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "text"})

    required = {"text", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "class"]).copy()
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .apply(lambda s: " ".join(s.split()))
    )
    df["class"] = df["class"].astype(str).str.strip()
    return df


def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_path, title, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=True,
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def summary_metrics(y_true, y_pred, labels) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    f_mic = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)

    return dict(
        accuracy=acc,
        macro_precision=p_mac, macro_recall=r_mac, macro_f1=f_mac,
        weighted_precision=p_w, weighted_recall=r_w, weighted_f1=f_w,
        micro_f1=f_mic,
    )


def print_summary(metrics: dict):
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Macro  P/R/F1      : {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"  Weighted P/R/F1    : {metrics['weighted_precision']:.4f} / {metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}")
    print(f"  Micro  F1          : {metrics['micro_f1']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ââ Load data ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    train_df = load_data("train.xlsx")
    print(f"[Info] Loaded {len(train_df)} rows from train.xlsx")

    dev_df = load_data("dev.xlsx")
    print(f"[Info] Loaded {len(dev_df)} rows from dev.xlsx")

    print("\n[Label distribution - train]")
    print(train_df["class"].value_counts())
    print("\n[Label distribution - dev]")
    print(dev_df["class"].value_counts())

    labels_sorted = sorted(
        set(train_df["class"].unique()).union(set(dev_df["class"].unique()))
    )

    X_train_raw = train_df["text"].values
    y_train     = train_df["class"].values
    X_dev_raw   = dev_df["text"].values
    y_true      = dev_df["class"].values

    # ââ Step 1: TF-IDF vectorisation (fit on train only) ââââââââââââââââââ
    print("\n[Step 1] Fitting TF-IDF vectoriser âŠ")
    #stoplist = get_dutch_stopwords()

    tfidf = TfidfVectorizer(
        lowercase=True,
        #stop_words=stoplist,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=100_000,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_dev_tfidf   = tfidf.transform(X_dev_raw)
    print(f"  TF-IDF shape (train): {X_train_tfidf.shape}")

    # ââ Step 2 + 3: Chi-squared selection âââââââââââââââ
    comparison_rows = []

    for k in K_VALUES:
        k_label = str(k)
        n_features = X_train_tfidf.shape[1]
        actual_k = n_features if k == "all" else min(k, n_features)
        effective_k = "all" if actual_k == n_features else actual_k
        print(f"\n{'='*60}")
        print(f"[Chi2 k={k_label}]  selecting {actual_k} features âŠ")

        # Feature selection
        selector = SelectKBest(chi2, k=effective_k)
        X_train_sel = selector.fit_transform(X_train_tfidf, y_train)
        X_dev_sel   = selector.transform(X_dev_tfidf)
        print(f"  Selected shape (train): {X_train_sel.shape}")

        # SVM
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_dev_sel)

        # Metrics
        metrics = summary_metrics(y_true, y_pred, labels_sorted)
        print_summary(metrics)

        print(f"\n  Classification report (k={k_label}):\n")
        print(classification_report(
            y_true, y_pred, labels=labels_sorted, digits=4, zero_division=0
        ))

        # Confusion matrices
        safe_k = k_label.replace(",", "")
        plot_and_save_confusion_matrix(
            y_true, y_pred, labels_sorted,
            out_path=f"confusion_matrix_k{safe_k}.png",
            title=f"SVM Confusion Matrix â counts  (k={k_label})",
        )
        plot_and_save_confusion_matrix(
            y_true, y_pred, labels_sorted,
            out_path=f"confusion_matrix_k{safe_k}_norm.png",
            title=f"SVM Confusion Matrix â row-normalised  (k={k_label})",
            normalize="true",
        )

        # Save best model for later error analysis
        row = {"k": k_label, "n_features": actual_k, **metrics}
        comparison_rows.append(row)

        # Store artefacts of this run for potential reuse
        comparison_rows[-1]["_tfidf"]    = tfidf
        comparison_rows[-1]["_selector"] = selector
        comparison_rows[-1]["_clf"]      = clf
        comparison_rows[-1]["_y_pred"]   = y_pred

    # ââ Comparison table âââââââââââââââââââââââââââââââââââââââââââââââââââ
    print(f"\n{'='*60}")
    print("[Comparison across k values]")
    display_cols = ["k", "n_features", "accuracy", "macro_f1", "weighted_f1", "micro_f1"]
    comp_df = pd.DataFrame(
        [{c: r[c] for c in display_cols} for r in comparison_rows]
    )
    print(comp_df.to_string(index=False))

    # Save comparison to CSV
    comp_df.to_csv("chi2_k_comparison.csv", index=False)
    print("\n[Saved] chi2_k_comparison.csv")

    # ââ Pick best k by macro F1 ââââââââââââââââââââââââââââââââââââââââââââ
    best_row = max(comparison_rows, key=lambda r: r["macro_f1"])
    best_k   = best_row["k"]
    print(f"\n[Best k by macro F1] k={best_k}  (macro F1 = {best_row['macro_f1']:.4f})")

    y_pred_best   = best_row["_y_pred"]
    tfidf_best    = best_row["_tfidf"]
    selector_best = best_row["_selector"]
    clf_best      = best_row["_clf"]

    # ââ Error analysis for best k ââââââââââââââââââââââââââââââââââââââââââ
    errors = pd.DataFrame({
        "text":            dev_df["text"].values,
        "true_label":      y_true,
        "predicted_label": y_pred_best,
    })
    errors = errors[errors["true_label"] != errors["predicted_label"]].copy()

    print(f"\n[Error analysis â k={best_k}]  "
          f"{len(errors)} misclassified out of {len(dev_df)} examples.")
    print(errors.head(20))

    confusion_pairs = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print("\n[Top confusions]")
    print(confusion_pairs.head(20))

    ct = (
        pd.crosstab(errors["true_label"], errors["predicted_label"])
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    print("\n[Confusion table of errors only] (rows=true, cols=predicted)")
    print(ct)

    with pd.ExcelWriter("misclassified_examples.xlsx", engine="openpyxl") as writer:
        errors.to_excel(writer,          index=False, sheet_name="row_errors")
        confusion_pairs.to_excel(writer, index=False, sheet_name="top_confusions")
        ct.to_excel(writer,                           sheet_name="error_matrix")
        comp_df.to_excel(writer,         index=False, sheet_name="k_comparison")
    print("[Saved] misclassified_examples.xlsx")

    # ââ Persist best model âââââââââââââââââââââââââââââââââââââââââââââââââ
    joblib.dump(
        {"tfidf": tfidf_best, "selector": selector_best, "clf": clf_best},
        "svm_text_model.joblib",
    )
    print("[Saved] svm_text_model.joblib  (tfidf + selector + clf for best k)")


if __name__ == "__main__":
    main()
