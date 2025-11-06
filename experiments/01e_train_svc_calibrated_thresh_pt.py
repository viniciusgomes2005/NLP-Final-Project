# experiments/01e_train_svc_calibrated_thresh_pt.py
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.metrics import compute_metrics, save_json, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="por")
    ap.add_argument("--word_min_df", type=int, default=1)
    ap.add_argument("--word_max_df", type=float, default=0.90)
    ap.add_argument("--char_min_df", type=int, default=1)
    ap.add_argument("--char_ng_min", type=int, default=3)
    ap.add_argument("--char_ng_max", type=int, default=7)
    ap.add_argument("--C", type=float, default=3.0)
    ap.add_argument("--recall_floor", type=float, default=0.60)
    ap.add_argument("--dedup_train", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    df = df[df["text"].astype(str).str.strip()!=""]
    df_lang = filter_lang(df, args.lang)
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)
    if args.dedup_train:
        splits.train = splits.train.drop_duplicates(subset=["text","label"]).reset_index(drop=True)

    labels_sorted = sorted(df_lang["label"].unique().tolist())
    pos_label = [l for l in labels_sorted if "bully" in l][0]
    neg_label = [l for l in labels_sorted if l != pos_label][0]

    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1,2),
            min_df=args.word_min_df, max_df=args.word_max_df,
            sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(args.char_ng_min, args.char_ng_max),
            min_df=args.char_min_df, lowercase=True))
    ])

    base = LinearSVC(C=args.C, class_weight="balanced")
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    pipe = Pipeline([("feats", feats), ("clf", clf)])
    pipe.fit(splits.train["text"], splits.train["label"])

    # proba via calibrado
    idx_pos = list(pipe.named_steps["clf"].classes_).index(pos_label)
    y_val_bin = (splits.val["label"]==pos_label).astype(int).values
    probs_val = pipe.predict_proba(splits.val["text"])[:, idx_pos]

    # threshold com piso de recall
    best_t, best_score, best_stats = 0.5, -1.0, None
    for t in np.linspace(0.1,0.9,81):
        y_hat = (probs_val>=t).astype(int)
        p,r,f,s = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
        f_macro = (f[0]+f[1])/2.0
        if r[1] >= args.recall_floor and f_macro > best_score:
            best_t, best_score = t, f_macro
            best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}
    if best_score < 0:
        for t in np.linspace(0.1,0.9,81):
            y_hat = (probs_val>=t).astype(int)
            p,r,f,s = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
            f_macro = (f[0]+f[1])/2.0
            if f_macro > best_score:
                best_t, best_score = t, f_macro
                best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}

    probs_test = pipe.predict_proba(splits.test["text"])[:, idx_pos]
    y_test_bin = (splits.test["label"]==pos_label).astype(int).values
    y_hat_test_bin = (probs_test>=best_t).astype(int)
    y_hat_test_lbl = np.where(y_hat_test_bin==1, pos_label, neg_label)
    mets_test = compute_metrics(splits.test["label"], y_hat_test_lbl)
    save_json(mets_test, out / f"{args.lang}_test_metrics.json")

    y_hat_val_bin = (probs_val>=best_t).astype(int)
    y_hat_val_lbl = np.where(y_hat_val_bin==1, pos_label, neg_label)
    mets_val = compute_metrics(splits.val["label"], y_hat_val_lbl)
    save_json(mets_val, out / f"{args.lang}_val_metrics.json")

    plot_confusion(splits.val["label"], y_hat_val_lbl, labels_sorted, out / f"{args.lang}_val_cm.png",
                   title=f"{args.lang} val — SVM Calibrado (thr={best_t:.2f})")
    plot_confusion(splits.test["label"], y_hat_test_lbl, labels_sorted, out / f"{args.lang}_test_cm.png",
                   title=f"{args.lang} test — SVM Calibrado (thr={best_t:.2f})")

    info = {
        "best_threshold": float(best_t),
        "val_macro_f1_at_best_t": float(best_score),
        "val_per_class_stats_at_best_t": best_stats,
        "params": {
            "word_min_df": args.word_min_df, "word_max_df": args.word_max_df,
            "char_min_df": args.char_min_df, "char_ng": [args.char_ng_min, args.char_ng_max],
            "C": args.C, "recall_floor": args.recall_floor, "dedup_train": args.dedup_train
        }
    }
    with open(out/"selection_info.json","w",encoding="utf-8") as f:
        json.dump(info,f,indent=2,ensure_ascii=False)
    import joblib; joblib.dump(pipe, out / f"{args.lang}_best_model.joblib")
    print(f"[OK] SVM Calibrado + thr {best_t:.2f} → {out}")
    # salvar CSV com predições do TEST
    import pandas as pd
    sel = json.load(open(out / "selection_info.json"))
    best_t = sel["best_threshold"]

    # índice da classe positiva, a partir do pipeline calibrado
    idx_pos = list(pipe.named_steps["clf"].classes_).index(pos_label)

    # probs no TEST via PIPE (vetorização + modelo)
    probs_test = pipe.predict_proba(splits.test["text"])[:, idx_pos]
    y_hat_test_bin = (probs_test >= best_t).astype(int)
    y_hat_test_lbl = np.where(y_hat_test_bin==1, pos_label, neg_label)

if __name__ == "__main__":
    main()
