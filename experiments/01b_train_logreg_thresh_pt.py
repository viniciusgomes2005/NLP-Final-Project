# experiments/01b_train_logreg_thresh_pt.py
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import argparse

# permite "from core..." funcionar quando rodar com -m
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.metrics import compute_metrics, save_json, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="por")
    # hiperparâmetros rápidos por CLI
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) dados e splits
    df = load_csv(args.csv)
    df_lang = filter_lang(df, args.lang)
    assert len(df_lang) > 0, f"Sem linhas para {args.lang}"
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)

    # identificar qual rótulo é o "positivo" (contém 'bully')
    labels_sorted = sorted(df_lang["label"].unique().tolist())
    pos_label = None
    for lab in labels_sorted:
        if "bully" in lab:
            pos_label = lab
            break
    assert pos_label is not None, "Não encontrei label positivo contendo 'bully' (ex.: por_bully)."
    neg_label = [l for l in labels_sorted if l != pos_label][0]

    # 2) pipeline com LogReg (probabilidades) + TF-IDF de palavras (1–2)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode")),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            C=args.C))
    ])

    # 3) treinar
    pipe.fit(splits.train["text"], splits.train["label"])

    # 4) escolher threshold no VAL maximizando Macro-F1
    # mapeia rótulos para binário 0/1 (1 = pos_label)
    y_val_bin = (splits.val["label"] == pos_label).astype(int).values
    probs_val = pipe.predict_proba(splits.val["text"])[:, list(pipe.classes_).index(pos_label)]

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (probs_val >= t).astype(int)
        f1 = f1_score(y_val_bin, y_hat, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # 5) avaliar no TEST usando o threshold escolhido
    y_test_bin = (splits.test["label"] == pos_label).astype(int).values
    probs_test = pipe.predict_proba(splits.test["text"])[:, list(pipe.classes_).index(pos_label)]
    y_hat_test_bin = (probs_test >= best_t).astype(int)

    # voltar 0/1 → rótulos string para métricas "completas"
    y_hat_test_lbl = np.where(y_hat_test_bin == 1, pos_label, neg_label)

    mets_test = compute_metrics(splits.test["label"], y_hat_test_lbl)
    save_json(mets_test, out / f"{args.lang}_test_metrics.json")

    # também salvamos métricas do VAL (com threshold ótimo), útil para o relatório
    y_hat_val_bin = (probs_val >= best_t).astype(int)
    y_hat_val_lbl = np.where(y_hat_val_bin == 1, pos_label, neg_label)
    mets_val = compute_metrics(splits.val["label"], y_hat_val_lbl)
    save_json(mets_val, out / f"{args.lang}_val_metrics.json")

    # plot confusions
    plot_confusion(splits.val["label"], y_hat_val_lbl, labels_sorted, out / f"{args.lang}_val_cm.png",
                   title=f"{args.lang} val — LogReg TFIDF (thr={best_t:.2f})")
    plot_confusion(splits.test["label"], y_hat_test_lbl, labels_sorted, out / f"{args.lang}_test_cm.png",
                   title=f"{args.lang} test — LogReg TFIDF (thr={best_t:.2f})")

    # salvar threshold escolhido e hiperparâmetros
    info = {
        "best_threshold": float(best_t),
        "val_macro_f1_at_best_t": float(best_f1),
        "pos_label": pos_label,
        "neg_label": neg_label,
        "params": {
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "C": args.C
        }
    }
    with open(out / "selection_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # salvar modelo
    import joblib
    joblib.dump(pipe, out / f"{args.lang}_best_model.joblib")
    print(f"[OK] treinado com LogReg + threshold @ {best_t:.2f} → {out}")

if __name__ == "__main__":
    main()
