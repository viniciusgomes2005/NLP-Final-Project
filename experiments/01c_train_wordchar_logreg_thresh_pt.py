# experiments/01c_train_wordchar_logreg_thresh_pt.py
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
import argparse

# permite imports "core.*" quando rodar com -m
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.metrics import compute_metrics, save_json, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="por")

    # hiperparâmetros de features
    ap.add_argument("--word_min_df", type=int, default=1)
    ap.add_argument("--word_max_df", type=float, default=0.95)
    ap.add_argument("--char_min_df", type=int, default=2)
    ap.add_argument("--char_ng_min", type=int, default=3)
    ap.add_argument("--char_ng_max", type=int, default=6)

    # modelo
    ap.add_argument("--C", type=float, default=2.0)

    # treino/seleção
    ap.add_argument("--oversample", action="store_true",
                    help="Faz oversampling APENAS no treino (RandomOverSampler).")
    ap.add_argument("--recall_floor", type=float, default=0.70,
                    help="Piso de recall para a classe positiva (minoria) ao escolher o threshold.")
    ap.add_argument("--dedup_train", action="store_true",
                    help="Remove duplicatas (texto+label) apenas do treino.")

    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) dados
    df = load_csv(args.csv)
    df = df[df["text"].astype(str).str.strip() != ""]  # drop vazios globais (ok)
    df_lang = filter_lang(df, args.lang)
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)

    # (opcional) dedup só no treino
    if args.dedup_train:
        before = len(splits.train)
        splits.train = splits.train.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
        print(f"[dedup] treino: {before} -> {len(splits.train)}")

    labels_sorted = sorted(df_lang["label"].unique().tolist())
    pos_label = [l for l in labels_sorted if "bully" in l][0]
    neg_label = [l for l in labels_sorted if l != pos_label][0]

    # 2) features: word(1–2) + char_wb(3–6)
    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=args.word_min_df, max_df=args.word_max_df,
            sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(args.char_ng_min, args.char_ng_max),
            min_df=args.char_min_df, lowercase=True))
    ])

    # 3) classificador (se oversample, NÃO usar class_weight)
    clf_kwargs = {"max_iter": 1000, "solver": "liblinear", "C": args.C}
    if not args.oversample:
        clf_kwargs["class_weight"] = "balanced"

    pipe = Pipeline([
        ("feats", feats),
        ("clf", LogisticRegression(**clf_kwargs))
    ])

    # 4) treino (com ou sem oversampling)
    X_tr = splits.train["text"]
    y_tr = splits.train["label"]

    if args.oversample:
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError:
            raise SystemExit("Instale primeiro: pip install imbalanced-learn")
        X_tr_df = splits.train[["text"]].copy()
        ros = RandomOverSampler(random_state=42)
        X_tr_res, y_tr_res = ros.fit_resample(X_tr_df, y_tr)
        X_tr = X_tr_res["text"].tolist()
        y_tr = y_tr_res.tolist()
        print(f"[oversample] treino equilibrado: {len(y_tr)} amostras")

    pipe.fit(X_tr, y_tr)

    # 5) escolher threshold no VAL
    # binário: 1 = pos_label
    y_val_bin = (splits.val["label"] == pos_label).astype(int).values
    idx_pos = list(pipe.named_steps["clf"].classes_).index(pos_label)
    probs_val = pipe.predict_proba(splits.val["text"])[:, idx_pos]

    best_t, best_score = 0.5, -1.0
    best_stats = None
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (probs_val >= t).astype(int)
        # métricas por classe [0,1]
        p, r, f, s = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
        f_macro = (f[0] + f[1]) / 2.0
        # exige piso de recall da classe positiva (1)
        if r[1] >= args.recall_floor and f_macro > best_score:
            best_score, best_t = f_macro, t
            best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}

    # fallback: se nenhum threshold bateu o piso, escolha pelo melhor Macro-F1
    if best_score < 0:
        for t in np.linspace(0.1, 0.9, 81):
            y_hat = (probs_val >= t).astype(int)
            p, r, f, s = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
            f_macro = (f[0] + f[1]) / 2.0
            if f_macro > best_score:
                best_score, best_t = f_macro, t
                best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}

    # 6) avaliação no TEST com o threshold escolhido
    probs_test = pipe.predict_proba(splits.test["text"])[:, idx_pos]
    y_test_bin = (splits.test["label"] == pos_label).astype(int).values
    y_hat_test_bin = (probs_test >= best_t).astype(int)
    y_hat_test_lbl = np.where(y_hat_test_bin == 1, pos_label, neg_label)

    mets_test = compute_metrics(splits.test["label"], y_hat_test_lbl)
    save_json(mets_test, out / f"{args.lang}_test_metrics.json")

    # Val com threshold (para relatório)
    y_hat_val_bin = (probs_val >= best_t).astype(int)
    y_hat_val_lbl = np.where(y_hat_val_bin == 1, pos_label, neg_label)
    mets_val = compute_metrics(splits.val["label"], y_hat_val_lbl)
    save_json(mets_val, out / f"{args.lang}_val_metrics.json")

    # gráficos
    plot_confusion(splits.val["label"], y_hat_val_lbl, labels_sorted, out / f"{args.lang}_val_cm.png",
                   title=f"{args.lang} val — Word+Char LogReg (thr={best_t:.2f})")
    plot_confusion(splits.test["label"], y_hat_test_lbl, labels_sorted, out / f"{args.lang}_test_cm.png",
                   title=f"{args.lang} test — Word+Char LogReg (thr={best_t:.2f})")

    # info da seleção
    info = {
        "best_threshold": float(best_t),
        "val_macro_f1_at_best_t": float(best_score),
        "val_per_class_stats_at_best_t": best_stats,
        "params": {
            "word_min_df": args.word_min_df, "word_max_df": args.word_max_df,
            "char_min_df": args.char_min_df, "char_ng": [args.char_ng_min, args.char_ng_max],
            "C": args.C,
            "oversample": args.oversample,
            "recall_floor": args.recall_floor,
            "dedup_train": args.dedup_train
        }
    }
    with open(out / "selection_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    import joblib
    joblib.dump(pipe, out / f"{args.lang}_best_model.joblib")
    print(f"[OK] Word+Char LogReg + thr {best_t:.2f} → {out}")

if __name__ == "__main__":
    main()
