from pathlib import Path
import argparse
from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.classic import ClassicConfig, build_pipeline, save_model
from core.metrics import compute_metrics, save_json, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to fewshot_dataset.csv")
    ap.add_argument("--out", required=True, help="output dir, e.g., runs/pt_baseline")
    ap.add_argument("--lang", default="por", help="3-letter lang code (por, eng, hin, ...)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    df_lang = filter_lang(df, args.lang)
    assert len(df_lang) > 0, f"No rows for language {args.lang}"
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)

    cfg = ClassicConfig(ngram_range=(3,5), min_df=2, C=1.0, calibrated=False)  # char n-grams
    model = build_pipeline(cfg)
    model.fit(splits.train['text'], splits.train['label'])

    for split_name, part in [("val", splits.val), ("test", splits.test)]:
        y_pred = model.predict(part['text'])
        mets = compute_metrics(part['label'], y_pred)
        save_json(mets, out / f"{args.lang}_{split_name}_metrics.json")
        labels_sorted = sorted(df_lang['label'].unique().tolist())
        plot_confusion(part['label'], y_pred, labels_sorted, out / f"{args.lang}_{split_name}_cm.png",
                       title=f"{args.lang} {split_name} — TFIDF+LinearSVC")

    save_model(model, out / f"{args.lang}_best_model.joblib")
    print("[OK] trained baseline for", args.lang, "→", out)

if __name__ == "__main__":
    main()
