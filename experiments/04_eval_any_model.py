from pathlib import Path
import argparse
import joblib
from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.metrics import compute_metrics, save_json, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lang", default="por")
    ap.add_argument("--model", required=True, help=".joblib saved by baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv)
    df_lang = filter_lang(df, args.lang)
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)
    model = joblib.load(args.model)

    for name, part in [("val", splits.val), ("test", splits.test)]:
        y_pred = model.predict(part["text"])
        mets = compute_metrics(part["label"], y_pred)
        save_json(mets, out / f"{args.lang}_{name}_metrics.json")
        labels_sorted = sorted(df_lang["label"].unique().tolist())
        plot_confusion(part["label"], y_pred, labels_sorted, out / f"{args.lang}_{name}_cm.png",
                       title=f"{args.lang} {name} — EVAL")
    print("[OK] Evaluation written to", out)

if __name__ == "__main__":
    main()
