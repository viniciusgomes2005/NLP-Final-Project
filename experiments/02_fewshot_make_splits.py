from pathlib import Path
import argparse
from core.data_utils import load_csv, filter_lang, stratified_splits_lang, make_k_shot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to full dataset")
    ap.add_argument("--lang", default="por", help="3-letter language code")
    ap.add_argument("--k", type=int, default=16, help="k-shot per class (train)")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    df_lang = filter_lang(df, args.lang)
    splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)

    train_k = make_k_shot(splits.train, k=args.k, seed=42)
    train_k.to_csv(out / f"{args.lang}_k{args.k}_train.csv", index=False)
    splits.val.to_csv(out / f"{args.lang}_val.csv", index=False)
    splits.test.to_csv(out / f"{args.lang}_test.csv", index=False)
    print("[OK] few-shot splits at", out)

if __name__ == "__main__":
    main()
