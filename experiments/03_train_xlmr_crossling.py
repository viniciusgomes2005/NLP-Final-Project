import argparse
from pathlib import Path
import pandas as pd

def load_lang(df, lang):
    return df[df['lang']==lang].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to full dataset")
    ap.add_argument("--pt_fewshot", required=True, help="CSV with PT few-shot train (from 02 script)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--eng_lang", default="eng", help="english code")
    ap.add_argument("--pt_lang", default="por", help="portuguese code")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        import evaluate
    except Exception as e:
        print("Transformers/Datasets not installed; cannot run XLM-R. Install extras in requirements.")
        return

    df_full = pd.read_csv(args.csv)
    df_full["lang"] = df_full["label"].astype(str).str.split("_").str[0]
    df_eng = load_lang(df_full, args.eng_lang)
    df_pt  = load_lang(df_full, args.pt_lang)
    df_pt_few = pd.read_csv(args.pt_fewshot)

    from sklearn.model_selection import train_test_split
    X_val, X_test, y_val, y_test = train_test_split(df_pt["text"], df_pt["label"], test_size=0.5, random_state=42, stratify=df_pt["label"])
    df_pt_val  = pd.DataFrame({"text": X_val, "label": y_val})
    df_pt_test = pd.DataFrame({"text": X_test, "label": y_test})

    labels = sorted(pd.concat([df_eng["label"], df_pt["label"]]).unique().tolist())
    label2id = {lab:i for i, lab in enumerate(labels)}
    id2label = {i:lab for lab,i in label2id.items()}

    def remap(df):
        return {"text": df["text"].tolist(), "label": [label2id[x] for x in df["label"].tolist()]}

    model_name = "xlm-roberta-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    ds_train = Dataset.from_dict(remap(pd.concat([df_eng[["text","label"]], df_pt_few[["text","label"]]]))).map(
        lambda b: tok(b["text"], truncation=True, padding=True, max_length=256), batched=True
    )
    ds_val = Dataset.from_dict(remap(df_pt_val)).map(lambda b: tok(b["text"], truncation=True, padding=True, max_length=256), batched=True)
    ds_test = Dataset.from_dict(remap(df_pt_test)).map(lambda b: tok(b["text"], truncation=True, padding=True, max_length=256), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

    acc = evaluate.load("accuracy")
    f1m = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels_ids = eval_pred
        preds = logits.argmax(-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels_ids)["accuracy"],
            "macro_f1": f1m.compute(predictions=preds, references=labels_ids, average="macro")["f1"]
        }

    args_tr = TrainingArguments(
        output_dir=str(out / "xlmr_out"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=compute_metrics
    )
    trainer.train()
    metrics_val = trainer.evaluate()
    with open(out/"val_metrics.json","w") as f:
        import json; json.dump(metrics_val, f, indent=2)
    metrics_test = trainer.evaluate(ds_test)
    with open(out/"test_metrics.json","w") as f:
        import json; json.dump(metrics_test, f, indent=2)
    print("[OK] XLM-R trained/evaluated. See", out)

if __name__ == "__main__":
    main()
