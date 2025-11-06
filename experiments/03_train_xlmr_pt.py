# experiments/03_train_xlmr_pt.py
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_utils import load_csv, filter_lang, stratified_splits_lang
from core.metrics import compute_metrics, save_json, plot_confusion

import torch
from torch.utils.data import Dataset
from torch import nn

import evaluate
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)

# ---------------- util ----------------
def normalize_text(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", "__url__", s)
    s = re.sub(r"@\w+", "__user__", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class TxtDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        # NÃO converter para tensor aqui; collator fará o padding/tensorização
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False
        )
        enc["labels"] = self.labels[i]  # int
        return enc

class WeightedWrapper(nn.Module):
    """
    Envolve o modelo HF e aplica CrossEntropy com pesos de classe
    de forma compatível com versões diferentes do Trainer.
    """
    def __init__(self, base_model: nn.Module, class_weight: torch.Tensor):
        super().__init__()
        self.base = base_model
        self.register_buffer("class_weight", class_weight.float())
    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        out = self.base(**kwargs)  # produz .logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weight.to(out.logits.device))
            loss = loss_fct(out.logits, labels)
            out.loss = loss
        return out

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="por")
    ap.add_argument("--model", default="xlm-roberta-base")  # use "distilbert-base-multilingual-cased" se memória apertar
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--recall_floor", type=float, default=0.60)
    ap.add_argument("--use_multilingual", action="store_true",
                    help="Mistura ENG/SPA com PT no treino (val/test seguem PT).")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- dados ----
    df = load_csv(args.csv)
    df_pt = filter_lang(df, args.lang)
    splits = stratified_splits_lang(df_pt, test_size=0.15, val_size=0.15, seed=42)

    if args.use_multilingual:
        # mistura ENG/SPA->binário PT no treino (val/test PT)
        def map_bin(lbl: str) -> str:
            return "por_bully" if "bully" in lbl else "por_nonbully"
        df_ml = df[df["label"].astype(str).str.startswith(("eng_", "spa_", "por_"))].copy()
        tr_extra = df_ml[~df_ml["label"].astype(str).str.startswith("por_")].copy()
        tr_extra["label"] = tr_extra["label"].apply(map_bin)
        train_df = splits.train.copy()
        train_df["text"] = train_df["text"].map(normalize_text)
        tr_extra["text"] = tr_extra["text"].map(normalize_text)
        train_df = pd.concat([train_df[["text","label"]], tr_extra[["text","label"]]], ignore_index=True)
    else:
        train_df = splits.train.copy()
        train_df["text"] = train_df["text"].map(normalize_text)

    val_df  = splits.val.copy()
    test_df = splits.test.copy()

    # ---- rótulos binários consistentes ----
    labels_sorted = sorted(df_pt["label"].unique().tolist())
    pos_label = [l for l in labels_sorted if "bully" in l][0]
    neg_label = [l for l in labels_sorted if l != pos_label][0]
    label_list = [neg_label, pos_label]  # classe 0=neg, 1=pos
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    # ---- tokenizer/model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=id2label, label2id=label2id
    )

    # classe minoritária mais pesada
    tr_counts = train_df["label"].value_counts()
    w_neg = float(tr_counts.get(neg_label, 1))
    w_pos = float(tr_counts.get(pos_label, 1))
    class_weight = torch.tensor([1.0, w_neg / max(w_pos, 1.0)], dtype=torch.float)

    model = WeightedWrapper(base_model, class_weight)

    # ---- datasets e collator ----
    train_ds = TxtDS(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len, label2id)
    val_ds   = TxtDS(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer, args.max_len, label2id)
    test_ds  = TxtDS(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, args.max_len, label2id)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # ---- treino (Trainer mínimo) ----
    args_tr = TrainingArguments(
        output_dir=str(out / "hf_ckpt"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        dataloader_num_workers=0,  # estável em CPU
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=None,                # sem avaliação automática
        data_collator=data_collator,
        tokenizer=tokenizer               # ok mesmo com FutureWarning
    )

    # DICA anti-OOM: descomente se precisar
    # if hasattr(base_model, "gradient_checkpointing_enable"):
    #     base_model.gradient_checkpointing_enable()

    trainer.train()

    # ---- seleção de limiar no VAL ----
    def get_probas(ds):
        raw = trainer.predict(ds).predictions  # logits
        probs = torch.softmax(torch.tensor(raw), dim=1).numpy()
        return probs[:, 1]  # prob da classe positiva

    probs_val = get_probas(val_ds)
    y_val_bin = (val_df["label"] == pos_label).astype(int).values

    best_t, best_score = 0.5, -1.0
    from sklearn.metrics import precision_recall_fscore_support
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (probs_val >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
        f_macro = (f[0] + f[1]) / 2.0
        if r[1] >= args.recall_floor and f_macro > best_score:
            best_t, best_score = t, f_macro
    if best_score < 0:  # fallback: melhor Macro-F1 puro
        for t in np.linspace(0.1, 0.9, 81):
            y_hat = (probs_val >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
            f_macro = (f[0] + f[1]) / 2.0
            if f_macro > best_score:
                best_t, best_score = t, f_macro

    # ---- avaliação no TEST ----
    probs_test   = get_probas(test_ds)
    y_test_bin   = (test_df["label"] == pos_label).astype(int).values
    y_hat_test_b = (probs_test >= best_t).astype(int)
    y_hat_test_l = np.where(y_hat_test_b == 1, pos_label, neg_label)

    mets_test = compute_metrics(test_df["label"], y_hat_test_l)
    save_json(mets_test, out / f"{args.lang}_test_metrics.json")

    # também salvamos val com o mesmo thr
    y_hat_val_b = (probs_val >= best_t).astype(int)
    y_hat_val_l = np.where(y_hat_val_b == 1, pos_label, neg_label)
    mets_val = compute_metrics(val_df["label"], y_hat_val_l)
    save_json(mets_val, out / f"{args.lang}_val_metrics.json")

    # confusões
    labels_bin_order = [neg_label, pos_label]
    plot_confusion(val_df["label"],  y_hat_val_l,  labels_bin_order, out / f"{args.lang}_val_cm.png",
                   title=f"{args.lang} val — {args.model} (thr={best_t:.2f})")
    plot_confusion(test_df["label"], y_hat_test_l, labels_bin_order, out / f"{args.lang}_test_cm.png",
                   title=f"{args.lang} test — {args.model} (thr={best_t:.2f})")

    # infos + predições test para análise de erro
    with open(out / "selection_info.json", "w") as f:
        json.dump({
            "best_threshold": float(best_t),
            "val_macro_f1_at_best_t": float(best_score),
            "params": {
                "model": args.model,
                "epochs": args.epochs, "lr": args.lr,
                "batch": args.batch, "max_len": args.max_len,
                "recall_floor": args.recall_floor,
                "use_multilingual": args.use_multilingual
            }
        }, f, indent=2)

    pd.DataFrame({
        "text": test_df["text"], "y_true": test_df["label"],
        "y_pred": y_hat_test_l, "proba_pos": probs_test
    }).to_csv(out / f"{args.lang}_test_predictions.csv", index=False)

    print(f"[OK] {args.model} + thr {best_t:.2f} → {out}")

if __name__ == "__main__":
    main()
