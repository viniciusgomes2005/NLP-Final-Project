# build_notebook.py
# Gera um Jupyter Notebook (.ipynb) com todo o pipeline SVM calibrado + análise

from __future__ import annotations
import argparse, json
from pathlib import Path
import nbformat as nbf


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())

def code(s: str):
    return nbf.v4.new_code_cell(s.rstrip() + "\n")

TEMPLATE_MD_TITLE = """
# Detecção de discurso ofensivo (PT) — SVM calibrado com controle de threshold

Notebook gerado automaticamente. Contém:
- Carga/normalização de dados
- *Splits* estratificados (train/val/test)
- Pipeline TF-IDF (word+char) + LinearSVC com calibração sigmóide
- Varredura de *threshold* no **VAL** com piso de *recall* para a minoria
- Avaliação no **TEST** (métricas, *confusion matrix*)
- Histograma de probabilidades e análise de erro (Top-3 FNs/FPs)
"""

TEMPLATE_CODE_SETUP = r"""
# %% [setup]
import os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    confusion_matrix, accuracy_score, f1_score
)

# Config geral de plots (sem seaborn)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (6.5, 4.2)

LANG_COL  = "lang"
TEXT_COL  = "text"
LABEL_COL = "label"
"""

TEMPLATE_CODE_IO_AND_UTILS = r"""
# %% [utils: dados/normalização/metrics]
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USR_RE = re.compile(r"@\w+")
def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = URL_RE.sub("__url__", s)
    s = USR_RE.sub("__user__", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {TEXT_COL, LABEL_COL}.issubset(df.columns), "CSV must have text,label"
    df[LANG_COL] = df[LABEL_COL].astype(str).str.split("_").str[0]
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    return df

def filter_lang(df: pd.DataFrame, lang3: str) -> pd.DataFrame:
    return df[df[LANG_COL] == lang3].reset_index(drop=True)

@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def stratified_splits_lang(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42) -> Splits:
    X = df[TEXT_COL]; y = df[LABEL_COL]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    rel_val = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-rel_val, random_state=seed, stratify=y_temp)
    train = pd.DataFrame({TEXT_COL: X_train, LABEL_COL: y_train})
    val   = pd.DataFrame({TEXT_COL: X_val,   LABEL_COL: y_val})
    test  = pd.DataFrame({TEXT_COL: X_test,  LABEL_COL: y_test})
    return Splits(train, val, test)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

def plot_confusion(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.show()
"""

TEMPLATE_CODE_PARAMS = r"""
# %% [params]
CSV_PATH      = "{csv}"
OUT_DIR       = Path("{out_dir}")
LANG          = "{lang}"
C             = {C}
RECALL_FLOOR  = {recall_floor}
WORD_MIN_DF   = {word_min_df}
WORD_MAX_DF   = {word_max_df}
CHAR_MIN_DF   = {char_min_df}
CHAR_NG_MIN   = {char_ng_min}
CHAR_NG_MAX   = {char_ng_max}

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Params:", dict(
    CSV_PATH=CSV_PATH, LANG=LANG, C=C, RECALL_FLOOR=RECALL_FLOOR,
    WORD_MIN_DF=WORD_MIN_DF, WORD_MAX_DF=WORD_MAX_DF,
    CHAR_MIN_DF=CHAR_MIN_DF, CHAR_NG=(CHAR_NG_MIN, CHAR_NG_MAX),
))
"""

TEMPLATE_CODE_LOAD_SPLITS = r"""
# %% [load & splits]
df = load_csv(CSV_PATH)
df = df[df[TEXT_COL].astype(str).str.strip()!=""]
df[TEXT_COL] = df[TEXT_COL].map(normalize_text)

df_lang = filter_lang(df, LANG)
print("shape total/lang:", df.shape, df_lang.shape)
df_lang.head(3)

splits = stratified_splits_lang(df_lang, test_size=0.15, val_size=0.15, seed=42)
for name, part in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
    print(name, part.shape, "| class dist:")
    print(part[LABEL_COL].value_counts(normalize=True).round(3), "\n")

labels_sorted = sorted(df_lang[LABEL_COL].unique().tolist())
pos_label = [l for l in labels_sorted if "bully" in l][0]
neg_label = [l for l in labels_sorted if l != pos_label][0]
labels_sorted_bin = [neg_label, pos_label]
print("pos_label:", pos_label, "| neg_label:", neg_label)
"""

TEMPLATE_CODE_PIPE_TRAIN = r"""
# %% [pipeline: TF-IDF word+char + LinearSVC calibrado]
feats = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1,2),
        min_df=WORD_MIN_DF, max_df=WORD_MAX_DF,
        sublinear_tf=True, strip_accents="unicode", lowercase=True)),
    ("char", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(CHAR_NG_MIN, CHAR_NG_MAX),
        min_df=CHAR_MIN_DF, lowercase=True))
])

base = LinearSVC(C=C, class_weight="balanced")
clf  = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

pipe = Pipeline([("feats", feats), ("clf", clf)])
pipe.fit(splits.train[TEXT_COL], splits.train[LABEL_COL])

idx_pos = list(pipe.named_steps["clf"].classes_).index(pos_label)
print("classes_:", list(pipe.named_steps["clf"].classes_), "| idx_pos:", idx_pos)
"""

TEMPLATE_CODE_SCAN_THRESHOLD = r"""
# %% [scan threshold no VAL com piso de recall]
y_val_bin  = (splits.val[LABEL_COL]==pos_label).astype(int).values
probs_val  = pipe.predict_proba(splits.val[TEXT_COL])[:, idx_pos]

best_t, best_score, best_stats = 0.5, -1.0, None
grid = np.linspace(0.1, 0.9, 81)

macro_hist = []
for t in grid:
    y_hat = (probs_val>=t).astype(int)
    p,r,f,s = precision_recall_fscore_support(
        y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
    f_macro = (f[0]+f[1])/2.0
    macro_hist.append((t, f_macro, r[1]))
    if r[1] >= RECALL_FLOOR and f_macro > best_score:
        best_t, best_score = t, f_macro
        best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}

if best_score < 0:  # fallback: melhor Macro-F1
    for t in grid:
        y_hat = (probs_val>=t).astype(int)
        p,r,f,s = precision_recall_fscore_support(
            y_val_bin, y_hat, average=None, labels=[0,1], zero_division=0)
        f_macro = (f[0]+f[1])/2.0
        if f_macro > best_score:
            best_t, best_score = t, f_macro
            best_stats = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}

print(f"Threshold escolhido: {best_t:.2f} | Val Macro-F1: {best_score:.4f} | Recall(minoria)≥{RECALL_FLOOR}")
# Curvas Macro-F1 e recall da minoria
mts = np.array(macro_hist)
plt.figure()
plt.plot(mts[:,0], mts[:,1], label="Macro-F1")
plt.plot(mts[:,0], mts[:,2], label="Recall (classe positiva)")
plt.axvline(best_t, linestyle="--")
plt.title("Val: Macro-F1 e Recall(+) vs Threshold")
plt.xlabel("threshold"); plt.ylabel("score"); plt.legend(); plt.tight_layout()
plt.show()
"""

TEMPLATE_CODE_EVAL_TEST = r"""
# %% [avaliação em TEST]
probs_test      = pipe.predict_proba(splits.test[TEXT_COL])[:, idx_pos]
y_test_bin      = (splits.test[LABEL_COL]==pos_label).astype(int).values
y_hat_test_bin  = (probs_test>=best_t).astype(int)
y_hat_test_lbl  = np.where(y_hat_test_bin==1, pos_label, neg_label)

mets_test = compute_metrics(splits.test[LABEL_COL], y_hat_test_lbl)
print(json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in mets_test.items() if k!='report'}, indent=2))

# também avaliar no VAL usando o threshold final
y_hat_val_bin = (probs_val>=best_t).astype(int)
y_hat_val_lbl = np.where(y_hat_val_bin==1, pos_label, neg_label)
mets_val = compute_metrics(splits.val[LABEL_COL], y_hat_val_lbl)
print("\\n[VAL] accuracy:", round(mets_val["accuracy"],4), "| macro_f1:", round(mets_val["macro_f1"],4))

# Confusion matrices
plot_confusion(splits.val[LABEL_COL],  y_hat_val_lbl,  labels_sorted_bin,
               title=f"{LANG} val — SVM Calibrado (thr={best_t:.2f})")
plot_confusion(splits.test[LABEL_COL], y_hat_test_lbl, labels_sorted_bin,
               title=f"{LANG} test — SVM Calibrado (thr={best_t:.2f})")

# Salvar métricas/artefatos
with open(OUT_DIR / f"{LANG}_test_metrics.json","w",encoding="utf-8") as f:
    json.dump(mets_test, f, indent=2, ensure_ascii=False)
with open(OUT_DIR / f"{LANG}_val_metrics.json","w",encoding="utf-8") as f:
    json.dump(mets_val, f, indent=2, ensure_ascii=False)

pd.DataFrame({
    "text": splits.test[TEXT_COL],
    "y_true": splits.test[LABEL_COL],
    "y_pred": y_hat_test_lbl,
    "proba_pos": probs_test
}).to_csv(OUT_DIR / f"{LANG}_test_predictions.csv", index=False)
print("Artefatos salvos em:", OUT_DIR)
"""

TEMPLATE_CODE_HISTO_FN_FP = r"""
# %% [histograma e top FNs/FPs]
pred_path = OUT_DIR / f"{LANG}_test_predictions.csv"
dfp = pd.read_csv(pred_path)

pos = pos_label
dfp["y_true_bin"] = (dfp["y_true"]==pos).astype(int)

plt.figure()
dfp[dfp.y_true_bin==1]["proba_pos"].hist(alpha=0.6, bins=20, label="bully (verdadeiro)")
dfp[dfp.y_true_bin==0]["proba_pos"].hist(alpha=0.6, bins=20, label="nonbully (verdadeiro)")
plt.axvline(best_t, linestyle="--")
plt.title(f"Distribuição de probabilidades (TEST) — linha = threshold {best_t:.2f}")
plt.xlabel("proba_pos"); plt.ylabel("contagem"); plt.legend(); plt.tight_layout()
plt.show()

# Top-3 FNs (era bully, previu nonbully): menores probabilidades
fns = dfp[(dfp.y_true==pos)&(dfp.y_pred!=pos)].sort_values("proba_pos").head(3)[["text","proba_pos"]]
# Top-3 FPs (era nonbully, previu bully): maiores probabilidades
fps = dfp[(dfp.y_true!=pos)&(dfp.y_pred==pos)].sort_values("proba_pos", ascending=False).head(3)[["text","proba_pos"]]

print("\\nTop-3 FNs:\n", fns.to_string(index=False))
print("\\nTop-3 FPs:\n", fps.to_string(index=False))
"""

TEMPLATE_MD_WRAP = """
## Conclusão (auto)

- Threshold escolhido pelo **VAL**: otimiza Macro-F1 com piso de *recall* para a minoria.
- O pipeline **word+char TF-IDF + LinearSVC calibrado** é leve e competitivo em PT ruidoso.
- Próximos passos sugeridos:
  1. *Ablation* (word-only vs char-only vs ambos)
  2. Regras leves para filtrar FPs “neutros”
  3. Comparativo rápido com um BERT PT (ex.: BERTimbau) e análise de custo/latência
"""

def build(args):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(TEMPLATE_MD_TITLE),
        code(TEMPLATE_CODE_SETUP),
        code(TEMPLATE_CODE_IO_AND_UTILS),
        code(TEMPLATE_CODE_PARAMS.format(
            csv=str(Path(args.csv).as_posix()),
            out_dir=str(Path(args.out).parent.as_posix()) if args.out.endswith(".ipynb") else str(Path(args.out).as_posix()),
            lang=args.lang,
            C=args.C,
            recall_floor=args.recall_floor,
            word_min_df=args.word_min_df,
            word_max_df=args.word_max_df,
            char_min_df=args.char_min_df,
            char_ng_min=args.char_ng_min,
            char_ng_max=args.char_ng_max,
        )),
        code(TEMPLATE_CODE_LOAD_SPLITS),
        code(TEMPLATE_CODE_PIPE_TRAIN),
        code(TEMPLATE_CODE_SCAN_THRESHOLD),
        code(TEMPLATE_CODE_EVAL_TEST),
        code(TEMPLATE_CODE_HISTO_FN_FP),
        md(TEMPLATE_MD_WRAP),
    ]

    # kernelspec básico
    nb["metadata"].update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    })

    out_ipynb = Path(args.out)
    if out_ipynb.suffix.lower() != ".ipynb":
        out_ipynb = out_ipynb.with_suffix(".ipynb")
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ipynb, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[OK] Notebook gerado em: {out_ipynb}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Caminho para dataset.csv (colunas text,label)")
    p.add_argument("--out", required=True, help="Saída do notebook, ex.: report/report.ipynb")
    p.add_argument("--lang", default="por")
    p.add_argument("--C", type=float, default=3.0)
    p.add_argument("--recall_floor", type=float, default=0.60)
    p.add_argument("--word_min_df", type=int, default=1)
    p.add_argument("--word_max_df", type=float, default=0.90)
    p.add_argument("--char_min_df", type=int, default=1)
    p.add_argument("--char_ng_min", type=int, default=3)
    p.add_argument("--char_ng_max", type=int, default=7)
    args = p.parse_args()
    build(args)
