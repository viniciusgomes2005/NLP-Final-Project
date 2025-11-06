from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LANG_COL = "lang"
TEXT_COL = "text"
LABEL_COL = "label"

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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    rel_val = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel_val, random_state=seed, stratify=y_temp)
    train = pd.DataFrame({TEXT_COL: X_train, LABEL_COL: y_train})
    val   = pd.DataFrame({TEXT_COL: X_val,   LABEL_COL: y_val})
    test  = pd.DataFrame({TEXT_COL: X_test,  LABEL_COL: y_test})
    return Splits(train, val, test)

def make_k_shot(df: pd.DataFrame, k: int, seed=42) -> pd.DataFrame:
    out = []
    for label, grp in df.groupby(LABEL_COL):
        take = min(k, len(grp))
        out.append(grp.sample(n=take, random_state=seed))
    return pd.concat(out).sample(frac=1.0, random_state=seed).reset_index(drop=True)
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USR_RE = re.compile(r"@\w+")
def normalize_text(s: str) -> str:
    s = s.lower()
    s = URL_RE.sub("__url__", s)
    s = USR_RE.sub("__user__", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
