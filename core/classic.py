from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
from sklearn.linear_model import LogisticRegression


@dataclass
class ClassicConfig:
    ngram_range: tuple[int,int]=(1,2)
    min_df: int=2
    max_df: float=0.95
    C: float=1.0
    calibrated: bool=False
    analyzer: str="word"
    sublinear_tf: bool=True

def build_pipeline(cfg: ClassicConfig) -> Pipeline:
    base_clf = LinearSVC(C=cfg.C, class_weight="balanced")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1,2),
            min_df=1, max_df=0.95, sublinear_tf=True,
            lowercase=True, strip_accents="unicode"
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    return pipe

def save_model(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str | Path):
    return joblib.load(path)
