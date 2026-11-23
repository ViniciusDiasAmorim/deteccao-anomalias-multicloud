"""Modelos e busca de hiperparâmetros.

Isolation Forest e LOF (não supervisionados) e Random Forest (supervisionado,
treinado com os rótulos da injeção sintética) são embrulhados em `Detector`,
que padroniza a saída: `predict` devolve 1 = anomalia e 0 = normal (IF e LOF
devolvem -1/1) e `pontuacao` devolve um escore em que maior = mais anômalo.

Cada modelo é um Pipeline StandardScaler → Detector. O scaler fica dentro do
Pipeline para que, na validação cruzada, ele seja ajustado só na parte de
treino de cada dobra.

Regra de escolha do GridSearchCV: entre as combinações
com FPR média ≤ TETO_FPR na validação cruzada, vence a de maior F1 média;
empate → menor FPR. Se nenhuma atingir o teto, vence a de menor FPR.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TETO_FPR = 0.02
N_DOBRAS = 5

MODELOS = ("isolation_forest", "lof", "random_forest")
NOMES = {"isolation_forest": "Isolation Forest", "lof": "LOF", "random_forest": "Random Forest"}


class Detector(BaseEstimator, ClassifierMixin):
    """Interface única para detectores supervisionados e não supervisionados."""

    def __init__(self, estimador=None):
        self.estimador = estimador

    def _supervisionado(self) -> bool:
        return hasattr(self.estimador, "predict_proba")

    def fit(self, X, y=None):
        self.estimador_ = clone(self.estimador)
        if self._supervisionado():
            self.estimador_.fit(X, y)
        else:
            self.estimador_.fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        if self._supervisionado():
            return np.asarray(self.estimador_.predict(X), dtype=int)
        return (self.estimador_.predict(X) == -1).astype(int)

    def pontuacao(self, X):
        if self._supervisionado():
            classes = list(self.estimador_.classes_)
            if 1 not in classes:
                return np.zeros(len(X))
            return self.estimador_.predict_proba(X)[:, classes.index(1)]
        return -self.estimador_.score_samples(X)


def estimador_base(modelo: str):
    if modelo == "isolation_forest":
        return IsolationForest(random_state=SEED, n_jobs=1)
    if modelo == "lof":
        return LocalOutlierFactor(novelty=True, n_jobs=1)
    if modelo == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=1)
    raise ValueError(modelo)


def pipeline(modelo: str, params: dict | None = None) -> Pipeline:
    pipe = Pipeline([("escala", StandardScaler()), ("detector", Detector(estimador_base(modelo)))])
    if params:
        pipe.set_params(**params)
    return pipe


_P = "detector__estimador__"
GRADES = {
    "isolation_forest": {
        _P + "n_estimators": [100, 300],
        _P + "max_samples": ["auto", 0.5],
        _P + "max_features": [0.5, 1.0],
        _P + "contamination": [0.01, 0.02, 0.05, 0.1],
    },
    "lof": {
        _P + "n_neighbors": [5, 10, 20, 50],
        _P + "metric": ["euclidean", "manhattan"],
        _P + "contamination": [0.01, 0.02, 0.05, 0.1],
    },
    "random_forest": {
        _P + "n_estimators": [100, 300],
        _P + "max_depth": [None, 10],
        _P + "min_samples_leaf": [1, 5],
        _P + "class_weight": [None, "balanced"],
    },
}


def fpr(y, p) -> float:
    tn, fp, _, _ = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return fp / (fp + tn) if fp + tn else 0.0


def pontuar(estimador, X, y) -> dict:
    """Scorer múltiplo do GridSearchCV: uma única predição por dobra."""
    p = estimador.predict(X)
    y = np.asarray(y)
    tp = int(((p == 1) & (y == 1)).sum())
    fp_ = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp_) if tp + fp_ else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr(y, p)}


def regra_f1_com_teto_fpr(cv_results, teto: float = TETO_FPR) -> int:
    f1 = np.asarray(cv_results["mean_test_f1"], dtype=float)
    taxa = np.asarray(cv_results["mean_test_fpr"], dtype=float)
    ok = np.flatnonzero(taxa <= teto)
    if len(ok) == 0:
        return int(np.argmin(taxa))
    ordem = np.lexsort((taxa[ok], -f1[ok]))  # maior F1; empate → menor FPR
    return int(ok[ordem[0]])


def busca(modelo: str, X, y, refit: bool = True, n_jobs: int = -1) -> GridSearchCV:
    """GridSearchCV com validação cruzada temporal (TimeSeriesSplit)."""
    gs = GridSearchCV(
        pipeline(modelo), GRADES[modelo], scoring=pontuar,
        refit=regra_f1_com_teto_fpr if refit else False,
        cv=TimeSeriesSplit(n_splits=N_DOBRAS), n_jobs=n_jobs, error_score="raise",
    )
    return gs.fit(X, y)


def tabela_cv(gs: GridSearchCV) -> pd.DataFrame:
    cols = ["params"] + [f"{s}_test_{m}" for m in ("f1", "precision", "recall", "fpr") for s in ("mean", "std")]
    return pd.DataFrame(gs.cv_results_)[cols]
