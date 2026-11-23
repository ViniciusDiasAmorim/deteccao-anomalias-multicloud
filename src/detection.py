"""Detecção de anomalias em uma série univariada qualquer (coletas AWS/OCI, NAB).

Reaproveita o protocolo do NAB, cujas séries são métricas reais do CloudWatch
— o caso mais próximo das coletas das nuvens:
- features do NAB (log1p do valor, variação e hora do dia);
- hiperparâmetros escolhidos no GridSearchCV do NAB (fase 4);
- o modelo aprende com o trecho inicial da série (`fracao_treino`) e marca
  anomalias no restante. O Random Forest, por ser supervisionado, aprende com
  anomalias sintéticas injetadas nesse trecho inicial, como na fase 4.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.inject import injetar
from src.models import MODELOS, pipeline
from src.preprocess import NAB, construir_features

ROOT = Path(__file__).resolve().parents[1]
PARAMETROS = ROOT / "results" / "experimentos" / "melhores_parametros_nab.json"
MIN_TREINO = 50  # janelas de 5 min (~4 h)


def parametros() -> dict:
    return json.loads(PARAMETROS.read_text())


def detectar(serie: pd.Series, modelo: str = "isolation_forest", fracao_treino: float = NAB.fracao_treino,
             params: dict | None = None) -> pd.DataFrame:
    """Recebe valores indexados por instante e devolve instante, valor, escore, anomalia e treino."""
    if modelo not in MODELOS:
        raise ValueError(f"modelo desconhecido: {modelo}")
    serie = serie.sort_index()
    serie = serie[~serie.index.duplicated()].astype(float)
    if (serie < 0).any():
        raise ValueError("a série tem valores negativos; as features usam log1p")
    n_treino = int(fracao_treino * len(serie))
    if n_treino < MIN_TREINO or len(serie) - n_treino < 1:
        raise ValueError(f"série curta: são necessários ao menos {MIN_TREINO} pontos de treino")

    df = serie.to_frame("value")
    treino = np.arange(len(df)) < n_treino
    y = np.zeros(len(df), dtype=int)
    if modelo == "random_forest":
        injetado, marcacao, _ = injetar(df[treino], NAB.injecao_treino, referencia=df[treino])
        df = pd.concat([injetado, df[~treino]])
        y[treino] = marcacao["rotulo"].to_numpy()

    X = construir_features(df, NAB.features)
    pipe = pipeline(modelo, (params or parametros())[modelo])
    pipe.fit(X[treino], y[treino])
    escore = pipe.named_steps["detector"].pontuacao(pipe[:-1].transform(X))
    anomalia = pipe.predict(X)
    anomalia[treino] = 0  # o trecho de treino não é avaliado
    return pd.DataFrame({"instante": serie.index, "valor": serie.to_numpy(), "escore": escore,
                         "anomalia": anomalia.astype(bool), "treino": treino})
