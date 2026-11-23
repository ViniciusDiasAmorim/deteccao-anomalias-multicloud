"""Estudo de caso: as rajadas de tarefas com falha do Alibaba aparecem nos escores?

As rajadas (EDA, seção 6.3) são um proxy definido pelo autor, não falhas
documentadas. Pergunta: os detectores não supervisionados, com os
hiperparâmetros da fase 4, dão escores mais altos às janelas de rajada do que
às demais?

Para que todo escore seja fora da amostra, usa-se validação cruzada por blocos
de 1 dia: cada dia é pontuado por um modelo treinado nos outros dias, sem as
janelas de rajada e sem as janelas de cobertura baixa. Não há injeção.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import data
from src.models import pipeline
from src.preprocess import ALIBABA, alibaba_cobertura_baixa, construir_features

ROOT = Path(__file__).resolve().parents[1]
CACHE_FALHAS = ROOT / "data" / "processed" / "alibaba_falhas_por_janela.csv"
JANELAS_POR_DIA = 288
N_JANELAS = 8 * JANELAS_POR_DIA


def falhas_por_janela() -> pd.DataFrame:
    """Tarefas em lote encerradas e com status Failed por janela de 5 min (trace original)."""
    if CACHE_FALHAS.exists():
        return pd.read_csv(CACHE_FALHAS, index_col="janela")
    bt = pd.read_csv(data.RAW / "alibaba_v2018" / "batch_task.csv", header=None,
                     names=["task_name", "instance_num", "job_name", "task_type", "status",
                            "start_time", "end_time", "plan_cpu", "plan_mem"],
                     usecols=["status", "end_time"])
    grade = pd.RangeIndex(N_JANELAS, name="janela")
    janela = bt["end_time"] // data.STEP_SECONDS
    out = pd.DataFrame({
        "falhas": janela[bt["status"] == "Failed"].value_counts().reindex(grade, fill_value=0),
        "encerradas": janela.value_counts().reindex(grade, fill_value=0),
    })
    out.to_csv(CACHE_FALHAS)
    return out


def rajadas(falhas: pd.DataFrame, fator: float = 20) -> list[tuple[int, int]]:
    """Trechos em que a média móvel de 1 h de falhas passa de `fator` × a mediana (mesmo critério da EDA)."""
    media_1h = falhas["falhas"].rolling(12, center=True, min_periods=1).mean()
    acima = media_1h > fator * falhas["falhas"].median()
    trechos = (acima != acima.shift()).cumsum()[acima]
    return [(int(g.index.min()), int(g.index.max())) for _, g in trechos.groupby(trechos)]


def escores_fora_da_amostra(params: dict, modelos=("isolation_forest", "lof")) -> pd.DataFrame:
    ali = data.load_alibaba()
    X = construir_features(ali, ALIBABA.features)
    janela = (ali.index / pd.Timedelta(seconds=data.STEP_SECONDS)).astype(int).to_numpy()
    dia = janela // JANELAS_POR_DIA
    em_rajada = np.zeros(len(ali), dtype=bool)
    for ini, fim in rajadas(falhas_por_janela()):
        em_rajada |= (janela >= ini) & (janela <= fim)
    cobertura_baixa = alibaba_cobertura_baixa().to_numpy()

    linhas = []
    for m in modelos:
        escore = np.full(len(ali), np.nan)
        anomalia = np.zeros(len(ali), dtype=int)
        for d in np.unique(dia):
            alvo = dia == d
            treino = ~alvo & ~em_rajada & ~cobertura_baixa
            pipe = pipeline(m, params[m]).fit(X[treino])
            escore[alvo] = pipe.named_steps["detector"].pontuacao(pipe[:-1].transform(X[alvo]))
            anomalia[alvo] = pipe.predict(X[alvo])
        linhas.append(pd.DataFrame({"modelo": m, "janela": janela, "dia": dia, "em_rajada": em_rajada,
                                    "cobertura_baixa": cobertura_baixa, "escore": escore, "anomalia": anomalia}))
    return pd.concat(linhas, ignore_index=True)


def resumo(escores: pd.DataFrame) -> pd.DataFrame:
    """Por modelo: fração marcada dentro e fora das rajadas e AUC do escore (rajada × resto)."""
    d = escores[~escores["cobertura_baixa"]]
    linhas = []
    for m, g in d.groupby("modelo"):
        linhas.append({"modelo": m,
                       "janelas_em_rajada": int(g["em_rajada"].sum()),
                       "marcadas_em_rajada": g.loc[g["em_rajada"], "anomalia"].mean(),
                       "marcadas_fora": g.loc[~g["em_rajada"], "anomalia"].mean(),
                       "auc_escore": roc_auc_score(g["em_rajada"], g["escore"])})
    return pd.DataFrame(linhas)


def parametros_alibaba() -> dict:
    return json.loads((ROOT / "results" / "experimentos" / "melhores_parametros_alibaba.json").read_text())
