"""Métricas de avaliação.

- Alibaba e Google (anomalias sintéticas): métricas ponto a ponto — cada janela
  de 5 min é um exemplo — e recall por evento para cada tipo de anomalia.
- NAB (anomalias reais): as janelas oficiais são largas (~10 % da série), então a
  avaliação é por evento:
    recall    = janelas com pelo menos um alarme / total de janelas;
    precision = alarmes dentro de alguma janela / total de alarmes;
    FPR       = alarmes fora das janelas / pontos fora das janelas;
    F1        = média harmônica de precision e recall.
"""
import numpy as np
import pandas as pd

from src.models import fpr


def metricas_pontuais(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr(y, p),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def recall_por_tipo(tipo, evento_id, p) -> pd.DataFrame:
    """Fração dos eventos de cada tipo com pelo menos uma janela marcada."""
    d = pd.DataFrame({"tipo": np.asarray(tipo), "evento": np.asarray(evento_id), "p": np.asarray(p)})
    d = d[d["evento"] > 0]
    por_evento = d.groupby(["tipo", "evento"])["p"].max()
    return por_evento.groupby("tipo").agg(eventos="size", detectados="sum").reset_index()


def contagens_por_evento(janela, p) -> dict:
    """Contagens para as métricas por evento (somáveis entre séries)."""
    janela, p = np.asarray(janela), np.asarray(p)
    ids = np.unique(janela[janela > 0])
    detectadas = sum(int(p[janela == k].any()) for k in ids)
    return {"janelas": len(ids), "janelas_detectadas": detectadas,
            "alarmes": int(p.sum()), "alarmes_em_janela": int(p[janela > 0].sum()),
            "pontos_fora": int((janela == 0).sum()), "alarmes_fora": int(p[janela == 0].sum())}


def metricas_por_evento(c: dict) -> dict:
    recall = c["janelas_detectadas"] / c["janelas"] if c["janelas"] else 0.0
    precision = c["alarmes_em_janela"] / c["alarmes"] if c["alarmes"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    taxa = c["alarmes_fora"] / c["pontos_fora"] if c["pontos_fora"] else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": taxa}


def resumo(df: pd.DataFrame, por: list[str], metricas: list[str]) -> pd.DataFrame:
    """Média e desvio padrão entre repetições."""
    g = df.groupby(por)[metricas].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()
