"""Esquema comum das métricas coletadas nas nuvens.

Cada coletor devolve um DataFrame no mesmo formato da tabela `metricas`
(fonte, recurso, metrica, instante, valor, unidade), com nomes canônicos de
métrica, instantes em UTC e granularidade de 5 min — a mesma dos datasets.
"""
from datetime import datetime, timezone

import pandas as pd

PERIODO_S = 300

# Nome canônico → unidade. Rede e disco são o total de bytes no período de 5 min.
CANONICAS = {
    "cpu_util_percent": "%",
    "mem_util_percent": "%",
    "net_in_bytes": "bytes/5min",
    "net_out_bytes": "bytes/5min",
    "disk_read_bytes": "bytes/5min",
    "disk_write_bytes": "bytes/5min",
}

COLUNAS = ["fonte", "recurso", "metrica", "instante", "valor", "unidade"]


def para_pontos(fonte: str, recurso: str, metrica: str, instantes, valores) -> pd.DataFrame:
    if metrica not in CANONICAS:
        raise ValueError(f"métrica não canônica: {metrica}")
    df = pd.DataFrame({"instante": pd.to_datetime(list(instantes), utc=True), "valor": list(valores)})
    df = df.dropna().drop_duplicates("instante").sort_values("instante")
    df["fonte"], df["recurso"], df["metrica"], df["unidade"] = fonte, recurso, metrica, CANONICAS[metrica]
    return df[COLUNAS].reset_index(drop=True)


def vazio() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUNAS)


def utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
