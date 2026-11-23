"""Carga dos datasets: Zenodo (registro 14564935) e NAB (Numenta Anomaly Benchmark).

Os arquivos não têm coluna de tempo: cada linha é uma janela de 300 s em
sequência, e janelas sem dados no trace original simplesmente não aparecem.
O índice temporal é reconstruído a partir da posição da linha.

- Alibaba 2018: a comparação com o machine_usage original (notebook 01_eda,
  seção 3) mostrou uma lacuna de 61 janelas (~5 h) após a linha 472; o índice
  abaixo já a considera. Tempo relativo ao início do trace original.
- Google 2019: não há como verificar lacunas sem o BigQuery; assume-se passo
  regular. Tempo relativo ao início do arquivo.
- NAB: séries univariadas com timestamp real e rótulos oficiais (janelas e
  instantes), baixadas do commit registrado em data/raw/nab/COMMIT.txt.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ALIBABA_CSV = RAW / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
GOOGLE_CSV = RAW / "instance_usage_grouped_300_seconds_month.csv"
# Gerado por src/build_alibaba_aggregate.py a partir do trace original.
ALIBABA_ORIGINAL_300S = PROCESSED / "alibaba_machine_usage_300s_com_timestamp.csv"

STEP_SECONDS = 300
ALIBABA_GAP_AFTER_ROW = 472
ALIBABA_GAP_WINDOWS = 61


def _with_time_index(df: pd.DataFrame, windows: np.ndarray) -> pd.DataFrame:
    df.index = pd.to_timedelta(windows * STEP_SECONDS, unit="s")
    df.index.name = "t"
    return df


def load_alibaba() -> pd.DataFrame:
    df = pd.read_csv(ALIBABA_CSV)
    windows = np.arange(len(df))
    windows[ALIBABA_GAP_AFTER_ROW + 1:] += ALIBABA_GAP_WINDOWS
    return _with_time_index(df, windows)


def load_google() -> pd.DataFrame:
    df = pd.read_csv(GOOGLE_CSV)
    return _with_time_index(df, np.arange(len(df)))


def load_alibaba_coverage() -> pd.Series:
    """Nº de máquinas que reportaram em cada janela, alinhado ao índice de load_alibaba()."""
    orig = pd.read_csv(ALIBABA_ORIGINAL_300S, usecols=["t_inicio_s", "n_maquinas"])
    cov = orig.set_index(pd.to_timedelta(orig["t_inicio_s"], unit="s"))["n_maquinas"]
    cov.index.name = "t"
    return cov


NAB_DIR = RAW / "nab"
NAB_GROUPS = ("realAWSCloudwatch", "realKnownCause")
# Fração inicial de cada série sem rótulos, reservada para aprendizado (protocolo do NAB).
NAB_PROBATIONARY = 0.15


def list_nab_series() -> list[str]:
    """Chaves no formato do NAB: '<grupo>/<arquivo>.csv'."""
    return sorted(f"{p.parent.name}/{p.name}" for g in NAB_GROUPS for p in (NAB_DIR / g).glob("*.csv"))


def load_nab(key: str) -> pd.DataFrame:
    """Série do NAB com colunas value, janela_anomalia e rotulo.

    Timestamps duplicados (presentes em 3 séries) são agregados pela média.
    'janela_anomalia' vale 0 fora das janelas oficiais e k dentro da k-ésima;
    'rotulo' (0/1) marca o ponto mais próximo de cada instante rotulado.
    """
    df = pd.read_csv(NAB_DIR / key, parse_dates=["timestamp"])
    df = df.groupby("timestamp", as_index=True)[["value"]].mean()
    windows = json.loads((NAB_DIR / "labels" / "combined_windows.json").read_text())[key]
    labels = json.loads((NAB_DIR / "labels" / "combined_labels.json").read_text())[key]
    df["janela_anomalia"] = 0
    for i, (ini, fim) in enumerate(windows, start=1):
        df.loc[pd.Timestamp(ini):pd.Timestamp(fim), "janela_anomalia"] = i
    df["rotulo"] = 0
    for ts in labels:
        df.iloc[df.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0], df.columns.get_loc("rotulo")] = 1
    return df
