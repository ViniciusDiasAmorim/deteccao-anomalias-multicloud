"""Reconstrói a série agregada do Alibaba 2018 COM timestamp, a partir do trace original.

O CSV do Zenodo não tem coluna de tempo, o que impede cruzá-lo com eventos do
trace original (ex.: tarefas com status Failed). Este script lê o machine_usage
original (tar.gz de ~1,7 GB, lido em streaming, sem descompactar em disco),
calcula a média de todas as máquinas por janela de 300 s e grava o resultado
com o início de cada janela em segundos.

Uso:
    tar -xzOf data/raw/alibaba_v2018/machine_usage.tar.gz \
        | .venv/bin/python -I src/build_alibaba_aggregate.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP_SECONDS = 300

COLS = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
        "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
METRICS = ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]
OUT = ROOT / "data" / "processed" / "alibaba_machine_usage_300s_com_timestamp.csv"


def main() -> None:
    sums, counts, machines = [], [], []
    reader = pd.read_csv(sys.stdin, header=None, names=COLS,
                         usecols=["machine_id", "time_stamp"] + METRICS,
                         chunksize=10_000_000)
    for i, chunk in enumerate(reader):
        chunk["janela"] = (chunk["time_stamp"] // STEP_SECONDS).astype("int64")
        # disk_io_percent usa -1 e 101 como valores inválidos (schema original)
        chunk.loc[~chunk["disk_io_percent"].between(0, 100), "disk_io_percent"] = pd.NA
        g = chunk.groupby("janela")
        sums.append(g[METRICS].sum())
        counts.append(g[METRICS].count())
        machines.append(chunk[["janela", "machine_id"]].drop_duplicates())
        print(f"bloco {i}: {len(chunk):,} linhas", file=sys.stderr)

    total = pd.concat(sums).groupby(level=0).sum()
    n = pd.concat(counts).groupby(level=0).sum()
    agg = total / n
    agg["n_amostras"] = n["cpu_util_percent"]
    agg["n_maquinas"] = (pd.concat(machines).drop_duplicates()
                         .groupby("janela").size())
    agg.insert(0, "t_inicio_s", agg.index * STEP_SECONDS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"gravado {OUT} ({len(agg)} janelas)", file=sys.stderr)


if __name__ == "__main__":
    main()
