"""Carga do banco: séries dos datasets e resultados da fase 4.

Uso:
    .venv/bin/python -m src.load_db metricas       # Alibaba, Google e NAB → tabela metricas
    .venv/bin/python -m src.load_db experimentos   # results/experimentos → experimentos/resultados
"""
import argparse

import pandas as pd

from src import data, db
from src.experiments import OUT

UNIDADES_ALIBABA = {"cpu_util_percent": "%", "mem_util_percent": "%", "net_in": "normalizado 0-100",
                    "net_out": "normalizado 0-100", "disk_io_percent": "%"}
UNIDADES_GOOGLE = {"avg_cpu": "normalizado 0-1", "avg_mem": "normalizado 0-1", "avg_assigned_mem": "normalizado 0-1",
                   "avg_cycles_per_instruction": "ciclos/instrução"}


def carregar_metricas(engine) -> dict:
    total = {}
    ali = db.series_para_pontos(data.load_alibaba(), "alibaba", "cluster", UNIDADES_ALIBABA)
    total["alibaba"] = db.gravar_metricas(engine, ali, substituir_fonte=True)
    goo = db.series_para_pontos(data.load_google(), "google", "celula", UNIDADES_GOOGLE)
    total["google"] = db.gravar_metricas(engine, goo, substituir_fonte=True)
    partes = []
    for key in data.list_nab_series():
        serie = data.load_nab(key)
        nome = key.split("/")[1].removesuffix(".csv")
        partes.append(db.series_para_pontos(serie[["value"]], "nab", nome, {"value": None}))
    total["nab"] = db.gravar_metricas(engine, pd.concat(partes, ignore_index=True), substituir_fonte=True)
    return total


def carregar_experimentos(engine) -> int:
    execucao = db.ler_execucao(OUT / "execucao.json")
    resultados = pd.concat([pd.read_csv(OUT / f"resultados_{ds}.csv") for ds in execucao["datasets"]],
                           ignore_index=True)
    return db.gravar_experimentos(engine, execucao, resultados)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("alvo", choices=["metricas", "experimentos"])
    args = ap.parse_args()
    engine = db.conectar()
    db.criar_tabelas(engine)
    if args.alvo == "metricas":
        print(carregar_metricas(engine))
    else:
        print(f"{carregar_experimentos(engine)} resultados gravados")


if __name__ == "__main__":
    main()
