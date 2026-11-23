"""Fase 4: busca de hiperparâmetros e avaliação em repetições.

1. GridSearchCV no treino da repetição 0 (validação cruzada temporal).
   - Alibaba e Google: uma busca por modelo.
   - NAB: uma busca por modelo e série; a combinação vencedora é escolhida pela
     média das métricas de validação entre as 19 séries (um único conjunto de
     hiperparâmetros por modelo, mais estável que 19 escolhas com ~600 pontos).
2. Com os hiperparâmetros fixos, cada modelo é treinado e testado em cada
   repetição da injeção. Duas variantes de features: "completo" e
   "sem_contexto" (sem seno/cosseno da hora), para medir o efeito do contexto.

Janelas sinalizadas (cobertura baixa no Alibaba) ficam fora do treino e das
métricas; depois do treino, registra-se quantas delas cada modelo marcou.

Uso:  .venv/bin/python -m src.experiments --repeticoes 30
Saídas em results/experimentos/.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src import preprocess as pp
from src.evaluate import contagens_por_evento, metricas_pontuais, metricas_por_evento, recall_por_tipo
from src.models import GRADES, MODELOS, N_DOBRAS, SEED, TETO_FPR, busca, pipeline, regra_f1_com_teto_fpr, tabela_cv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experimentos"
META = {"treino", "rotulo", "tipo_anomalia", "evento_id", "excluir", "n_maquinas", "serie"}
CONTEXTO = ("hora_sin", "hora_cos")
VARIANTES = ("completo", "sem_contexto")
PREPARADORES = {"alibaba": pp.preparar_alibaba, "google": pp.preparar_google}

# Em dobras pequenas do NAB, n_neighbors pode passar do nº de amostras; o sklearn ajusta e avisa.
warnings.filterwarnings("ignore", message=r"n_neighbors \(\d+\) is greater than")


def features(tab: pd.DataFrame, variante: str) -> list[str]:
    cols = [c for c in tab.columns if c not in META]
    return [c for c in cols if c not in CONTEXTO] if variante == "sem_contexto" else cols


def _cronometrar(func, *args):
    t0 = time.perf_counter()
    out = func(*args)
    return out, time.perf_counter() - t0


# --- Alibaba e Google ----------------------------------------------------------

def ajustar(dataset: str) -> tuple[dict, pd.DataFrame]:
    tab, _ = PREPARADORES[dataset](0)
    tr = tab[tab["treino"] & ~tab["excluir"]]
    X, y = tr[features(tab, "completo")], tr["rotulo"]
    melhores, cvs = {}, []
    for m in MODELOS:
        gs = busca(m, X, y)
        melhores[m] = gs.best_params_
        cv = tabela_cv(gs)
        cv["escolhida"] = cv.index == gs.best_index_
        cvs.append(cv.assign(dataset=dataset, modelo=m))
        print(f"  {dataset}/{m}: {gs.best_params_}", flush=True)
    return melhores, pd.concat(cvs, ignore_index=True)


def avaliar(dataset: str, melhores: dict, repeticoes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas, tipos = [], []
    for r in range(repeticoes):
        tab, _ = PREPARADORES[dataset](r)
        tr = tab[tab["treino"] & ~tab["excluir"]]
        te = tab[~tab["treino"] & ~tab["excluir"]]
        sinalizadas = tab[tab["excluir"]]
        for v in VARIANTES:
            cols = features(tab, v)
            for m in MODELOS:
                pipe = pipeline(m, melhores[m])
                _, t_fit = _cronometrar(pipe.fit, tr[cols], tr["rotulo"])
                p, t_pred = _cronometrar(pipe.predict, te[cols])
                linha = {"dataset": dataset, "variante": v, "modelo": m, "repeticao": r,
                         **metricas_pontuais(te["rotulo"], p),
                         "tempo_treino_s": t_fit, "tempo_inferencia_ms": 1000 * t_pred / len(te),
                         "n_treino": len(tr), "n_teste": len(te)}
                if len(sinalizadas):
                    linha["sinalizadas"] = len(sinalizadas)
                    linha["sinalizadas_marcadas"] = int(pipe.predict(sinalizadas[cols]).sum())
                linhas.append(linha)
                rt = recall_por_tipo(te["tipo_anomalia"], te["evento_id"], p)
                tipos.append(rt.assign(dataset=dataset, variante=v, modelo=m, repeticao=r))
        print(f"  {dataset}: repetição {r} ok", flush=True)
    return pd.DataFrame(linhas), pd.concat(tipos, ignore_index=True)


# --- NAB -----------------------------------------------------------------------

def ajustar_nab() -> tuple[dict, pd.DataFrame]:
    tab, _ = pp.preparar_nab(0)
    cols = features(tab, "completo")
    melhores, cvs = {}, []
    for m in MODELOS:
        por_serie = []
        for serie, g in tab.groupby("serie"):
            tr = g[g["treino"]]
            gs = busca(m, tr[cols], tr["rotulo"], refit=False)
            cv = tabela_cv(gs)
            por_serie.append(cv.assign(serie=serie, combinacao=np.arange(len(cv))))
        df = pd.concat(por_serie, ignore_index=True)
        media = df.groupby("combinacao")[["mean_test_f1", "mean_test_fpr", "mean_test_precision", "mean_test_recall"]].mean()
        melhor = regra_f1_com_teto_fpr(media)
        melhores[m] = df.loc[df["combinacao"] == melhor, "params"].iloc[0]
        media = media.assign(params=df.groupby("combinacao")["params"].first(), escolhida=media.index == melhor)
        cvs.append(media.reset_index().assign(dataset="nab", modelo=m))
        print(f"  nab/{m}: {melhores[m]}", flush=True)
    return melhores, pd.concat(cvs, ignore_index=True)


def avaliar_nab(melhores: dict, repeticoes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas, series = [], []
    for r in range(repeticoes):
        tab, _ = pp.preparar_nab(r)
        for v in VARIANTES:
            cols = features(tab, v)
            for m in MODELOS:
                total = dict.fromkeys(["janelas", "janelas_detectadas", "alarmes", "alarmes_em_janela",
                                       "pontos_fora", "alarmes_fora"], 0)
                t_fit_total = t_pred_total = 0.0
                n_teste = 0
                for serie, g in tab.groupby("serie"):
                    tr, te = g[g["treino"]], g[~g["treino"]]
                    pipe = pipeline(m, melhores[m])
                    _, t_fit = _cronometrar(pipe.fit, tr[cols], tr["rotulo"])
                    p, t_pred = _cronometrar(pipe.predict, te[cols])
                    c = contagens_por_evento(te["evento_id"], p)
                    for k in total:
                        total[k] += c[k]
                    t_fit_total += t_fit
                    t_pred_total += t_pred
                    n_teste += len(te)
                    series.append({"variante": v, "modelo": m, "repeticao": r, "serie": serie, **c,
                                   **metricas_por_evento(c)})
                n_series = tab["serie"].nunique()
                linhas.append({"dataset": "nab", "variante": v, "modelo": m, "repeticao": r,
                               **metricas_por_evento(total), **total,
                               "tempo_treino_s": t_fit_total / n_series,
                               "tempo_inferencia_ms": 1000 * t_pred_total / n_teste,
                               "n_teste": n_teste})
        print(f"  nab: repetição {r} ok", flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(series)


# --- Execução ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=pp.REPETICOES)
    ap.add_argument("--datasets", nargs="+", default=["alibaba", "google", "nab"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    inicio = datetime.now(timezone.utc)
    melhores_todos = {}
    for ds in args.datasets:
        print(f"[{ds}] GridSearchCV", flush=True)
        if ds == "nab":
            melhores, cv = ajustar_nab()
        else:
            melhores, cv = ajustar(ds)
        cv["params"] = cv["params"].map(lambda d: json.dumps(d, default=str))
        cv.to_csv(OUT / f"gridsearch_{ds}.csv", index=False)
        melhores_todos[ds] = melhores
        (OUT / f"melhores_parametros_{ds}.json").write_text(json.dumps(melhores, indent=2, default=str))

        print(f"[{ds}] avaliação em {args.repeticoes} repetições", flush=True)
        if ds == "nab":
            res, por_serie = avaliar_nab(melhores, args.repeticoes)
            por_serie.to_csv(OUT / "nab_por_serie.csv", index=False)
        else:
            res, tipos = avaliar(ds, melhores, args.repeticoes)
            tipos.to_csv(OUT / f"recall_por_tipo_{ds}.csv", index=False)
        res.to_csv(OUT / f"resultados_{ds}.csv", index=False)

    execucao = {
        "inicio_utc": inicio.isoformat(), "fim_utc": datetime.now(timezone.utc).isoformat(),
        "repeticoes": args.repeticoes, "datasets": args.datasets, "seed": SEED, "teto_fpr": TETO_FPR,
        "n_dobras_cv": N_DOBRAS, "grades": GRADES, "melhores_parametros": melhores_todos,
        "python": platform.python_version(),
        "bibliotecas": {p: version(p) for p in ("numpy", "pandas", "scikit-learn")},
        "maquina": {"processador": platform.processor() or platform.machine(), "sistema": platform.platform()},
    }
    (OUT / "execucao.json").write_text(json.dumps(execucao, indent=2, default=str))
    print("fim", flush=True)


if __name__ == "__main__":
    main()
