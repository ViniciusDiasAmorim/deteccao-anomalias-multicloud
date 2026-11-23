"""API de detecção de anomalias e painel de alertas.

Rodar:  .venv/bin/uvicorn src.api.main:app --port 8000
Painel: http://localhost:8000/   Documentação: http://localhost:8000/docs
"""
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import distinct, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src import db
from src.detection import detectar
from src.preprocess import NAB

ESTATICO = Path(__file__).parent / "static"
Modelo = Literal["isolation_forest", "lof", "random_forest"]

app = FastAPI(title="Detecção de anomalias em infraestrutura multi-cloud",
              description="TCC — comparação de Isolation Forest, LOF e Random Forest.", version="1.0")


def obter_engine(request: Request) -> Engine:
    if getattr(request.app.state, "engine", None) is None:
        request.app.state.engine = db.conectar()
        db.criar_tabelas(request.app.state.engine)
    return request.app.state.engine


class Ponto(BaseModel):
    instante: datetime
    valor: float = Field(ge=0)


# Padrão: Isolation Forest, que teve a menor FPR no NAB (métricas reais do CloudWatch). Pela regra da fase 4
# (maior F1 com FPR ≤ 2 %; se nenhum atingir o teto, menor FPR), nenhum modelo ficou abaixo de 2 % no teste do NAB.
MODELO_PADRAO = "isolation_forest"


class EntradaDeteccao(BaseModel):
    pontos: list[Ponto] = Field(min_length=2)
    modelo: Modelo = MODELO_PADRAO
    fracao_treino: float = Field(NAB.fracao_treino, gt=0, lt=1)


class PontoDetectado(Ponto):
    escore: float
    anomalia: bool
    treino: bool


class SaidaDeteccao(BaseModel):
    modelo: Modelo
    n_treino: int
    n_anomalias: int
    pontos: list[PontoDetectado]


class EntradaRecurso(BaseModel):
    fonte: str
    recurso: str
    metrica: str
    modelo: Modelo = MODELO_PADRAO


def _saida(resultado: pd.DataFrame, modelo: str) -> SaidaDeteccao:
    pontos = [PontoDetectado(instante=r.instante, valor=r.valor, escore=r.escore, anomalia=r.anomalia,
                             treino=r.treino) for r in resultado.itertuples()]
    return SaidaDeteccao(modelo=modelo, n_treino=int(resultado["treino"].sum()),
                         n_anomalias=int(resultado["anomalia"].sum()), pontos=pontos)


@app.get("/", include_in_schema=False)
def painel():
    return FileResponse(ESTATICO / "index.html")


@app.get("/saude")
def saude(engine: Engine = Depends(obter_engine)):
    with engine.connect() as c:
        c.execute(text("select 1"))
    return {"status": "ok", "banco": "ok"}


@app.post("/api/detectar", response_model=SaidaDeteccao)
def api_detectar(entrada: EntradaDeteccao):
    """Detecta anomalias numa série enviada no corpo (não grava nada)."""
    # Instantes sem fuso são tratados como UTC.
    serie = pd.Series([p.valor for p in entrada.pontos],
                      index=pd.to_datetime([p.instante for p in entrada.pontos], utc=True))
    try:
        resultado = detectar(serie, entrada.modelo, entrada.fracao_treino)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return _saida(resultado, entrada.modelo)


@app.get("/api/recursos")
def api_recursos(engine: Engine = Depends(obter_engine)):
    """Séries com horário real no banco (NAB e coletas de AWS/OCI)."""
    m = db.Metrica
    consulta = (select(m.fonte, m.recurso, m.metrica, func.count().label("pontos"),
                       func.min(m.instante).label("inicio"), func.max(m.instante).label("fim"))
                .where(m.instante.is_not(None)).group_by(m.fonte, m.recurso, m.metrica)
                .order_by(m.fonte, m.recurso, m.metrica))
    with Session(engine) as s:
        return [dict(r._mapping) for r in s.execute(consulta)]


@app.get("/api/series")
def api_series(fonte: str, recurso: str, metrica: str, engine: Engine = Depends(obter_engine)):
    serie = db.ler_serie(engine, fonte, recurso, metrica)
    if serie.empty:
        raise HTTPException(404, "série não encontrada")
    return [{"instante": i, "valor": v} for i, v in serie.items()]


@app.post("/api/recursos/detectar", response_model=SaidaDeteccao)
def api_detectar_recurso(entrada: EntradaRecurso, engine: Engine = Depends(obter_engine)):
    """Detecta anomalias numa série do banco e grava os alertas."""
    serie = db.ler_serie(engine, entrada.fonte, entrada.recurso, entrada.metrica)
    if serie.empty:
        raise HTTPException(404, "série não encontrada")
    try:
        resultado = detectar(serie, entrada.modelo)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    db.gravar_alertas(engine, resultado, entrada.fonte, entrada.recurso, entrada.metrica, entrada.modelo)
    return _saida(resultado, entrada.modelo)


@app.get("/api/alertas")
def api_alertas(fonte: str | None = None, recurso: str | None = None, limite: int = Query(100, ge=1, le=5000),
                engine: Engine = Depends(obter_engine)):
    a = db.Alerta
    consulta = select(a).order_by(a.instante.desc()).limit(limite)
    if fonte:
        consulta = consulta.where(a.fonte == fonte)
    if recurso:
        consulta = consulta.where(a.recurso == recurso)
    with Session(engine) as s:
        return [{"fonte": x.fonte, "recurso": x.recurso, "metrica": x.metrica, "instante": x.instante,
                 "valor": x.valor, "escore": x.escore, "modelo": x.modelo} for x in s.scalars(consulta)]


@app.get("/api/resultados")
def api_resultados(engine: Engine = Depends(obter_engine)):
    """Resumo da fase 4 (média entre repetições), lido das tabelas experimentos/resultados."""
    e, r = db.Experimento, db.Resultado
    ultima = select(func.max(e.execucao)).scalar_subquery()
    consulta = (select(e.dataset, e.modelo, e.variante, func.count(distinct(r.repeticao)).label("repeticoes"),
                       *(func.avg(getattr(r, c)).label(c) for c in
                         ("precision", "recall", "f1", "fpr", "tempo_treino_s", "tempo_inferencia_ms")),
                       func.stddev_samp(r.f1).label("f1_desvio"))
                .join(r, r.experimento_id == e.id).where(e.execucao == ultima)
                .group_by(e.dataset, e.modelo, e.variante).order_by(e.dataset, e.variante, e.modelo))
    with Session(engine) as s:
        return [dict(x._mapping) for x in s.execute(consulta)]
