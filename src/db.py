"""Persistência em PostgreSQL via SQLAlchemy.

Tabelas:
- metricas: pontos das séries (formato longo), de qualquer fonte — traces
  (alibaba, google), NAB e coletores de nuvem (aws, oci). Os traces não têm
  horário real: guardam `deslocamento_s` (segundos desde o início) e deixam
  `instante` nulo; as demais fontes guardam `instante`.
- experimentos: uma linha por (dataset, modelo, variante) avaliado na fase 4,
  com hiperparâmetros, features e versões das bibliotecas.
- resultados: uma linha por repetição de cada experimento.
- alertas: anomalias detectadas pela API sobre métricas coletadas.

A URL de conexão vem de DATABASE_URL no .env (ver scripts/db.sh).
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (BigInteger, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint,
                        create_engine, delete, func, select)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

ROOT = Path(__file__).resolve().parents[1]


class Base(DeclarativeBase):
    pass


class Metrica(Base):
    __tablename__ = "metricas"
    __table_args__ = (
        UniqueConstraint("fonte", "recurso", "metrica", "instante", "deslocamento_s", name="uq_metrica_ponto",
                         postgresql_nulls_not_distinct=True),
        CheckConstraint("(instante IS NULL) <> (deslocamento_s IS NULL)", name="ck_metrica_um_tempo"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fonte: Mapped[str] = mapped_column(String(20), index=True)
    recurso: Mapped[str] = mapped_column(String(120))
    metrica: Mapped[str] = mapped_column(String(60))
    instante: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deslocamento_s: Mapped[int | None] = mapped_column(BigInteger)
    valor: Mapped[float] = mapped_column(Float)
    unidade: Mapped[str | None] = mapped_column(String(30))


class Experimento(Base):
    __tablename__ = "experimentos"
    __table_args__ = (UniqueConstraint("dataset", "modelo", "variante", "execucao", name="uq_experimento"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    execucao: Mapped[str] = mapped_column(String(40))  # início da execução (UTC, ISO)
    dataset: Mapped[str] = mapped_column(String(20))
    modelo: Mapped[str] = mapped_column(String(30))
    variante: Mapped[str] = mapped_column(String(30))
    hiperparametros: Mapped[dict] = mapped_column(JSONB)
    configuracao: Mapped[dict] = mapped_column(JSONB)  # teto de FPR, dobras, sementes, versões
    registrado_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    resultados: Mapped[list["Resultado"]] = relationship(back_populates="experimento", cascade="all, delete-orphan")


class Resultado(Base):
    __tablename__ = "resultados"
    __table_args__ = (UniqueConstraint("experimento_id", "repeticao", name="uq_resultado"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experimento_id: Mapped[int] = mapped_column(ForeignKey("experimentos.id", ondelete="CASCADE"), index=True)
    repeticao: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    fpr: Mapped[float] = mapped_column(Float)
    tempo_treino_s: Mapped[float] = mapped_column(Float)
    tempo_inferencia_ms: Mapped[float] = mapped_column(Float)
    detalhes: Mapped[dict] = mapped_column(JSONB)  # contagens (tp, fp, ...) e campos específicos do dataset

    experimento: Mapped[Experimento] = relationship(back_populates="resultados")


class Alerta(Base):
    __tablename__ = "alertas"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fonte: Mapped[str] = mapped_column(String(20))
    recurso: Mapped[str] = mapped_column(String(120))
    metrica: Mapped[str] = mapped_column(String(60))
    instante: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valor: Mapped[float] = mapped_column(Float)
    escore: Mapped[float] = mapped_column(Float)
    modelo: Mapped[str] = mapped_column(String(30))
    detectado_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def url_banco() -> str:
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL não definida; rode scripts/db.sh init")
    return url


def conectar(url: str | None = None) -> Engine:
    return create_engine(url or url_banco())


def criar_tabelas(engine: Engine) -> None:
    Base.metadata.create_all(engine)


# --- Carga de métricas -----------------------------------------------------------

def gravar_metricas(engine: Engine, pontos: pd.DataFrame, substituir_fonte: bool = False) -> int:
    """Grava pontos no formato longo (colunas de Metrica, exceto id).

    Com substituir_fonte=True, apaga antes os pontos das fontes presentes em
    `pontos` (carga idempotente dos datasets). Sem isso, pontos repetidos são
    ignorados (ON CONFLICT DO NOTHING), o que permite rodar um coletor várias
    vezes sobre a mesma janela de tempo.
    """
    from sqlalchemy.dialects.postgresql import insert

    cols = ["fonte", "recurso", "metrica", "instante", "deslocamento_s", "valor", "unidade"]
    registros = pontos.reindex(columns=cols).astype(object).where(pontos.reindex(columns=cols).notna(), None)
    linhas = registros.to_dict("records")
    with Session(engine) as s, s.begin():
        if substituir_fonte:
            s.execute(delete(Metrica).where(Metrica.fonte.in_(pontos["fonte"].unique().tolist())))
        # 5.000 linhas × 7 colunas fica abaixo do limite de 65.535 parâmetros por comando do PostgreSQL.
        for i in range(0, len(linhas), 5_000):
            stmt = insert(Metrica).values(linhas[i:i + 5_000]).on_conflict_do_nothing(constraint="uq_metrica_ponto")
            s.execute(stmt)
    return len(linhas)


def series_para_pontos(df: pd.DataFrame, fonte: str, recurso: str, unidades: dict[str, str]) -> pd.DataFrame:
    """Converte uma tabela larga (índice de tempo, uma coluna por métrica) para o formato longo."""
    longo = df.rename_axis("t").reset_index().melt(id_vars="t", var_name="metrica", value_name="valor")
    if isinstance(df.index, pd.TimedeltaIndex):
        longo["deslocamento_s"] = (longo["t"] / pd.Timedelta(seconds=1)).astype("int64")
        longo["instante"] = None
    else:
        idx = pd.to_datetime(longo["t"])
        longo["instante"] = idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx
        longo["deslocamento_s"] = None
    longo["fonte"] = fonte
    longo["recurso"] = recurso
    longo["unidade"] = longo["metrica"].map(unidades)
    return longo.drop(columns="t")


# --- Experimentos ----------------------------------------------------------------

def gravar_experimentos(engine: Engine, execucao: dict, resultados: pd.DataFrame) -> int:
    """Grava a execução da fase 4 (results/experimentos/execucao.json + resultados_*.csv).

    Idempotente: uma nova carga da mesma execução substitui a anterior.
    """
    principais = ["precision", "recall", "f1", "fpr", "tempo_treino_s", "tempo_inferencia_ms"]
    chave = ["dataset", "modelo", "variante", "repeticao"]
    config = {k: execucao[k] for k in ("teto_fpr", "n_dobras_cv", "seed", "repeticoes", "python", "bibliotecas",
                                       "maquina")}
    n = 0
    with Session(engine) as s, s.begin():
        s.execute(delete(Experimento).where(Experimento.execucao == execucao["inicio_utc"]))
        for (ds, modelo, variante), g in resultados.groupby(["dataset", "modelo", "variante"]):
            exp = Experimento(execucao=execucao["inicio_utc"], dataset=ds, modelo=modelo, variante=variante,
                              hiperparametros=execucao["melhores_parametros"][ds][modelo], configuracao=config)
            for _, linha in g.iterrows():
                extras = {k: _json(v) for k, v in linha.items() if k not in principais + chave and pd.notna(v)}
                exp.resultados.append(Resultado(repeticao=int(linha["repeticao"]),
                                                **{k: float(linha[k]) for k in principais}, detalhes=extras))
                n += 1
            s.add(exp)
    return n


def gravar_alertas(engine: Engine, deteccao: pd.DataFrame, fonte: str, recurso: str, metrica: str,
                   modelo: str) -> int:
    """Substitui os alertas de (fonte, recurso, métrica, modelo) pelas anomalias de `deteccao`."""
    anomalias = deteccao[deteccao["anomalia"]]
    with Session(engine) as s, s.begin():
        s.execute(delete(Alerta).where(Alerta.fonte == fonte, Alerta.recurso == recurso, Alerta.metrica == metrica,
                                       Alerta.modelo == modelo))
        s.add_all(Alerta(fonte=fonte, recurso=recurso, metrica=metrica, modelo=modelo,
                         instante=linha.instante.to_pydatetime(), valor=float(linha.valor), escore=float(linha.escore))
                  for linha in anomalias.itertuples())
    return len(anomalias)


def ler_serie(engine: Engine, fonte: str, recurso: str, metrica: str) -> pd.Series:
    """Série com horário real (instante) de um recurso, em ordem cronológica."""
    with Session(engine) as s:
        linhas = s.execute(select(Metrica.instante, Metrica.valor)
                           .where(Metrica.fonte == fonte, Metrica.recurso == recurso, Metrica.metrica == metrica,
                                  Metrica.instante.is_not(None))
                           .order_by(Metrica.instante)).all()
    if not linhas:
        return pd.Series(dtype=float)
    instantes, valores = zip(*linhas)
    return pd.Series(valores, index=pd.DatetimeIndex(instantes, tz="UTC"), dtype=float)


def _json(v):
    if hasattr(v, "item"):
        return v.item()
    return v


def ler_execucao(caminho: Path) -> dict:
    return json.loads(caminho.read_text())


def contar(engine: Engine, modelo) -> int:
    with Session(engine) as s:
        return s.scalar(select(func.count()).select_from(modelo))
