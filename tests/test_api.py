"""Testes da API com um banco PostgreSQL temporário (criado e apagado aqui).

A série usada é sintética e existe só no banco de teste: 10 dias com ciclo diário
e ruído, e três picos grandes depois do trecho de treino. O treino (15 % iniciais)
cobre mais de um ciclo diário, como acontece nas séries do NAB.
"""
import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from src import db
from src.api.main import app, obter_engine

BANCO_TESTE = "tcc_anomalias_teste"
N = 2880  # 10 dias de janelas de 5 min
PICOS = [1000, 1800, 2600]


def _serie() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-10-01", periods=N, freq="5min", tz="UTC")
    base = 20 + 5 * np.sin(2 * np.pi * np.arange(N) / 288) + rng.normal(0, 0.5, N)
    base[PICOS] += 70
    return pd.Series(base, index=idx)


@pytest.fixture(scope="module")
def engine():
    try:
        url = make_url(db.url_banco())
    except RuntimeError:
        pytest.skip("DATABASE_URL não definida")
    admin = f"host={url.host} port={url.port} user={url.username} password={url.password} dbname=postgres"
    try:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {BANCO_TESTE}")
            c.execute(f"CREATE DATABASE {BANCO_TESTE}")
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL indisponível (scripts/db.sh start)")
    eng = db.conectar(url.set(database=BANCO_TESTE).render_as_string(hide_password=False))
    db.criar_tabelas(eng)
    pontos = _serie().to_frame("valor").rename_axis("instante").reset_index()
    pontos = pontos.assign(fonte="teste", recurso="vm-teste", metrica="cpu_util_percent", deslocamento_s=None,
                           unidade="%")
    db.gravar_metricas(eng, pontos)
    yield eng
    eng.dispose()
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE {BANCO_TESTE}")


@pytest.fixture(scope="module")
def cliente(engine):
    app.dependency_overrides[obter_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_saude(cliente):
    assert cliente.get("/saude").json() == {"status": "ok", "banco": "ok"}


def test_painel_html(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_recursos_e_serie(cliente):
    recursos = cliente.get("/api/recursos").json()
    assert [(r["fonte"], r["recurso"], r["metrica"], r["pontos"]) for r in recursos] == \
        [("teste", "vm-teste", "cpu_util_percent", N)]
    serie = cliente.get("/api/series", params={"fonte": "teste", "recurso": "vm-teste",
                                               "metrica": "cpu_util_percent"}).json()
    assert len(serie) == N
    assert cliente.get("/api/series", params={"fonte": "x", "recurso": "y", "metrica": "z"}).status_code == 404


@pytest.mark.parametrize("modelo", ["isolation_forest", "lof", "random_forest"])
def test_detectar_sem_gravar(cliente, modelo):
    s = _serie()
    corpo = {"modelo": modelo, "pontos": [{"instante": i.isoformat(), "valor": float(v)} for i, v in s.items()]}
    r = cliente.post("/api/detectar", json=corpo)
    assert r.status_code == 200, r.text
    saida = r.json()
    assert saida["n_treino"] == int(0.15 * N)
    marcados = {i for i, p in enumerate(saida["pontos"]) if p["anomalia"]}
    assert not any(p["anomalia"] for p in saida["pontos"][:saida["n_treino"]])
    assert set(PICOS) <= marcados, f"{modelo} não marcou todos os picos: {sorted(marcados)[:20]}"
    falsos = len(marcados - set(PICOS)) / (N - saida["n_treino"] - len(PICOS))
    assert falsos < 0.05, f"{modelo}: {falsos:.1%} de falsos alarmes"


def test_detectar_valida_entrada(cliente):
    base = {"instante": "2026-10-01T00:00:00Z", "valor": 1.0}
    assert cliente.post("/api/detectar", json={"pontos": [base, {**base, "valor": -1}]}).status_code == 422
    assert cliente.post("/api/detectar", json={"pontos": [base, base]}).status_code == 422  # série curta
    assert cliente.post("/api/detectar", json={"pontos": [base] * 3, "modelo": "svm"}).status_code == 422


def test_detectar_recurso_grava_alertas(cliente):
    corpo = {"fonte": "teste", "recurso": "vm-teste", "metrica": "cpu_util_percent", "modelo": "lof"}
    saida = cliente.post("/api/recursos/detectar", json=corpo).json()
    filtro = {"recurso": "vm-teste", "limite": 5000}
    alertas = cliente.get("/api/alertas", params=filtro).json()
    assert len(alertas) == saida["n_anomalias"] > 0
    # rodar de novo substitui os alertas em vez de duplicar
    cliente.post("/api/recursos/detectar", json=corpo)
    assert len(cliente.get("/api/alertas", params=filtro).json()) == saida["n_anomalias"]


def test_resultados_responde_mesmo_sem_dados(cliente):
    r = cliente.get("/api/resultados")
    assert r.status_code == 200 and r.json() == []
