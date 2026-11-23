"""Testes das regras da injeção de anomalias (src/inject.py)."""
import numpy as np
import pandas as pd
import pytest

from src.inject import ConfigInjecao, TipoAnomalia, escala_robusta, injetar

N = 2000


def _df(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.to_timedelta(np.arange(N) * 300, unit="s")
    return pd.DataFrame({"cpu": 40 + rng.normal(0, 5, N), "mem": 80 + rng.normal(0, 2, N)}, index=idx)


TIPOS = (
    TipoAnomalia("pico", ("cpu",), "pico", (1, 3), (3, 5)),
    TipoAnomalia("patamar", ("mem",), "patamar", (6, 24), (2, 4)),
    TipoAnomalia("queda", ("cpu",), "patamar", (6, 12), (0.3, 0.6), sinal=-1, unidade="fracao"),
)


def _config(**kw):
    return ConfigInjecao(TIPOS, random_state=7, limites={"cpu": (0, 100), "mem": (0, 100)}, **kw)


def test_rotulos_coincidem_com_valores_alterados():
    df = _df()
    out, marc, ev = injetar(df, _config(), referencia=df)
    alterado = (out != df).any(axis=1).to_numpy()
    assert (alterado == (marc["rotulo"] == 1).to_numpy()).all()
    assert marc["rotulo"].sum() == ev["duracao"].sum()


def test_fracao_alvo_e_todos_os_tipos_aparecem():
    df = _df()
    _, marc, ev = injetar(df, _config(), referencia=df)
    assert 0.05 <= marc["rotulo"].mean() <= 0.07
    assert set(ev["tipo"]) == {"pico", "patamar", "queda"}


def test_espaco_minimo_entre_eventos():
    df = _df()
    _, _, ev = injetar(df, _config(espaco_minimo=12), referencia=df)
    ev = ev.sort_values("posicao")
    folga = ev["posicao"].iloc[1:].to_numpy() - (ev["posicao"] + ev["duracao"]).iloc[:-1].to_numpy()
    assert (folga >= 12).all()


def test_nao_injeta_em_janelas_excluidas():
    df = _df()
    excluir = pd.Series(False, index=df.index)
    excluir.iloc[100:900] = True
    _, marc, _ = injetar(df, _config(), referencia=df, excluir=excluir)
    assert marc.loc[excluir.to_numpy(), "rotulo"].sum() == 0


def test_mesma_semente_mesmo_resultado_e_sementes_diferentes_mudam():
    df = _df()
    a = injetar(df, _config(), referencia=df)
    b = injetar(df, _config(), referencia=df)
    c = injetar(df, ConfigInjecao(TIPOS, random_state=8, limites={"cpu": (0, 100)}), referencia=df)
    assert a[0].equals(b[0]) and a[2].equals(b[2])
    assert not a[1]["rotulo"].equals(c[1]["rotulo"])


def test_magnitudes_sigma_e_fracao():
    df = _df()
    out, marc, ev = injetar(df, _config(), referencia=df)
    escala = escala_robusta(df)
    for e in ev.itertuples():
        col = e.colunas
        trecho = slice(e.posicao, e.posicao + e.duracao)
        delta = out[col].iloc[trecho].to_numpy() - df[col].iloc[trecho].to_numpy()
        if e.unidade == "fracao":
            assert np.allclose(delta, -e.magnitude * df[col].iloc[trecho].to_numpy())
        else:
            assert np.allclose(delta, e.magnitude * escala[col])


def test_limites_respeitados():
    df = _df()
    cfg = ConfigInjecao((TipoAnomalia("enorme", ("cpu",), "pico", (1, 2), (50, 60)),), random_state=1,
                        limites={"cpu": (0, 100)})
    out, _, _ = injetar(df, cfg, referencia=df)
    assert out["cpu"].max() <= 100


def test_tipo_invalido():
    with pytest.raises(ValueError):
        TipoAnomalia("x", ("cpu",), "degrau", (1, 2), (1, 2))
