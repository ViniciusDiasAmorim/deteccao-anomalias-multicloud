"""Pré-processamento: divisão temporal, injeção e construção de features.

Ordem das etapas (a mesma para os três datasets):
1. divisão temporal treino/teste, sem embaralhar (séries autocorrelacionadas
   vazariam informação entre treino e teste numa divisão aleatória);
2. injeção de anomalias em cada trecho, com configurações diferentes para treino
   e teste, para não medir memorização do Random Forest. No NAB o teste usa os rótulos reais;
3. features: métricas brutas, diferenças de 1ª ordem e contexto horário.

A normalização (StandardScaler) NÃO é aplicada aqui: ela entra no Pipeline do
modelo (fase 4) para ser ajustada só nos dados de treino de cada dobra da
validação cruzada.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from src import data
from src.inject import ConfigInjecao, TipoAnomalia, injetar

SEED = 42
# A injeção é repetida com sementes diferentes; a fase 4 reporta média e desvio
# padrão das métricas entre repetições. A repetição r usa as sementes
# SEED + 1000·r (treino) e SEED + 1000·r + 1 (teste).
REPETICOES = 30


@dataclass(frozen=True)
class ConfigFeatures:
    brutas: tuple[str, ...]
    diferencas: tuple[str, ...] = ()
    log1p: bool = False
    contexto_horario: bool = True


@dataclass(frozen=True)
class Experimento:
    nome: str
    fracao_treino: float
    features: ConfigFeatures
    injecao_treino: ConfigInjecao
    injecao_teste: ConfigInjecao | None  # None: teste usa rótulos reais (NAB)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["injecao_treino"] = self.injecao_treino.to_dict()
        d["injecao_teste"] = None if self.injecao_teste is None else self.injecao_teste.to_dict()
        return d


# Rótulos dos tipos de anomalia para figuras e tabelas.
ROTULOS_TIPO = {
    "pico_cpu": "pico de CPU", "memoria_sustentada": "memória sustentada", "rajada_rede": "rajada de rede",
    "vazamento_memoria": "vazamento de memória", "queda_cpu": "queda de CPU", "pico": "pico", "patamar": "patamar",
}

# --- Configurações -----------------------------------------------------------
# Treino e teste usam magnitudes e durações diferentes, e o teste inclui dois
# tipos que o treino não tem (vazamento de memória e queda de CPU). Assim o
# Random Forest não é avaliado no mesmo padrão que memorizou.

def _tipos_alibaba(teste: bool) -> tuple[TipoAnomalia, ...]:
    base = (
        TipoAnomalia("pico_cpu", ("cpu_util_percent",), "pico", (1, 3), (2, 6) if teste else (3, 5)),
        TipoAnomalia("memoria_sustentada", ("mem_util_percent",), "patamar",
                     (4, 36) if teste else (6, 24), (1.5, 5) if teste else (2, 4)),
        TipoAnomalia("rajada_rede", ("net_in", "net_out"), "pico", (1, 8) if teste else (2, 6), (2, 6) if teste else (3, 5)),
    )
    if not teste:
        return base
    return base + (
        TipoAnomalia("vazamento_memoria", ("mem_util_percent",), "rampa", (12, 48), (2, 5)),
        TipoAnomalia("queda_cpu", ("cpu_util_percent",), "patamar", (6, 18), (0.3, 0.6), sinal=-1, unidade="fracao"),
    )


def _tipos_google(teste: bool) -> tuple[TipoAnomalia, ...]:
    base = (
        TipoAnomalia("pico_cpu", ("avg_cpu",), "pico", (1, 3), (2, 6) if teste else (3, 5)),
        TipoAnomalia("memoria_sustentada", ("avg_mem",), "patamar",
                     (4, 36) if teste else (6, 24), (1.5, 5) if teste else (2, 4)),
    )
    if not teste:
        return base
    return base + (
        TipoAnomalia("vazamento_memoria", ("avg_mem",), "rampa", (12, 48), (2, 5)),
        TipoAnomalia("queda_cpu", ("avg_cpu",), "patamar", (6, 18), (0.3, 0.6), sinal=-1, unidade="fracao"),
    )


def _tipos_nab() -> tuple[TipoAnomalia, ...]:
    return (
        TipoAnomalia("pico", ("value",), "pico", (1, 3), (3, 6)),
        TipoAnomalia("patamar", ("value",), "patamar", (6, 24), (2, 4)),
    )


ALIBABA = Experimento(
    nome="alibaba",
    fracao_treino=0.7,
    # net_in/net_out brutos são uma rampa ao longo do trace (EDA, seção 5): entram só como variação.
    features=ConfigFeatures(brutas=("cpu_util_percent", "mem_util_percent", "disk_io_percent"),
                            diferencas=("net_in", "net_out")),
    injecao_treino=ConfigInjecao(_tipos_alibaba(False), random_state=SEED,
                                 limites={c: (0, 100) for c in ("cpu_util_percent", "mem_util_percent", "net_in", "net_out")}),
    injecao_teste=ConfigInjecao(_tipos_alibaba(True), random_state=SEED + 1,
                                limites={c: (0, 100) for c in ("cpu_util_percent", "mem_util_percent", "net_in", "net_out")}),
)

GOOGLE = Experimento(
    nome="google",
    fracao_treino=0.7,
    features=ConfigFeatures(brutas=("avg_cpu", "avg_mem", "avg_assigned_mem")),
    injecao_treino=ConfigInjecao(_tipos_google(False), random_state=SEED, limites={"avg_cpu": (0, 1), "avg_mem": (0, 1)}),
    injecao_teste=ConfigInjecao(_tipos_google(True), random_state=SEED + 1, limites={"avg_cpu": (0, 1), "avg_mem": (0, 1)}),
)

NAB = Experimento(
    nome="nab",
    fracao_treino=data.NAB_PROBATIONARY,
    # Valores de várias ordens de grandeza (bytes, contagens): log1p evita que poucos picos dominem a escala.
    features=ConfigFeatures(brutas=("value",), diferencas=("value",), log1p=True),
    injecao_treino=ConfigInjecao(_tipos_nab(), random_state=SEED, limites={"value": (0, None)}),
    injecao_teste=None,
)


# --- Etapas ------------------------------------------------------------------

def dividir(n: int, fracao_treino: float) -> np.ndarray:
    """Máscara booleana: True para as primeiras `fracao_treino` janelas (treino)."""
    corte = int(fracao_treino * n)
    return np.arange(n) < corte


def _horas(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.TimedeltaIndex):
        return (index / pd.Timedelta(hours=1)).to_numpy() % 24
    return (index.hour + index.minute / 60).to_numpy()


def construir_features(df: pd.DataFrame, cfg: ConfigFeatures) -> pd.DataFrame:
    base = np.log1p(df) if cfg.log1p else df
    feats = pd.DataFrame(index=df.index)
    for c in cfg.brutas:
        feats[c] = base[c]

    passo = df.index.to_series().diff()
    salto = (passo > 2 * passo.median()).to_numpy()  # ex.: lacuna de 5 h do Alibaba
    for c in cfg.diferencas:
        d = base[c].diff().to_numpy(copy=True)
        d[0] = 0.0
        d[salto] = 0.0  # variação através de uma lacuna não é informativa
        feats[f"{c}_diff"] = d

    if cfg.contexto_horario:
        ang = 2 * np.pi * _horas(df.index) / 24
        feats["hora_sin"] = np.sin(ang)
        feats["hora_cos"] = np.cos(ang)
    return feats


def _semente(cfg: ConfigInjecao, repeticao: int) -> ConfigInjecao:
    return replace(cfg, random_state=cfg.random_state + 1000 * repeticao)


def injetar_trechos(exp: Experimento, df: pd.DataFrame, excluir: pd.Series | None = None,
                    rotulo_real: pd.Series | None = None, repeticao: int = 0):
    """Divisão temporal + injeção, sem features.

    Retorna (valores brutos com anomalias, marcação, eventos, máscara de treino).
    A marcação tem `rotulo`, `tipo_anomalia` e `evento_id`. Para o NAB, no teste,
    `evento_id` é o número da janela oficial e `tipo_anomalia` vale "real".
    """
    if excluir is None:
        excluir = pd.Series(False, index=df.index)
    treino = dividir(len(df), exp.fracao_treino)
    ref = df[treino]

    df_tr, marc_tr, ev_tr = injetar(df[treino], _semente(exp.injecao_treino, repeticao), ref, excluir[treino])
    if exp.injecao_teste is not None:
        df_te, marc_te, ev_te = injetar(df[~treino], _semente(exp.injecao_teste, repeticao), ref, excluir[~treino])
    else:
        df_te = df[~treino]
        jan = rotulo_real[~treino].astype(int)
        marc_te = pd.DataFrame({"rotulo": (jan > 0).astype(int),
                                "tipo_anomalia": np.where(jan > 0, "real", ""),
                                "evento_id": jan}, index=df_te.index)
        ev_te = pd.DataFrame()

    eventos = pd.concat([ev_tr.assign(trecho="treino"), ev_te.assign(trecho="teste")], ignore_index=True)
    return pd.concat([df_tr, df_te]), pd.concat([marc_tr, marc_te]), eventos, treino


def preparar(exp: Experimento, df: pd.DataFrame, excluir: pd.Series | None = None,
             rotulo_real: pd.Series | None = None, repeticao: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divisão + injeção + features para uma série.

    Retorna (tabela, eventos). A tabela tem as features, `treino` (bool), a
    marcação de injetar_trechos e `excluir`.
    """
    injetado, marcacao, eventos, treino = injetar_trechos(exp, df, excluir, rotulo_real, repeticao)
    tabela = construir_features(injetado, exp.features)
    tabela["treino"] = treino
    tabela = tabela.join(marcacao)
    tabela["excluir"] = False if excluir is None else excluir.to_numpy(dtype=bool)
    return tabela, eventos


# --- Datasets ----------------------------------------------------------------

COBERTURA_MINIMA = 2000  # máquinas; abaixo disso a média não representa o cluster


def alibaba_cobertura_baixa() -> pd.Series:
    cobertura = data.load_alibaba_coverage().reindex(data.load_alibaba().index)
    return cobertura < COBERTURA_MINIMA


def preparar_alibaba(repeticao: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data.load_alibaba()
    tabela, eventos = preparar(ALIBABA, df, excluir=alibaba_cobertura_baixa(), repeticao=repeticao)
    cobertura = data.load_alibaba_coverage().reindex(df.index)
    tabela.insert(len(tabela.columns) - 1, "n_maquinas", cobertura.to_numpy())
    return tabela, eventos


def preparar_google(repeticao: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    return preparar(GOOGLE, data.load_google(), repeticao=repeticao)


def preparar_nab(repeticao: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabelas, eventos = [], []
    for key in data.list_nab_series():
        serie = data.load_nab(key)
        nome = key.split("/")[1].removesuffix(".csv")
        tab, ev = preparar(NAB, serie[["value"]], rotulo_real=serie["janela_anomalia"], repeticao=repeticao)
        tabelas.append(tab.assign(serie=nome))
        eventos.append(ev.assign(serie=nome))
    return pd.concat(tabelas), pd.concat(eventos, ignore_index=True)
