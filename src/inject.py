"""Injeção de anomalias sintéticas em séries de métricas.

Cada anomalia é um *evento*: um trecho contíguo de janelas em que uma ou mais
colunas recebem uma perturbação aditiva. A magnitude é dada em múltiplos de um
desvio padrão robusto (1,4826 × MAD) calculado numa série de referência — sempre
o trecho de treino sem injeção —, o que torna a mesma configuração aplicável a
métricas com unidades diferentes.

A magnitude também pode ser relativa (`unidade="fracao"`): uma queda de 0,3
remove 30 % do valor corrente. É a forma usada para quedas, que em desvios
padrão levariam a métrica a zero com facilidade.

Formas disponíveis:
- "pico": salto constante durante poucas janelas;
- "patamar": salto constante sustentado por mais tempo;
- "rampa": cresce linearmente de zero até a magnitude (ex.: vazamento de memória).
Sinal negativo produz quedas.

Os eventos só são colocados em janelas "normais": fora de `excluir` e sem valores
já extremos (|z robusto| > z_max_candidato) nas colunas afetadas.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

FORMAS = ("pico", "patamar", "rampa")


@dataclass(frozen=True)
class TipoAnomalia:
    nome: str
    colunas: tuple[str, ...]
    forma: str
    duracao: tuple[int, int]        # nº de janelas, intervalo fechado
    magnitude: tuple[float, float]  # em desvios padrão robustos ou fração do valor (ver unidade)
    sinal: int = 1
    unidade: str = "sigma"          # "sigma" ou "fracao"

    def __post_init__(self):
        if self.forma not in FORMAS:
            raise ValueError(f"forma inválida: {self.forma}")
        if self.sinal not in (1, -1):
            raise ValueError("sinal deve ser 1 ou -1")
        if self.unidade not in ("sigma", "fracao"):
            raise ValueError(f"unidade inválida: {self.unidade}")


@dataclass(frozen=True)
class ConfigInjecao:
    tipos: tuple[TipoAnomalia, ...]
    fracao_alvo: float = 0.05     # fração aproximada de janelas anômalas
    espaco_minimo: int = 12       # janelas normais obrigatórias entre eventos
    z_max_candidato: float = 3.0
    random_state: int = 42
    limites: dict = field(default_factory=dict)  # coluna -> (mín, máx); None = sem limite

    def to_dict(self) -> dict:
        return asdict(self)


def escala_robusta(ref: pd.DataFrame) -> pd.Series:
    """1,4826 × MAD por coluna; usa o desvio padrão quando o MAD é zero (séries quase constantes)."""
    mad = (ref - ref.median()).abs().median() * 1.4826
    return mad.where(mad > 0, ref.std())


def _perfil(forma: str, duracao: int) -> np.ndarray:
    if forma == "rampa":
        return np.linspace(1 / duracao, 1.0, duracao)
    return np.ones(duracao)


def injetar(
    df: pd.DataFrame,
    config: ConfigInjecao,
    referencia: pd.DataFrame,
    excluir: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Injeta anomalias em uma cópia de `df`.

    Retorna (df_injetado, marcacao, eventos):
    - marcacao: colunas `rotulo` (0/1), `tipo_anomalia` e `evento_id` por janela;
    - eventos: uma linha por evento, com posição, duração, magnitude e colunas.
    """
    rng = np.random.default_rng(config.random_state)
    n = len(df)
    escala = escala_robusta(referencia)
    mediana = referencia.median()

    colunas_tipos = sorted({c for t in config.tipos for c in t.colunas})
    z = (df[colunas_tipos] - mediana[colunas_tipos]).abs() / escala[colunas_tipos]
    livre = np.ones(n, dtype=bool)
    if excluir is not None:
        livre &= ~excluir.to_numpy(dtype=bool)

    out = df.copy()
    rotulo = np.zeros(n, dtype=int)
    tipo = np.full(n, "", dtype=object)
    evento_id = np.zeros(n, dtype=int)
    eventos = []
    ocupado = np.zeros(n, dtype=bool)  # eventos + margem de espaco_minimo
    alvo = int(round(config.fracao_alvo * n))

    # Tipos sorteados em ciclos (permutação aleatória a cada ciclo) para que todos
    # apareçam, mesmo quando um tipo longo consome boa parte do orçamento.
    fila: list[int] = []
    ciclos = 0
    while rotulo.sum() < alvo and ciclos < 1_000:
        if not fila:
            fila = list(rng.permutation(len(config.tipos)))
            ciclos += 1
        t = config.tipos[fila.pop(0)]
        for _ in range(500):  # tentativas de posicionar este evento
            dur = int(rng.integers(t.duracao[0], t.duracao[1] + 1))
            if dur > n:
                continue
            ini = int(rng.integers(0, n - dur + 1))
            fim = ini + dur
            if ocupado[ini:fim].any() or not livre[ini:fim].all():
                continue
            if (z[list(t.colunas)].iloc[ini:fim] > config.z_max_candidato).any().any():
                continue

            mag = float(rng.uniform(*t.magnitude))
            perfil = _perfil(t.forma, dur)
            novos = {}
            for c in t.colunas:
                antes = out[c].iloc[ini:fim].to_numpy()
                lo, hi = config.limites.get(c, (None, None))
                passo = antes if t.unidade == "fracao" else escala[c]
                novos[c] = np.clip(antes + t.sinal * mag * passo * perfil, lo, hi)
            if all(np.allclose(out[c].iloc[ini:fim].to_numpy(), v) for c, v in novos.items()):
                continue  # perturbação anulada pelo limite (ex.: queda em série já em zero)
            for c, v in novos.items():
                out.iloc[ini:fim, out.columns.get_loc(c)] = v

            k = len(eventos) + 1
            rotulo[ini:fim] = 1
            tipo[ini:fim] = t.nome
            evento_id[ini:fim] = k
            ocupado[max(0, ini - config.espaco_minimo):min(n, fim + config.espaco_minimo)] = True
            eventos.append({"evento_id": k, "tipo": t.nome, "forma": t.forma, "colunas": ",".join(t.colunas),
                            "inicio": df.index[ini], "posicao": ini, "duracao": dur,
                            "magnitude": mag, "unidade": t.unidade, "sinal": t.sinal})
            break

    marcacao = pd.DataFrame({"rotulo": rotulo, "tipo_anomalia": tipo, "evento_id": evento_id}, index=df.index)
    return out, marcacao, pd.DataFrame(eventos)
