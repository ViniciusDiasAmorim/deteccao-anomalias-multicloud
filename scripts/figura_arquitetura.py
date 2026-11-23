"""Desenha a arquitetura do sistema (results/figures/fig_sistema_arquitetura.png).

Coordenadas em polegadas (1 unidade = 1 pol.), para que caixas e fontes fiquem proporcionais.
Uso:  .venv/bin/python scripts/figura_arquitetura.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SAIDA = Path(__file__).resolve().parents[1] / "results" / "figures" / "fig_sistema_arquitetura.png"
TEXTO, TEXTO2, BORDA, FUNDO, DESTAQUE, SETA = "#0b0b0b", "#52514e", "#8a8984", "#f6f5f2", "#2a78d6", "#52514e"
L, A = 7.5, 4.3


def caixa(ax, x, y, w, h, titulo, linhas, destaque=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="#eaf2fc" if destaque else FUNDO,
                                edgecolor=DESTAQUE if destaque else BORDA, linewidth=1.0))
    ax.text(x + w / 2, y + h - 0.12, titulo, ha="center", va="top", fontsize=8, fontweight="bold", color=TEXTO)
    ax.text(x + w / 2, y + h - 0.36, "\n".join(linhas), ha="center", va="top", fontsize=6.8, color=TEXTO2,
            linespacing=1.45)
    return x, y, w, h


def seta(ax, origem, destino, dupla=False):
    ax.add_patch(FancyArrowPatch(origem, destino, arrowstyle="<|-|>" if dupla else "-|>", mutation_scale=9,
                                 linewidth=0.9, color=SETA, shrinkA=1, shrinkB=1))


def rotulo(ax, x, y, texto, ha="center"):
    ax.text(x, y, texto, ha=ha, va="bottom", fontsize=6.3, color=TEXTO2, style="italic")


fig = plt.figure(figsize=(L, A))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, L)
ax.set_ylim(0, A)
ax.axis("off")

X = {"fontes": 0.05, "ingestao": 2.30, "banco": 4.30, "analise": 5.95}
W = {"fontes": 1.75, "ingestao": 1.55, "banco": 1.05, "analise": 1.50}
for col, nome in [("fontes", "Fontes de dados"), ("ingestao", "Ingestão"), ("banco", "Persistência"),
                  ("analise", "Análise e serviço")]:
    ax.text(X[col] + W[col] / 2, A - 0.08, nome.upper(), ha="center", va="top", fontsize=7, color=TEXTO2,
            fontweight="bold")

f = X["fontes"], W["fontes"]
traces = caixa(ax, f[0], 3.10, f[1], 0.82, "Traces públicos", ["Alibaba 2018 e Google 2019", "Zenodo (CC-BY 4.0)"])
nab = caixa(ax, f[0], 2.15, f[1], 0.82, "NAB", ["métricas reais do CloudWatch", "com anomalias rotuladas"])
aws = caixa(ax, f[0], 1.20, f[1], 0.82, "AWS CloudWatch", ["instância EC2 (free tier)", "GetMetricStatistics"])
oci = caixa(ax, f[0], 0.25, f[1], 0.82, "OCI Monitoring", ["instância Always Free", "consultas MQL"])

i = X["ingestao"], W["ingestao"]
carga = caixa(ax, i[0], 2.45, i[1], 1.20, "Carga dos datasets", ["src/data.py", "src/load_db.py", "formato longo, UTC"])
coletores = caixa(ax, i[0], 0.55, i[1], 1.20, "Coletores", ["boto3 e oci-python-sdk", "esquema comum", "janelas de 5 min"])

b = X["banco"], W["banco"]
banco = caixa(ax, b[0], 0.85, b[1], 2.50, "PostgreSQL", ["metricas", "experimentos", "resultados", "alertas", "",
                                                        "via SQLAlchemy"], destaque=True)

a = X["analise"], W["analise"]
ml = caixa(ax, a[0], 2.30, a[1], 1.62, "Pipeline de ML", ["pré-processamento", "injeção de anomalias", "IF · LOF · RF",
                                                         "GridSearchCV", "30 repetições"])
api = caixa(ax, a[0], 0.25, a[1], 1.55, "API FastAPI", ["/api/detectar", "/api/alertas", "/api/resultados",
                                                       "painel de alertas"])

direita = lambda c, dy=0.5: (c[0] + c[2], c[1] + c[3] * dy)
esquerda = lambda c, dy=0.5: (c[0], c[1] + c[3] * dy)
seta(ax, direita(traces), esquerda(carga, 0.72))
seta(ax, direita(nab), esquerda(carga, 0.28))
seta(ax, direita(aws), esquerda(coletores, 0.72))
seta(ax, direita(oci), esquerda(coletores, 0.28))
seta(ax, direita(carga), esquerda(banco, 0.80))
rotulo(ax, (direita(carga)[0] + banco[0]) / 2, 3.12, "séries")
seta(ax, direita(coletores), esquerda(banco, 0.12))
rotulo(ax, (direita(coletores)[0] + banco[0]) / 2, 1.20, "coletas")
seta(ax, esquerda(ml, 0.35), direita(banco, 0.73))
rotulo(ax, (banco[0] + banco[2] + ml[0]) / 2, 2.95, "resultados")
seta(ax, esquerda(api, 0.75), direita(banco, 0.25), dupla=True)
rotulo(ax, (banco[0] + banco[2] + api[0]) / 2, 1.43, "séries e\nalertas")
seta(ax, (ml[0] + ml[2] / 2, ml[1]), (api[0] + api[2] / 2, api[1] + api[3]))
rotulo(ax, ml[0] + ml[2] / 2 + 0.06, 1.95, "hiperparâmetros\ndo NAB", ha="left")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SAIDA, dpi=300)
print(SAIDA)
