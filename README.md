# Detecção de anomalias em infraestrutura multi-cloud com machine learning

Código do Trabalho de Conclusão de Curso de Vinicius Dias Amorim de Matos, Bacharelado em Engenharia da Computação
(Faculdade Impacta, 2026), orientado pelo Prof. Me. André C. Rocha. Comparação de **Isolation Forest**, **LOF** e
**Random Forest** na detecção de anomalias em métricas de recursos computacionais, com dados públicos de
datacenters (Alibaba 2018, Google 2019), anomalias reais rotuladas (NAB, métricas do AWS CloudWatch), persistência
em PostgreSQL, coletores para AWS CloudWatch e OCI Monitoring e uma API FastAPI com painel de alertas.

## Estrutura

```
data/raw/          dados originais (fora do git; scripts/baixar_dados.sh)
data/processed/    séries derivadas e repetição 0 do pré-processamento
notebooks/         01_eda, 01b_eda_nab, 02_preprocessing, 03_modelos
src/
  data.py                     carga dos datasets (Zenodo, NAB) e cobertura do Alibaba
  build_alibaba_aggregate.py  agrega o machine_usage original com timestamp
  inject.py                   injeção de anomalias sintéticas
  preprocess.py               divisão temporal, configurações de injeção, features
  models.py                   IF, LOF, RF, GridSearchCV e regra F1 com teto de FPR
  evaluate.py                 métricas ponto a ponto e por evento
  experiments.py              fase 4: busca + 30 repetições
  case_study.py               estudo de caso das rajadas de falha do Alibaba
  detection.py                detecção em séries univariadas (API e coletores)
  db.py, load_db.py           PostgreSQL via SQLAlchemy
  collectors/                 aws.py, oci.py, coletar.py
  api/                        main.py (FastAPI) e static/index.html (painel)
results/figures, results/tables, results/experimentos
tests/             pytest (injeção, coletores, API)
scripts/           baixar_dados.sh, db.sh
```

## Reproduzir

Requisitos: Python 3.11+ (desenvolvido em 3.14.7), PostgreSQL 18 instalado (só os binários; não precisa do serviço
do sistema) e ~5 GB livres.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/baixar_dados.sh                                   # ~2 GB; confere MD5 do Zenodo
tar -xzOf data/raw/alibaba_v2018/machine_usage.tar.gz | .venv/bin/python -I src/build_alibaba_aggregate.py

scripts/db.sh init                                        # PostgreSQL do projeto em localhost:5433 (.pgdata/)
.venv/bin/python -m src.experiments --repeticoes 30       # fase 4 (~35 min em 4 núcleos)
.venv/bin/python -m src.load_db metricas
.venv/bin/python -m src.load_db experimentos

cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
cd .. && .venv/bin/python -m pytest

.venv/bin/python scripts/figura_arquitetura.py            # figura da arquitetura
scripts/captura_painel.sh                                 # captura do painel (requer Chrome)
```

Os números, tabelas e figuras citados no texto do TCC são os gerados por estes comandos em `results/`.

Todas as sementes são fixas (`random_state=42`; a repetição *r* da injeção usa `42 + 1000·r`). Configurações e
versões de cada execução ficam em `results/experimentos/execucao.json` e na tabela `experimentos` do banco.

## Banco de dados

`scripts/db.sh init` cria uma instância PostgreSQL só do projeto (dados em `.pgdata/`, porta 5433, apenas
localhost) e grava a URL de conexão em `.env` (fora do git). Use `scripts/db.sh start|stop|status|psql`.
Para usar outro servidor, basta trocar `DATABASE_URL` no `.env`.

Tabelas: `metricas` (séries de todas as fontes), `experimentos` e `resultados` (fase 4) e `alertas` (detecções da API).

## API e painel

```bash
.venv/bin/uvicorn src.api.main:app --port 8000
```

- Painel: <http://localhost:8000/>; documentação interativa: <http://localhost:8000/docs>
- `POST /api/detectar` — detecta anomalias numa série enviada no corpo (não grava nada)
- `POST /api/recursos/detectar` — detecta numa série do banco e grava os alertas
- `GET /api/recursos`, `/api/series`, `/api/alertas`, `/api/resultados`, `/saude`

A detecção usa o protocolo do NAB: o modelo aprende com os primeiros 15 % da série e marca anomalias no restante,
com os hiperparâmetros escolhidos no GridSearchCV do NAB.

## Coleta multi-cloud (AWS e OCI)

Os coletores normalizam as métricas para um esquema comum (`cpu_util_percent`, `mem_util_percent`,
`net_in_bytes`, `net_out_bytes`, `disk_read_bytes`, `disk_write_bytes`, em janelas de 5 min) e gravam na tabela
`metricas`. Os testes (`tests/test_collectors.py`) validam as chamadas contra o modelo oficial da API do
CloudWatch e os objetos reais do SDK da OCI, sem acessar a nuvem.

**AWS (free tier):** uma instância EC2 elegível ao free tier publica CPU, rede e disco a cada 5 min no
monitoramento básico do CloudWatch, sem custo. Memória exige o CloudWatch Agent. Crie um usuário IAM com permissão
apenas `cloudwatch:GetMetricStatistics` e configure `~/.aws/credentials` (`aws configure`). O coletor usa
`GetMetricStatistics`, que entra na cota gratuita do CloudWatch; o `GetMetricData` é cobrado.

```bash
.venv/bin/python -m src.collectors.coletar aws --instancia i-0123456789abcdef0 --regiao us-east-1 --horas 72 --detectar isolation_forest
```

**OCI (Always Free):** instâncias com imagem da Oracle já trazem o Oracle Cloud Agent, que publica as métricas do
namespace `oci_computeagent`. Gere uma chave de API para o usuário, configure `~/.oci/config` e crie a política
`allow group <grupo> to read metrics in compartment <compartimento>`.

```bash
.venv/bin/python -m src.collectors.coletar oci --instancia ocid1.instance.oc1... \
    --compartimento ocid1.compartment.oc1... --horas 72 --detectar isolation_forest
```

Credenciais nunca entram no repositório: ficam em `~/.aws`, `~/.oci` ou `.env` (ignorado pelo git).

## Dados e licenças

- Fernández-Montes e Fernández Cerero (2024), *DataCenter-Traces-Datasets*, Zenodo 10.5281/zenodo.14564935, CC-BY 4.0.
- Alibaba Group, *cluster-trace-v2018* (github.com/alibaba/clusterdata).
- Numenta Anomaly Benchmark (github.com/numenta/NAB), licença MIT, commit `ea702d7`.
