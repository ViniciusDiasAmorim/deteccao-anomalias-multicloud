"""Coletor OCI Monitoring para instâncias de computação.

As métricas do namespace oci_computeagent são publicadas pelo Oracle Cloud
Agent (plugin Compute Instance Monitoring), ativo nas imagens da Oracle,
inclusive nas instâncias Always Free. A consulta usa MQL com intervalo de 5 min.

Rede e disco são contadores cumulativos (crescem a cada sessão do agente e
zeram quando o sistema operacional reinicia), então a estatística usada é
increment() — a variação no intervalo, ou seja, bytes em 5 min. Incrementos
negativos (reinício do contador) são descartados. Quando a instância tem mais
de uma série para a mesma métrica (ex.: várias VNICs ou discos), os bytes são
somados e os percentuais viram média.

Limite do serviço: com intervalo de 5 min, a OCI só devolve os últimos 30 dias.

Credenciais: arquivo ~/.oci/config do oci-python-sdk. Permissão mínima: uma
política "allow group <grupo> to read metrics in compartment <compartimento>".
"""
from datetime import datetime, timedelta

import pandas as pd

from src.collectors.base import para_pontos, utc, vazio

NAMESPACE = "oci_computeagent"
# métrica canônica → (nome no OCI, estatística MQL)
METRICAS = {
    "cpu_util_percent": ("CpuUtilization", "mean"),
    "mem_util_percent": ("MemoryUtilization", "mean"),
    "net_in_bytes": ("NetworksBytesIn", "increment"),
    "net_out_bytes": ("NetworksBytesOut", "increment"),
    "disk_read_bytes": ("DiskBytesRead", "increment"),
    "disk_write_bytes": ("DiskBytesWritten", "increment"),
}
JANELA = timedelta(days=7)  # consulta em blocos para não esbarrar no limite de pontos por resposta


def cliente(arquivo_config: str = "~/.oci/config", perfil: str = "DEFAULT"):
    import oci
    return oci.monitoring.MonitoringClient(oci.config.from_file(arquivo_config, perfil))


def consulta(nome: str, estatistica: str, instancia: str) -> str:
    return f'{nome}[5m]{{resourceId = "{instancia}"}}.{estatistica}()'


def coletar(instancia: str, compartimento: str, inicio: datetime, fim: datetime, monitoring=None,
            metricas: list[str] | None = None) -> pd.DataFrame:
    from oci.monitoring.models import SummarizeMetricsDataDetails

    monitoring = monitoring or cliente()
    inicio, fim = utc(inicio), utc(fim)
    partes = []
    for canonica in metricas or list(METRICAS):
        nome, estatistica = METRICAS[canonica]
        pontos = []
        t = inicio
        while t < fim:
            detalhes = SummarizeMetricsDataDetails(
                namespace=NAMESPACE, query=consulta(nome, estatistica, instancia),
                start_time=t, end_time=min(t + JANELA, fim), resolution="5m",
            )
            resp = monitoring.summarize_metrics_data(compartimento, detalhes)
            for k, serie in enumerate(resp.data):
                pontos += [(k, p.timestamp, p.value) for p in serie.aggregated_datapoints]
            t += JANELA
        if not pontos:
            continue
        df = pd.DataFrame(pontos, columns=["serie", "instante", "valor"])
        if estatistica == "increment":
            df = df[df["valor"] >= 0]  # reinício do contador cumulativo
            combinado = df.groupby("instante")["valor"].sum()
        else:
            combinado = df.groupby("instante")["valor"].mean()
        partes.append(para_pontos("oci", instancia, canonica, combinado.index, combinado.to_numpy()))
    return pd.concat(partes, ignore_index=True) if partes else vazio()
