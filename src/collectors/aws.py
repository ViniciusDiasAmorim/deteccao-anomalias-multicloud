"""Coletor AWS CloudWatch para instâncias EC2.

Usa GetMetricStatistics, que entra na cota gratuita do CloudWatch (o
GetMetricData não entra). O monitoramento básico do EC2, gratuito, publica
métricas a cada 5 min. Memória não é publicada pelo EC2: só existe se o
CloudWatch Agent estiver instalado na instância (namespace CWAgent).

Credenciais: as do boto3 (~/.aws/credentials, variáveis de ambiente ou perfil).
Permissão mínima: cloudwatch:GetMetricStatistics.
"""
from datetime import datetime, timedelta

import pandas as pd

from src.collectors.base import PERIODO_S, para_pontos, utc, vazio

# métrica canônica → (namespace, nome no CloudWatch, estatística)
METRICAS = {
    "cpu_util_percent": ("AWS/EC2", "CPUUtilization", "Average"),
    "net_in_bytes": ("AWS/EC2", "NetworkIn", "Sum"),
    "net_out_bytes": ("AWS/EC2", "NetworkOut", "Sum"),
    "disk_read_bytes": ("AWS/EC2", "EBSReadBytes", "Sum"),
    "disk_write_bytes": ("AWS/EC2", "EBSWriteBytes", "Sum"),
    "mem_util_percent": ("CWAgent", "mem_used_percent", "Average"),
}
MAX_PONTOS = 1440  # limite de pontos por chamada do GetMetricStatistics


def cliente(regiao: str | None = None, perfil: str | None = None):
    import boto3
    return boto3.Session(profile_name=perfil, region_name=regiao).client("cloudwatch")


def coletar(instancia: str, inicio: datetime, fim: datetime, cloudwatch=None,
            metricas: list[str] | None = None) -> pd.DataFrame:
    cloudwatch = cloudwatch or cliente()
    inicio, fim = utc(inicio), utc(fim)
    passo = timedelta(seconds=PERIODO_S * MAX_PONTOS)
    partes = []
    for canonica in metricas or list(METRICAS):
        namespace, nome, estatistica = METRICAS[canonica]
        instantes, valores = [], []
        t = inicio
        while t < fim:
            resp = cloudwatch.get_metric_statistics(
                Namespace=namespace, MetricName=nome,
                Dimensions=[{"Name": "InstanceId", "Value": instancia}],
                StartTime=t, EndTime=min(t + passo, fim), Period=PERIODO_S, Statistics=[estatistica],
            )
            for p in resp["Datapoints"]:
                instantes.append(p["Timestamp"])
                valores.append(p[estatistica])
            t += passo
        if instantes:
            partes.append(para_pontos("aws", instancia, canonica, instantes, valores))
    return pd.concat(partes, ignore_index=True) if partes else vazio()
