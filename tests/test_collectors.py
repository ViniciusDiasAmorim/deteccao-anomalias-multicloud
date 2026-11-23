"""Testes dos coletores com respostas simuladas das APIs (sem acesso à nuvem).

AWS: botocore.stub.Stubber valida cada chamada contra o modelo oficial da API do
CloudWatch. OCI: um cliente falso devolve objetos reais do SDK (MetricData,
AggregatedDatapoint).
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import boto3
import pytest
from botocore.stub import ANY, Stubber
from oci.monitoring.models import AggregatedDatapoint, MetricData

from src.collectors import aws, oci
from src.collectors.base import CANONICAS, COLUNAS, para_pontos

INICIO = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _pontos_aws(n, estatistica, inicio=INICIO):
    # O CloudWatch não garante ordem: devolve-se fora de ordem de propósito.
    return [{"Timestamp": inicio + timedelta(minutes=5 * i), estatistica: float(i), "Unit": "None"}
            for i in reversed(range(n))]


def test_para_pontos_ordena_remove_duplicados_e_converte_utc():
    df = para_pontos("aws", "i-1", "cpu_util_percent",
                     ["2026-10-01T00:05:00Z", "2026-10-01T00:00:00Z", "2026-10-01T00:05:00Z"], [2.0, 1.0, 2.0])
    assert list(df.columns) == COLUNAS
    assert df["valor"].tolist() == [1.0, 2.0]
    assert str(df["instante"].dt.tz) == "UTC"
    assert (df["unidade"] == "%").all()


def test_para_pontos_rejeita_metrica_nao_canonica():
    with pytest.raises(ValueError):
        para_pontos("aws", "i-1", "CPUUtilization", [INICIO], [1.0])


def test_aws_pagina_em_blocos_de_1440_pontos_e_normaliza():
    cw = boto3.client("cloudwatch", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
    fim = INICIO + timedelta(days=6)  # 1728 períodos de 5 min → 2 chamadas por métrica
    with Stubber(cw) as stub:
        for canonica in ("cpu_util_percent", "net_in_bytes"):
            namespace, nome, estat = aws.METRICAS[canonica]
            esperado = {"Namespace": namespace, "MetricName": nome,
                        "Dimensions": [{"Name": "InstanceId", "Value": "i-abc"}],
                        "StartTime": ANY, "EndTime": ANY, "Period": 300, "Statistics": [estat]}
            stub.add_response("get_metric_statistics", {"Label": nome, "Datapoints": _pontos_aws(1440, estat)},
                              esperado)
            segundo = _pontos_aws(288, estat, INICIO + timedelta(days=5))
            stub.add_response("get_metric_statistics", {"Label": nome, "Datapoints": segundo}, esperado)
        df = aws.coletar("i-abc", INICIO, fim, cloudwatch=cw, metricas=["cpu_util_percent", "net_in_bytes"])
        stub.assert_no_pending_responses()

    assert set(df["metrica"]) == {"cpu_util_percent", "net_in_bytes"}
    for _, g in df.groupby("metrica"):
        assert len(g) == 1728
        assert g["instante"].is_monotonic_increasing
    assert (df["fonte"] == "aws").all() and (df["recurso"] == "i-abc").all()
    assert set(df.loc[df["metrica"] == "net_in_bytes", "unidade"]) == {CANONICAS["net_in_bytes"]}


def test_aws_metrica_sem_dados_nao_gera_linhas():
    cw = boto3.client("cloudwatch", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
    with Stubber(cw) as stub:
        stub.add_response("get_metric_statistics", {"Label": "mem_used_percent", "Datapoints": []})
        df = aws.coletar("i-abc", INICIO, INICIO + timedelta(hours=1), cloudwatch=cw, metricas=["mem_util_percent"])
    assert df.empty and list(df.columns) == COLUNAS


class MonitoringFalso:
    """Devolve, para cada consulta, `series` séries com `n_por_bloco` pontos a partir do início do bloco."""

    def __init__(self, n_por_bloco, series=1, valores=None):
        self.n, self.series, self.valores = n_por_bloco, series, valores
        self.chamadas = []

    def summarize_metrics_data(self, compartimento, detalhes):
        self.chamadas.append((compartimento, detalhes))
        dados = []
        for k in range(self.series):
            valores = self.valores or [float(i) for i in range(self.n)]
            pontos = [AggregatedDatapoint(timestamp=detalhes.start_time + timedelta(minutes=5 * i), value=v + k)
                      for i, v in enumerate(valores)]
            dados.append(MetricData(name="x", aggregated_datapoints=pontos))
        return SimpleNamespace(data=dados)


def test_oci_monta_consulta_mql_e_divide_em_blocos_de_7_dias():
    falso = MonitoringFalso(n_por_bloco=12)
    df = oci.coletar("ocid1.instance.oc1..x", "ocid1.compartment.oc1..y", INICIO, INICIO + timedelta(days=10),
                     monitoring=falso, metricas=["cpu_util_percent"])
    assert len(falso.chamadas) == 2
    compartimento, detalhes = falso.chamadas[0]
    assert compartimento == "ocid1.compartment.oc1..y"
    assert detalhes.namespace == "oci_computeagent" and detalhes.resolution == "5m"
    assert detalhes.query == 'CpuUtilization[5m]{resourceId = "ocid1.instance.oc1..x"}.mean()'
    assert len(df) == 24 and (df["metrica"] == "cpu_util_percent").all() and (df["fonte"] == "oci").all()
    assert df["instante"].is_monotonic_increasing


def test_oci_contadores_usam_increment_somam_series_e_descartam_reinicio():
    # duas VNICs; o 3º ponto simula o reinício do contador (variação negativa)
    falso = MonitoringFalso(n_por_bloco=4, series=2, valores=[100.0, 200.0, -50.0, 300.0])
    df = oci.coletar("i", "c", INICIO, INICIO + timedelta(hours=1), monitoring=falso, metricas=["net_in_bytes"])
    assert falso.chamadas[0][1].query == 'NetworksBytesIn[5m]{resourceId = "i"}.increment()'
    assert df["valor"].tolist() == [201.0, 401.0, 601.0]  # soma das duas séries; ponto negativo descartado
    assert (df["unidade"] == "bytes/5min").all()


def test_oci_percentual_vira_media_entre_series():
    falso = MonitoringFalso(n_por_bloco=2, series=2, valores=[10.0, 20.0])
    df = oci.coletar("i", "c", INICIO, INICIO + timedelta(hours=1), monitoring=falso, metricas=["mem_util_percent"])
    assert df["valor"].tolist() == [10.5, 20.5]


def test_esquema_comum_entre_provedores():
    assert set(aws.METRICAS) <= set(CANONICAS) and set(oci.METRICAS) <= set(CANONICAS)
