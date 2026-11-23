"""Coleta métricas de uma instância AWS ou OCI e grava na tabela `metricas`.

Exemplos:
    .venv/bin/python -m src.collectors.coletar aws --instancia i-0123456789abcdef0 --regiao us-east-1 --horas 72
    .venv/bin/python -m src.collectors.coletar oci --instancia ocid1.instance.oc1... \\
        --compartimento ocid1.compartment.oc1... --horas 72 --detectar lof

Com --detectar, roda a detecção (src/detection.py) sobre cada métrica coletada,
usando todo o histórico da série no banco, e grava os alertas.
"""
import argparse
from datetime import datetime, timedelta, timezone

from src import db
from src.collectors import aws, oci
from src.detection import detectar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("provedor", choices=["aws", "oci"])
    ap.add_argument("--instancia", required=True, help="InstanceId (AWS) ou OCID da instância (OCI)")
    ap.add_argument("--horas", type=float, default=24)
    ap.add_argument("--regiao", help="AWS: região da instância")
    ap.add_argument("--perfil", help="AWS: perfil do ~/.aws/credentials; OCI: perfil do ~/.oci/config")
    ap.add_argument("--compartimento", help="OCI: OCID do compartimento")
    ap.add_argument("--config-oci", default="~/.oci/config")
    ap.add_argument("--detectar", choices=["isolation_forest", "lof", "random_forest"])
    args = ap.parse_args()

    fim = datetime.now(timezone.utc)
    inicio = fim - timedelta(hours=args.horas)
    if args.provedor == "aws":
        pontos = aws.coletar(args.instancia, inicio, fim, cloudwatch=aws.cliente(args.regiao, args.perfil))
    else:
        if not args.compartimento:
            ap.error("--compartimento é obrigatório para OCI")
        pontos = oci.coletar(args.instancia, args.compartimento, inicio, fim,
                             monitoring=oci.cliente(args.config_oci, args.perfil or "DEFAULT"))

    engine = db.conectar()
    db.criar_tabelas(engine)
    db.gravar_metricas(engine, pontos)
    print(f"{len(pontos)} pontos coletados de {args.instancia}")
    print(pontos.groupby("metrica").size().to_string() if len(pontos) else "nenhum ponto retornado")

    if args.detectar:
        for metrica in sorted(pontos["metrica"].unique()):
            serie = db.ler_serie(engine, args.provedor, args.instancia, metrica)
            try:
                resultado = detectar(serie, args.detectar)
            except ValueError as e:
                print(f"{metrica}: detecção não executada ({e})")
                continue
            n = db.gravar_alertas(engine, resultado, args.provedor, args.instancia, metrica, args.detectar)
            print(f"{metrica}: {n} alertas")


if __name__ == "__main__":
    main()
