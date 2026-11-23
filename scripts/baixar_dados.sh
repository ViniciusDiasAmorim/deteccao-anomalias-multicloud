#!/usr/bin/env bash
# Baixa todos os dados usados no projeto para data/raw/ e confere a integridade.
#
# - Zenodo 14564935: séries agregadas do Alibaba 2018 e do Google 2019 (MD5 do registro).
# - Alibaba cluster-trace-v2018: machine_meta, batch_task e machine_usage (~1,9 GB,
#   usados na EDA e no estudo de caso; o machine_usage não é descompactado).
# - NAB: séries realAWSCloudwatch e realKnownCause e rótulos, fixados num commit.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$RAIZ/data/raw"
mkdir -p "$RAW/alibaba_v2018" "$RAW/nab/realAWSCloudwatch" "$RAW/nab/realKnownCause" "$RAW/nab/labels"

baixar() {  # url destino
    [ -s "$2" ] && { echo "já existe: ${2#"$RAIZ"/}"; return; }
    echo "baixando ${2#"$RAIZ"/}"
    curl -sSfL --retry 3 -o "$2.parcial" "$1" && mv "$2.parcial" "$2"
}

# Zenodo
ZENODO=https://zenodo.org/records/14564935/files
baixar "$ZENODO/machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1" \
       "$RAW/machine_usage_days_1_to_8_grouped_300_seconds.csv"
baixar "$ZENODO/instance_usage_grouped_300_seconds_month.csv?download=1" \
       "$RAW/instance_usage_grouped_300_seconds_month.csv"
(cd "$RAW" && md5sum -c - <<'EOF'
6c2312c01f1c4dc4b6863c4402976dee  machine_usage_days_1_to_8_grouped_300_seconds.csv
08d31b15f3203cd40407bdfb6ca902ea  instance_usage_grouped_300_seconds_month.csv
EOF
)

# Alibaba cluster-trace-v2018 (repositório oficial: github.com/alibaba/clusterdata)
ALI=http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces
for f in machine_meta batch_task machine_usage; do
    baixar "$ALI/$f.tar.gz" "$RAW/alibaba_v2018/$f.tar.gz"
done
for f in machine_meta batch_task; do
    [ -s "$RAW/alibaba_v2018/$f.csv" ] || tar -xzf "$RAW/alibaba_v2018/$f.tar.gz" -C "$RAW/alibaba_v2018" "$f.csv"
done

# NAB, commit fixo
NAB_COMMIT=ea702d75cc2258d9d7dd35ca8e5e2539d71f3140
NAB=https://raw.githubusercontent.com/numenta/NAB/$NAB_COMMIT
for f in ec2_cpu_utilization_24ae8d ec2_cpu_utilization_53ea38 ec2_cpu_utilization_5f5533 ec2_cpu_utilization_77c1ca \
         ec2_cpu_utilization_825cc2 ec2_cpu_utilization_ac20cd ec2_cpu_utilization_c6585a ec2_cpu_utilization_fe7f93 \
         ec2_disk_write_bytes_1ef3de ec2_disk_write_bytes_c0d644 ec2_network_in_257a54 ec2_network_in_5abac7 \
         elb_request_count_8c0756 grok_asg_anomaly iio_us-east-1_i-a2eb1cd9_NetworkIn rds_cpu_utilization_cc0c53 \
         rds_cpu_utilization_e47b3b; do
    baixar "$NAB/data/realAWSCloudwatch/$f.csv" "$RAW/nab/realAWSCloudwatch/$f.csv"
done
for f in cpu_utilization_asg_misconfiguration ec2_request_latency_system_failure; do
    baixar "$NAB/data/realKnownCause/$f.csv" "$RAW/nab/realKnownCause/$f.csv"
done
for f in combined_windows.json combined_labels.json; do
    baixar "$NAB/labels/$f" "$RAW/nab/labels/$f"
done
baixar "$NAB/LICENSE.txt" "$RAW/nab/LICENSE.txt"
[ -s "$RAW/nab/COMMIT.txt" ] || echo "$NAB_COMMIT" > "$RAW/nab/COMMIT.txt"

echo "dados prontos em data/raw/"
echo "próximo passo: tar -xzOf data/raw/alibaba_v2018/machine_usage.tar.gz | .venv/bin/python -I src/build_alibaba_aggregate.py"
