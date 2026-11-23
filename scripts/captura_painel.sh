#!/usr/bin/env bash
# Captura a tela do painel (results/figures/fig_sistema_painel.png).
# Sobe a API numa porta temporária, abre o painel no Chrome headless com um perfil
# descartável e derruba a API ao final. Requer o banco ligado (scripts/db.sh start).
#
# Uso: scripts/captura_painel.sh [recurso] [modelo]
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
RECURSO="${1:-ec2_request_latency_system_failure}"
MODELO="${2:-isolation_forest}"
PORTA=8765
SAIDA="$RAIZ/results/figures/fig_sistema_painel.png"
PERFIL="$(mktemp -d)"

cd "$RAIZ"
.venv/bin/uvicorn src.api.main:app --port "$PORTA" --log-level warning &
API=$!
trap 'kill $API 2>/dev/null; rm -rf "$PERFIL"' EXIT

for _ in $(seq 60); do
    curl -sf "http://localhost:$PORTA/saude" >/dev/null && break
    sleep 0.5
done

google-chrome-stable --headless=new --disable-gpu --hide-scrollbars --user-data-dir="$PERFIL" \
    --window-size=1280,1500 --force-device-scale-factor=2 --virtual-time-budget=20000 --screenshot="$SAIDA" \
    "http://localhost:$PORTA/?recurso=$RECURSO&modelo=$MODELO" 2>/dev/null
echo "$SAIDA"
