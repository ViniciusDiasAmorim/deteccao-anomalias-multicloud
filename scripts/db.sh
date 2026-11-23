#!/usr/bin/env bash
# Instância PostgreSQL do projeto, usando o PostgreSQL instalado no sistema.
# Os dados ficam em .pgdata/ (fora do git); escuta só em localhost:5433, para
# não conflitar com o serviço do sistema. Não precisa de sudo.
#
# Uso: scripts/db.sh init|start|stop|status|psql
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DADOS="$RAIZ/.pgdata"
ENV="$RAIZ/.env"
PORTA=5433
USUARIO=tcc
BANCO=tcc_anomalias

ler_senha() {
    grep -E '^DATABASE_URL=' "$ENV" | sed -E 's#^DATABASE_URL=postgresql\+psycopg://[^:]+:([^@]+)@.*#\1#'
}

case "${1:-}" in
init)
    if [ -d "$DADOS" ]; then echo "já inicializado em $DADOS"; exit 0; fi
    umask 077
    if ! grep -qs '^DATABASE_URL=' "$ENV"; then
        SENHA="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
        echo "DATABASE_URL=postgresql+psycopg://$USUARIO:$SENHA@localhost:$PORTA/$BANCO" >> "$ENV"
    fi
    PWFILE="$(mktemp)"
    ler_senha > "$PWFILE"
    initdb -D "$DADOS" -U "$USUARIO" --pwfile="$PWFILE" --auth=scram-sha-256 -E UTF8 --locale=C.UTF-8 >/dev/null
    rm -f "$PWFILE"
    cat >> "$DADOS/postgresql.conf" <<EOF

# --- tcc-anomalias ---
port = $PORTA
listen_addresses = 'localhost'
unix_socket_directories = ''
EOF
    "$0" start
    PGPASSWORD="$(ler_senha)" createdb -h localhost -p "$PORTA" -U "$USUARIO" "$BANCO"
    echo "banco $BANCO criado em localhost:$PORTA"
    ;;
start)
    pg_ctl -D "$DADOS" -l "$DADOS/servidor.log" -w start
    ;;
stop)
    pg_ctl -D "$DADOS" -w stop
    ;;
status)
    pg_ctl -D "$DADOS" status
    ;;
psql)
    shift
    PGPASSWORD="$(ler_senha)" psql -h localhost -p "$PORTA" -U "$USUARIO" "$BANCO" "$@"
    ;;
*)
    echo "uso: $0 init|start|stop|status|psql" >&2
    exit 1
    ;;
esac
