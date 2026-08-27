#!/bin/bash
# Check local vLLM/Proxy service health, print HTTP code (200=healthy).
# Usage: bash check_health.sh <port> [health_path] [bind_ip]
#   health_path defaults to /health (vLLM); proxy uses /healthcheck
#   bind_ip defaults to 127.0.0.1; 0.0.0.0 or empty → 127.0.0.1
port="${1:?missing port}"
path="${2:-/health}"
ip="${3:-127.0.0.1}"
[ "$ip" = "0.0.0.0" ] && ip="127.0.0.1"
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "http://${ip}:${port}${path}" 2>/dev/null)
echo "${code:-000}"
