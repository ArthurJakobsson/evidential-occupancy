#!/usr/bin/env bash
# One-command public demo: launch the viser occupancy viewer + a cloudflared quick tunnel.
#
# Why cloudflared (not viser's own Share): this network blocks outbound TCP on high ports, so
# viser's share tunnel (and cloudflared's HTTP/2 fallback) can't connect. cloudflared's QUIC
# transport (UDP 7844) DOES get through here, so its quick tunnel works.
#
# Install cloudflared once (no root):
#   curl -fsSL -o ~/.local/bin/cloudflared \
#     https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared
#
# Usage:
#   bash scripts/demo_share.sh                 # 5 samples, port 8080 (auto-bumps if busy)
#   NUM_SAMPLES=10 PORT=8090 bash scripts/demo_share.sh
#   VIEWER_ARGS="--num_samples 5 --no_cameras" bash scripts/demo_share.sh
# Prints a https://<random>.trycloudflare.com URL to share. Ctrl+C stops both.
set -uo pipefail
cd "$(dirname "$0")/.."

PIXI="${PIXI:-pixi}"; command -v "$PIXI" >/dev/null 2>&1 || PIXI="$HOME/.pixi/bin/pixi"
CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
PORT="${PORT:-8080}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
VIEWER_ARGS="${VIEWER_ARGS:-}"
VLOG=/tmp/demo_viewer.log
CLOG=/tmp/demo_cf.log

command -v "$CF" >/dev/null 2>&1 || { echo "cloudflared not found at $CF"; exit 1; }
# pick a free port (avoid viser's internal auto-bump so the tunnel target matches)
if command -v ss >/dev/null 2>&1; then
    while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$((PORT + 1)); done
fi

VIEWER_PID=""; CF_PID=""
cleanup() { kill "$VIEWER_PID" "$CF_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[viewer] starting viser on :$PORT ..."
# python -u so prints flush; viser also prints its own "listening *:<port>" banner.
if [ -n "$VIEWER_ARGS" ]; then
    "$PIXI" run python -u scripts/vis_occupancy_viser.py --port "$PORT" $VIEWER_ARGS > "$VLOG" 2>&1 &
else
    "$PIXI" run python -u scripts/vis_occupancy_viser.py --num_samples "$NUM_SAMPLES" --port "$PORT" > "$VLOG" 2>&1 &
fi
VIEWER_PID=$!
# wait for viser's own banner (reliable) and read the ACTUAL bound port (viser auto-bumps if busy)
for _ in $(seq 1 150); do grep -qaE "listening \*:[0-9]+" "$VLOG" 2>/dev/null && break; sleep 1; done
ACTUAL_PORT=$(grep -aoE "listening \*:[0-9]+" "$VLOG" | grep -oE "[0-9]+$" | head -1)
[ -n "$ACTUAL_PORT" ] || { echo "[viewer] failed to start:"; tail -25 "$VLOG"; exit 1; }
echo "[viewer] up -> http://localhost:$ACTUAL_PORT"

echo "[tunnel] opening cloudflared quick tunnel -> :$ACTUAL_PORT ..."
"$CF" tunnel --url "http://localhost:$ACTUAL_PORT" --no-autoupdate > "$CLOG" 2>&1 &
CF_PID=$!
URL=""
for _ in $(seq 1 45); do
    URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$CLOG" | head -1)
    [ -n "$URL" ] && break; sleep 1
done
for _ in $(seq 1 25); do grep -qa "Registered tunnel connection" "$CLOG" && break; sleep 1; done

echo
echo "==================== DEMO IS LIVE ===================="
echo "  Public URL : ${URL:-<none — see $CLOG>}"
echo "  Local URL  : http://localhost:$ACTUAL_PORT"
if grep -qa "Registered tunnel connection" "$CLOG"; then echo "  Tunnel     : connected (QUIC)"; else echo "  Tunnel     : NOT confirmed — check $CLOG"; fi
echo "  Stop       : Ctrl+C"
echo "======================================================"
wait "$CF_PID"
