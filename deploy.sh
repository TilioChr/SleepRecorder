#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Pull..."
cd "$ROOT_DIR"
git pull --rebase

echo "[2/4] Build frontend -> ./dist ..."
rm -rf "$ROOT_DIR/dist"
mkdir -p "$ROOT_DIR/dist"

docker run --rm \
  -v "$ROOT_DIR/frontend:/app" \
  -w /app \
  node:20-alpine sh -lc "npm ci && npm run build"

cp -r "$ROOT_DIR/frontend/dist/"* "$ROOT_DIR/dist/"

echo "[3/4] Start/Update services..."
docker compose up -d --build

echo "[4/4] Status:"
docker compose ps
echo "UI: http://<server-ip>/"
echo "API: http://<server-ip>/api/recordings"
echo "Files: http://<server-ip>/recordings/"
