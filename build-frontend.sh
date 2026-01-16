#!/usr/bin/env bash
set -e

echo "▶ Build frontend (Docker Node 20)"

rm -rf dist
mkdir -p dist

docker run --rm \
  -v "$PWD/frontend:/app" \
  -w /app \
  node:20-alpine sh -lc "npm install && npm run build"

cp -a frontend/dist/. dist/

echo "✔ Frontend build terminé"
