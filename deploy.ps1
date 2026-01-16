$ErrorActionPreference = "Stop"

Write-Host "[1/4] Build frontend (Vite)..." -ForegroundColor Cyan
if (Test-Path .\frontend\node_modules) { Remove-Item -Recurse -Force .\frontend\node_modules }

podman run --rm `
  -v "${PWD}\frontend:/app" `
  -w /app `
  docker.io/library/node:20-alpine sh -lc "npm ci && npm run build"

Write-Host "[2/4] Publish frontend to ./dist (served by nginx)..." -ForegroundColor Cyan
Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\dist | Out-Null
Copy-Item -Recurse -Force .\frontend\dist\* .\dist\

Write-Host "[3/4] Restart nginx..." -ForegroundColor Cyan
podman restart sleeprec_nginx | Out-Null

Write-Host "[4/4] Done. Open http://localhost:8080/" -ForegroundColor Green
