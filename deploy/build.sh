#!/usr/bin/env bash
# Build the MkDocs site nested under /ml/ for Cloudflare Pages upload.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist-deploy
.venv/bin/mkdocs build --strict -d dist-deploy/ml
printf '/ml/* /ml/404.html 404\n' > dist-deploy/_redirects
echo "Built: $(find dist-deploy -type f | wc -l | tr -d ' ') files in dist-deploy/ml"
