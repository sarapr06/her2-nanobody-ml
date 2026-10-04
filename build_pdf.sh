#!/bin/bash
# Build REPORT.pdf and mini_overview.pdf from the markdown sources.
#   ./build_pdf.sh
# Needs: brew install pandoc pango cairo gdk-pixbuf; uv pip install weasyprint
set -euo pipefail
cd "$(dirname "$0")"
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

for f in REPORT mini_overview; do
  # Fragment only, not --standalone: pandoc's standalone template embeds its own
  # stylesheet (body{margin:auto;max-width:36em}) which confines the text to a
  # narrow column and wins over an external sheet. We supply the whole page instead.
  {
    printf '<!DOCTYPE html><html><head><meta charset="utf-8">'
    printf '<style>%s</style></head><body>' "$(cat docs/print.css)"
    pandoc "docs/$f.md" -f gfm -t html5
    printf '</body></html>'
  } > "/tmp/$f.html"
done

uv run --no-project python - <<'PY'
from weasyprint import HTML
import os
for f, limit in (("REPORT", 6), ("mini_overview", 2)):
    doc = HTML(filename=f"/tmp/{f}.html").render()
    doc.write_pdf(f"docs/{f}.pdf")
    n = len(doc.pages)
    status = "OK" if n <= limit else f"OVER by {n - limit}"
    print(f"docs/{f}.pdf: {n} pages (limit {limit}) {status}  "
          f"{os.path.getsize(f'docs/{f}.pdf') / 1024:.0f} KB")
PY
