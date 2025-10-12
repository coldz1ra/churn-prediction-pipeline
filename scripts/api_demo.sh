#!/usr/bin/env bash
set -euo pipefail
curl -s -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d @<(python - <<'PY'
import csv,json,sys
import pandas as pd
df = pd.read_csv("demo/scoring_demo.csv")
print(json.dumps({"data": df.to_dict(orient="records")}))
PY
)
echo
