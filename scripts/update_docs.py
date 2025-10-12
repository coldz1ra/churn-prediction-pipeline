import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parents[1]
reports = root / "reports"
figs = reports / "figures"
docs = root / "docs"

docs.mkdir(exist_ok=True, parents=True)

# 1) copy figures
(docs / "figures").mkdir(exist_ok=True, parents=True)
if figs.exists():
    for p in figs.glob("*.png"):
        shutil.copy2(p, docs / "figures" / p.name)

# 2) copy markdown report
rep_md = reports / "churn_report.md"
if rep_md.exists():
    shutil.copy2(rep_md, docs / "report.md")

# 3) metrics snippet
metrics_path = reports / "metrics.json"
snippet = docs / "metrics.html"
if metrics_path.exists():
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    def fmt(k: str, d: int = 4) -> str:
        v = m.get(k, "-")
        try:
            return f"{float(v):.{d}f}"
        except Exception:
            return str(v)

    html = f"""
<table class="min-w-full divide-y divide-gray-200 text-sm">
<thead><tr>
<th class="px-3 py-2 text-right">ROC-AUC</th>
<th class="px-3 py-2 text-right">PR-AUC</th>
<th class="px-3 py-2 text-right">LogLoss</th>
<th class="px-3 py-2 text-right">Brier</th>
<th class="px-3 py-2 text-right">Lift@10%</th>
<th class="px-3 py-2 text-right">Threshold@10%</th>
</tr></thead>
<tbody><tr>
<td class="px-3 py-2 text-right">{fmt('roc_auc')}</td>
<td class="px-3 py-2 text-right">{fmt('pr_auc')}</td>
<td class="px-3 py-2 text-right">{fmt('logloss')}</td>
<td class="px-3 py-2 text-right">{fmt('brier')}</td>
<td class="px-3 py-2 text-right">{fmt('lift@k',3)}</td>
<td class="px-3 py-2 text-right">{fmt('threshold@k',3)}</td>
</tr></tbody></table>
""".strip()
    snippet.write_text(html, encoding="utf-8")
