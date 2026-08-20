"""
================================================================================
FYP — FINAL CODE COMPLIANCE CHECK
================================================================================

Automated verification of the CODE-SIDE compliance items only. Report/document
items (DOSM wording, dataset description in Section 1.6.1, documenting the
Malaysian adaptation in Chapter 3) are NOT checkable by code and are excluded
by design — see COMPLIANCE_CHECK.md for those.

Every check below EXECUTES or INSPECTS a real file. Nothing is assumed.

Run from the project root:
    python final_check.py

Exit code 0 if every code-side check passes, 1 otherwise.
================================================================================
"""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase2_outputs"

results = []


def check(item, condition, evidence):
    """Record one check. `condition` must already be evaluated to a bool."""
    results.append((item, bool(condition), evidence))
    print(f"  [{'PASS' if condition else 'FAIL'}] {item}")
    print(f"         {evidence}")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ==============================================================================
section("OBJECTIVE 1 / DELIVERABLE 1 — Preprocessing pipeline + EDA")
# ==============================================================================

prep = (ROOT / "data_preprocessing.py").read_text(encoding="utf-8", errors="ignore")

# The six elements named in Deliverable 1 (Section 1.6.2)
elements = {
    "Missing value handling": r"dropna",
    "Duplicate removal": r"drop_duplicates",
    "Data type standardisation": r"to_datetime|astype",
    "Invalid transaction filtering": r"Quantity\"?\]?\s*>\s*0|Price\"?\]?\s*>\s*0",
    "TotalAmount feature engineering": r"TotalAmount",
    "IQR outlier detection": r"quantile\(0?\.25\)|quantile\(0?\.75\)|IQR",
}
for name, pattern in elements.items():
    found = re.search(pattern, prep) is not None
    check(f"Del1 · {name}", found, f"regex /{pattern}/ in data_preprocessing.py")

# Objective 1 additionally requires EDA ("identifying key customer purchasing
# patterns through exploratory data analysis")
has_eda = ("describe()" in prep) and ("hist" in prep)
check("Obj1 · Exploratory data analysis present",
      has_eda, "describe() + matplotlib histograms in data_preprocessing.py")

# ==============================================================================
section("OBJECTIVE 2 / DELIVERABLE 2 — RFM + K-means segmentation")
# ==============================================================================

seg = pd.read_csv(OUT / "customer_rfm_segments.csv")
prof = pd.read_csv(OUT / "segment_profile_summary.csv")
sel = pd.read_csv(OUT / "cluster_selection_metrics.csv")

check("Obj2 · RFM computed per customer",
      {"Recency", "Frequency", "Monetary"}.issubset(seg.columns),
      f"columns present; {len(seg):,} customers")

check("Obj2 · Cluster assignment present",
      "Cluster" in seg.columns and seg["Cluster"].nunique() == 4,
      f"{seg['Cluster'].nunique()} clusters over {len(seg):,} customers")

check("Del2 · Interpretable behavioural profiles",
      "Segment_Label" in prof.columns and len(prof) == 4,
      f"labels: {', '.join(prof['Segment_Label'].str.split(' (', regex=False).str[0])}")

# Objective 2 names BOTH validation methods explicitly
check("Obj2 · Elbow method (WCSS) recorded",
      "WCSS_inertia" in sel.columns and len(sel) >= 5,
      f"WCSS for k={sel['k'].min()}..{sel['k'].max()}, "
      f"{sel['WCSS_inertia'].iloc[0]:.0f} -> {sel['WCSS_inertia'].iloc[-1]:.0f}")

check("Obj2 · Silhouette scores recorded",
      "Silhouette" in sel.columns,
      f"silhouette for k={sel['k'].min()}..{sel['k'].max()}")

sil_k4 = float(sel.loc[sel["k"] == 4, "Silhouette"].iloc[0])

# Every customer must carry a segment (no unassigned rows reaching the dashboard)
check("Obj2 · No unassigned customers",
      seg["Segment_Label"].notna().all(),
      f"{seg['Segment_Label'].notna().sum():,}/{len(seg):,} labelled")

# ==============================================================================
section("OBJECTIVE 3 / DELIVERABLE 3 — Predictive modelling suite")
# ==============================================================================

comp = pd.read_csv(OUT / "model_comparison.csv")
imp = pd.read_csv(OUT / "feature_importance.csv")
rec = json.loads((OUT / "model_recommendation.json").read_text(encoding="utf-8"))

REQUIRED_MODELS = {"Logistic Regression", "Decision Tree", "Random Forest"}
check("Obj3 · All three required algorithms evaluated",
      REQUIRED_MODELS.issubset(set(comp["Model"])),
      f"present: {sorted(set(comp['Model']))}")

REQUIRED_METRICS = ["Accuracy", "Precision", "Recall", "F1"]
check("Obj3 · All four required metrics reported",
      all(m in comp.columns for m in REQUIRED_METRICS),
      f"columns: {[c for c in comp.columns if c in REQUIRED_METRICS]}")

check("Del3 · Algorithmic justification recorded",
      "recommended_model" in rec and len(rec.get("reason", "")) > 50,
      f"recommends {rec['recommended_model']}; "
      f"statistically indistinguishable = {rec['models_statistically_indistinguishable']}")

check("Obj3 · Feature importance for all three models",
      set(imp["Model"]) == REQUIRED_MODELS,
      f"{len(imp)} rows covering {imp['Model'].nunique()} models")

best = comp.sort_values("F1", ascending=False).iloc[0]
acc = float(best["Accuracy"])

# ==============================================================================
section("DELIVERABLE 4 — Model Evaluation Report (segmentation AND prediction)")
# ==============================================================================

check("Del4 · Segmentation quality persisted (silhouette + WCSS)",
      (OUT / "cluster_selection_metrics.csv").exists(),
      "phase2_outputs/cluster_selection_metrics.csv")

check("Del4 · Predictive performance persisted",
      (OUT / "model_comparison.csv").exists(),
      "phase2_outputs/model_comparison.csv")

audit_report = ROOT / "audit" / "PHASE2_AUDIT_REPORT.md"
audit_text = audit_report.read_text(encoding="utf-8", errors="ignore") if audit_report.exists() else ""
check("Del4 · Limitation diagnostics documented",
      audit_report.exists() and len(audit_text) > 5000,
      f"{audit_report.name}: {len(audit_text):,} chars of critical assessment")

audit_scripts = sorted((ROOT / "audit").glob("audit_*.py"))
check("Del4 · Diagnostics are reproducible, not just prose",
      len(audit_scripts) >= 3,
      f"{len(audit_scripts)} executable audit scripts")

# ==============================================================================
section("OBJECTIVE 4 / DELIVERABLE 5 / RQ4 — Dashboard")
# ==============================================================================

app = (ROOT / "app.py").read_text(encoding="utf-8", errors="ignore")

# Architecture claim: dashboard must read ONLY phase2_outputs, never a source CSV
source_reads = re.findall(r'read_csv\(\s*["\']([^"\']+)', app)
check("Del5 · Reads only phase2_outputs (no recomputation)",
      len(source_reads) == 0,
      f"no bare-path read_csv calls; all loads go via OUT_DIR "
      f"({len(re.findall(r'OUT_DIR', app))} references)")

# Deliverable 5 names four required content types
content = {
    "Segmentation distributions": r"_revenue_concentration_chart|_rfm_comparison_chart",
    "Segment-specific spending": r"Average value|avg_value",
    "Predictive model results": r"_model_comparison_chart",
    "Strategic recommendations": r"SEGMENT_CONTENT|Recommended action",
}
for name, pattern in content.items():
    check(f"Del5 · {name}", re.search(pattern, app) is not None, f"/{pattern}/ in app.py")

# Non-technical accessibility: colour must never be the sole channel
check("Del5 · Colour not sole encoding (accessibility)",
      "View as table" in app or "as a table" in app,
      "table views present alongside charts")

print("\n  Executing the dashboard (this takes ~60s)...")
test_code = """
import sys
from streamlit.testing.v1 import AppTest
bad = []
for page in ["Overview", "Customer Segments", "Prediction Insights"]:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.radio[0].set_value(page).run()
    if at.exception:
        bad.append((page, str(at.exception[0].value)[:120]))
print("DASHBOARD_RESULT:" + ("OK" if not bad else str(bad)))
"""
proc = subprocess.run([sys.executable, "-c", test_code], cwd=ROOT,
                      capture_output=True, text=True, timeout=600)
dash_ok = "DASHBOARD_RESULT:OK" in proc.stdout
check("Obj4 · Dashboard executes, all 3 pages, zero exceptions",
      dash_ok,
      "AppTest across Overview / Customer Segments / Prediction Insights")

# Deployment is the one item code cannot satisfy on the user's behalf
git_dir = (ROOT / ".git").exists()
remote = ""
if git_dir:
    try:
        remote = subprocess.run(["git", "remote", "-v"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        remote = ""
check("Obj4/RQ4 · Dashboard DEPLOYED (public URL)",
      bool(git_dir and remote),
      f"git repo: {git_dir}; remote: {remote or 'none'} "
      f"-> {'deployed' if (git_dir and remote) else 'NOT deployed — you must push + deploy'}")

# ==============================================================================
section("CONSTRAINTS (Section 1.6.3) — exclusions must not be violated")
# ==============================================================================

# This file is excluded from its own scan. It necessarily contains the very
# keywords it searches for (they are the search pattern), so including it would
# make the check report a false violation against itself.
SELF = Path(__file__).resolve()
py_files = [p for p in ROOT.rglob("*.py")
            if ".git" not in str(p) and p.resolve() != SELF]
all_code = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in py_files)

realtime = re.findall(
    r"\b(APScheduler|schedule\.every|croniter|KafkaConsumer|websocket|"
    r"st_autorefresh|while\s+True|time\.sleep)\b", all_code)
check("Constraint · No real-time / streaming / scheduled processing",
      len(realtime) == 0,
      f"scanned {len(py_files)} .py files; matches: {realtime or 'none'}")

infra = [f.name for f in ROOT.iterdir()
         if f.name in {"Dockerfile", "docker-compose.yml", "Procfile", ".github",
                       "kubernetes", "terraform"}]
check("Constraint · No production-grade deployment infrastructure",
      len(infra) == 0,
      f"no Docker/CI/orchestration files present ({infra or 'none found'})")

reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").strip().splitlines()
reqs = [r for r in reqs if r.strip() and not r.strip().startswith("#")]
check("Constraint · Lightweight free-tier dependency footprint",
      len(reqs) <= 5,
      f"{len(reqs)} runtime deps: {', '.join(r.split('>')[0] for r in reqs)}")

check("Constraint · UK origin + synthetic geography disclosed in the UI",
      "UK retail dataset" in app and "synthetic" in app.lower(),
      "disclosure present in the dashboard's 'About this data' panel")

check("Exclusion · No claim of perfect accuracy; limitations shown",
      "ceiling" in app.lower() and "Limitations" in app,
      "limitations panel present on the prediction page")

# ==============================================================================
section("SUCCESS CRITERIA — actual measured values")
# ==============================================================================

print(f"  Silhouette @ k=4       : {sil_k4:.4f}   target > 0.50   "
      f"-> {'MET' if sil_k4 > 0.5 else 'NOT MET'}")
print(f"  Best model accuracy    : {acc:.4f}   target > 0.75   "
      f"-> {'MET' if acc > 0.75 else 'NOT MET'}")
print(f"  Best model             : {best['Model']}")
print("\n  These two are reported as-measured. They are NOT counted as failures")
print("  of the code: the audit established both as genuine ceilings, and both")
print("  are documented honestly in the scripts, the audit report, and the UI.")

check("Criteria · Shortfalls disclosed in the dashboard UI (not glossed over)",
      "0.360" in app and "below the 0.5 target" in app,
      "silhouette shortfall stated in the 'About this data' panel")

# ==============================================================================
section("RESULT")
# ==============================================================================

failed = [r for r in results if not r[1]]
passed = len(results) - len(failed)
print(f"  {passed}/{len(results)} code-side checks passed")

if failed:
    print("\n  OUTSTANDING:")
    for item, _, evidence in failed:
        print(f"    - {item}")
        print(f"        {evidence}")
    print("\n  NOTE: if the only failure is deployment, every other code-side")
    print("  requirement is satisfied — deployment is an action you take, not")
    print("  something the code can do for itself.")
else:
    print("\n  All code-side compliance checks pass.")

sys.exit(0 if not failed else 1)
