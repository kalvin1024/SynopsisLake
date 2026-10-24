#!/usr/bin/env bash
# Download the ml_xy_pipeline report from <OUTPUT_VOLUME_DIR>/ml/<ML_DATASET>/report/ into REPORT_DIR
# and append report.md to the GitHub Actions run summary.
# Env: DATABRICKS_HOST, DATABRICKS_TOKEN, OUTPUT_VOLUME_DIR, ML_DATASET, REPORT_DIR (default ml-report),
#      GITHUB_STEP_SUMMARY
set -euo pipefail

: "${DATABRICKS_HOST:?}" "${DATABRICKS_TOKEN:?}" "${OUTPUT_VOLUME_DIR:?}" "${ML_DATASET:?}"
REPORT_DIR="${REPORT_DIR:-ml-report}"
REMOTE="${OUTPUT_VOLUME_DIR}/ml/${ML_DATASET}/report"
mkdir -p "$REPORT_DIR"

for F in report.md v0_results.csv feature_importance.csv; do
  CODE=$(curl -s -o "$REPORT_DIR/$F" -w "%{http_code}" \
    -H "Authorization: Bearer ${DATABRICKS_TOKEN}" "${DATABRICKS_HOST}/api/2.0/fs/files${REMOTE}/$F")
  echo "Download $REMOTE/$F: HTTP $CODE"
  if [ "$CODE" != "200" ]; then
    if [ "$F" = "report.md" ]; then
      echo "::error::report.md not found at $REMOTE (HTTP $CODE): $(head -c 500 "$REPORT_DIR/$F" 2>/dev/null)"
      exit 1
    fi
    echo "::warning::$F not found at $REMOTE (HTTP $CODE)"
    rm -f "$REPORT_DIR/$F"
  fi
done

MD="$REPORT_DIR/report.md"
if [ ! -s "$MD" ]; then
  echo "::error::report.md is empty"
  exit 1
fi
# An API error comes back as JSON ({"error_code": ...}), not Markdown
if head -c 1 "$MD" | grep -q '{' && grep -q '"error_code"' "$MD"; then
  echo "::error::report.md is an API error: $(head -c 500 "$MD")"
  exit 1
fi

{
  echo "<!-- ml_xy_pipeline report: ${REMOTE}/report.md -->"
  cat "$MD"
  echo
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
echo "Report appended to the run summary"
