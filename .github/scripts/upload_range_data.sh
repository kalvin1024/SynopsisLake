#!/usr/bin/env bash
# Upload the range-query inputs of the ml_xy_pipeline job to the data volume via the Files API.
#   <DATA_VOLUME_DIR>/range_query/{0..3}.csv                     skipped when already there (HEAD 200)
#                                                                unless REUPLOAD_RANGE_DATA=true
#   <DATA_VOLUME_DIR>/range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv   always uploaded
#   <DATA_VOLUME_DIR>/range_query/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt   always uploaded
# Env: DATABRICKS_HOST, DATABRICKS_TOKEN, DATA_VOLUME_DIR, OUTPUT_VOLUME_DIR, RANGE_LOCAL_DIR, REUPLOAD_RANGE_DATA
set -euo pipefail

: "${DATABRICKS_HOST:?}" "${DATABRICKS_TOKEN:?}" "${DATA_VOLUME_DIR:?}" "${OUTPUT_VOLUME_DIR:?}"
RANGE_LOCAL_DIR="${RANGE_LOCAL_DIR:-range_query}"
REUPLOAD_RANGE_DATA="${REUPLOAD_RANGE_DATA:-false}"
REMOTE="${DATA_VOLUME_DIR}/range_query"
API="${DATABRICKS_HOST}/api/2.0/fs"
AUTH=(-H "Authorization: Bearer ${DATABRICKS_TOKEN}")

if [ ! -d "$RANGE_LOCAL_DIR" ]; then
  echo "::error::Local folder '$RANGE_LOCAL_DIR' not found"
  exit 1
fi

for DIR in "$REMOTE" "$REMOTE/dataDrivenQuery" "$REMOTE/dataDrivenQuery/results" "${OUTPUT_VOLUME_DIR}/ml"; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "${AUTH[@]}" "${API}/directories${DIR}")
  echo "Create directory $DIR: HTTP $CODE"
  case "$CODE" in
    200|201|204|409) ;;
    *) echo "::error::Creating $DIR failed with HTTP $CODE"; exit 1 ;;
  esac
done

for GZ in "$RANGE_LOCAL_DIR"/*.csv.gz; do
  [ -e "$GZ" ] || continue
  if [ ! -e "${GZ%.gz}" ]; then
    echo "gunzip $GZ"
    gunzip -k "$GZ"
  fi
done

# upload LOCAL DEST: streaming PUT (curl -T), so large files are not read into memory
upload() {
  local CODE
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "${AUTH[@]}" \
    -H "Content-Type: application/octet-stream" -T "$1" "${API}/files${2}?overwrite=true")
  echo "Upload $1 -> $2: HTTP $CODE"
  if [ "$CODE" != "200" ] && [ "$CODE" != "204" ]; then
    echo "::error::Upload of $1 failed with HTTP $CODE"
    exit 1
  fi
}

for i in 0 1 2 3; do
  F="$RANGE_LOCAL_DIR/$i.csv"
  if [ ! -e "$F" ]; then
    echo "::error::$F not found"
    exit 1
  fi
  DEST="$REMOTE/$i.csv"
  if [ "$REUPLOAD_RANGE_DATA" != "true" ]; then
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -I "${AUTH[@]}" "${API}/files${DEST}")
    if [ "$CODE" = "200" ]; then
      echo "Skip $F: $DEST already exists (set reupload_range_data to replace it)"
      continue
    fi
  fi
  upload "$F" "$DEST"
done

upload "$RANGE_LOCAL_DIR/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv" \
  "$REMOTE/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv"
upload "$RANGE_LOCAL_DIR/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt" \
  "$REMOTE/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt"
