#!/usr/bin/env bash
set -euo pipefail

YML="${1:-environment.yml}"

if [[ ! -f "$YML" ]]; then
  echo "YAML not found: $YML" >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  PM=mamba
else
  PM=conda
fi

echo ">> Using $PM with $YML"

# Create; if it already exists, update with --prune
if ! $PM env create -f "$YML"; then
  echo ">> Env likely exists; updating instead..."
  $PM env update -f "$YML" --prune
fi

