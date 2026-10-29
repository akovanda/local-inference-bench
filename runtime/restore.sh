#!/usr/bin/env bash
set -euo pipefail
# Stops only this experiment's explicitly named containers, then restarts the
# original container. It never removes containers, volumes, models, or sessions.
for trial in moe-test-ollama-old-f16 moe-test-ollama-old-q8 moe-test-ollama-new-f16 moe-test-ollama-new-q8 moe-test-strata; do
  if docker container inspect "$trial" >/dev/null 2>&1; then
    docker stop --time 20 "$trial"
  fi
done
docker start local-codex-ollama-cpu
