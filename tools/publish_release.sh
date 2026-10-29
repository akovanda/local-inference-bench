#!/usr/bin/env bash
# Explicit, repeatable publication of the reviewed export only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/publish_release.sh [--check | --publish]

--check    Validate the export, rebuild reports, and create local evidence assets.
           This is the default. It makes no GitHub or Git history changes.
--publish  Perform the same checks, then create/use the public
           akovanda/local-inference-bench repository, push main and the v0.1.0
           tag, and publish the release with its evidence archive and checksums.

Run from a clean, separately initialized export repository on branch main.
Use PYTHON=/path/to/python to select the environment containing pytest.
EOF
}

mode="${1:---check}"
case "$mode" in
  --help|-h) usage; exit 0 ;;
  --check|--publish) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$#" -gt 1 ]]; then usage >&2; exit 2; fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
python_bin="${PYTHON:-python3}"
repository='akovanda/local-inference-bench'
version='v0.1.0'
artifacts_dir="$repo_root/../artifacts"
archive_name="local-inference-bench-evidence-$version.tar.gz"

for command in git "$python_bin"; do command -v "$command" >/dev/null; done
actual_root="$(git rev-parse --show-toplevel)"
if [[ "$actual_root" != "$repo_root" ]]; then
  printf '%s\n' 'Refusing to use an enclosing workspace repository. Initialize and commit the export as its own repository first.' >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != 'main' ]]; then
  printf '%s\n' 'The reviewed export must be on branch main.' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' 'Commit the reviewed export before running release preflight.' >&2
  exit 1
fi

"$python_bin" tools/release_evidence.py verify --root .
"$python_bin" tools/release_evidence.py audit --root .
"$python_bin" -m pytest -q \
  tests/test_bench.py \
  tests/test_hard_cases.py \
  tests/test_specialist_cases.py \
  tests/test_public_benchmark_pilot.py \
  tests/test_public_model_expansion_report.py \
  tests/test_final_report.py \
  tests/test_public_evidence.py \
  tests/test_runtime_import_safety.py
rebuilt_dir="$(mktemp -d "${TMPDIR:-/tmp}/local-inference-bench-release.XXXXXX")"
trap 'rm -rf -- "$rebuilt_dir"' EXIT
"$python_bin" -m runtime.model_evidence_report_20261023 \
  --evidence-root . --output-dir "$rebuilt_dir/cross-study"
"$python_bin" -m runtime.public_model_expansion_report \
  --evidence-root . --output-dir "$rebuilt_dir/expansion"
"$python_bin" tools/release_evidence.py check-reports --root . --rebuilt "$rebuilt_dir"
"$python_bin" tools/release_evidence.py archive --root . --output "$artifacts_dir"
test -s "$artifacts_dir/$archive_name"
test -s "$artifacts_dir/SHA256SUMS"
(cd -- "$artifacts_dir" && sha256sum -c SHA256SUMS)
printf 'Local release checks passed. Assets: %s\n' "$artifacts_dir"

if [[ "$mode" == '--check' ]]; then
  printf '%s\n' 'No publication performed. tools/publish_release.sh --publish performs the documented GitHub writes.'
  exit 0
fi

command -v gh >/dev/null
gh auth status --hostname github.com
if [[ "$(gh api user --jq .login)" != 'akovanda' ]]; then
  printf '%s\n' 'Publication requires the authenticated akovanda GitHub account.' >&2
  exit 1
fi
if git show-ref --verify --quiet "refs/tags/$version"; then
  printf 'Local tag %s already exists; inspect the existing release manually.\n' "$version" >&2
  exit 1
fi
if git remote get-url origin >/dev/null 2>&1; then
  origin_url="$(git remote get-url origin)"
  case "$origin_url" in
    https://github.com/akovanda/local-inference-bench|https://github.com/akovanda/local-inference-bench.git|git@github.com:akovanda/local-inference-bench|git@github.com:akovanda/local-inference-bench.git) ;;
    *) printf 'Unexpected origin: %s\n' "$origin_url" >&2; exit 1 ;;
  esac
fi
if gh repo view "$repository" --json nameWithOwner,isPrivate >/dev/null 2>&1; then
  if [[ "$(gh repo view "$repository" --json isPrivate --jq .isPrivate)" != 'false' ]]; then
    printf '%s\n' 'The target repository exists but is not public; inspect it before publishing.' >&2
    exit 1
  fi
else
  gh repo create "$repository" --public \
    --description 'Auditable local inference experiments on NVIDIA T4 and Particle Tachyon: durable runners and curated evidence'
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "https://github.com/$repository.git"
fi
if git ls-remote --exit-code --tags origin "refs/tags/$version" >/dev/null 2>&1; then
  printf 'Remote tag %s already exists; inspect the existing release manually.\n' "$version" >&2
  exit 1
fi
git push -u origin main
git tag -a "$version" -m 'Local Inference Bench v0.1.0: source and curated evidence'
git push origin "refs/tags/$version"
gh release create "$version" \
  --repo "$repository" \
  --verify-tag \
  --title 'v0.1.0 — source and curated evidence' \
  --notes-file docs/release-notes-v0.1.0.md \
  "$artifacts_dir/$archive_name" \
  "$artifacts_dir/SHA256SUMS"
