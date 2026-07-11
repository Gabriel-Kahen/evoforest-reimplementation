#!/usr/bin/env bash
set -euo pipefail

readonly FEAT_COMMIT="2967e6e5f7eee75ecf34062708e7b0b87c0b9145"
readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PREFIX="${FEAT_ENV_PREFIX:-${HERE}/.feat-env}"
readonly SOURCE="${FEAT_SOURCE_DIR:-${HERE}/.feat-src}"
readonly MAMBA="${MAMBA_EXE:-micromamba}"

command -v "${MAMBA}" >/dev/null || {
  echo "micromamba is required; set MAMBA_EXE to its absolute path." >&2
  exit 2
}

"${MAMBA}" create -y -p "${PREFIX}" -f "${HERE}/feat-linux-64.yml"

if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/lacava/feat.git "${SOURCE}"
fi
git -C "${SOURCE}" fetch --depth=1 origin "${FEAT_COMMIT}"
git -C "${SOURCE}" checkout --detach "${FEAT_COMMIT}"
rm -rf "${SOURCE}/build"

CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  "${MAMBA}" run -p "${PREFIX}" python -m pip install \
  --no-deps --no-build-isolation "${SOURCE}"

"${PREFIX}/bin/python" "${HERE}/feat_runner.py" --self-test
echo "FEAT environment ready: ${PREFIX}"
