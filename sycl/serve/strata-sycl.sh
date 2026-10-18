#!/usr/bin/env bash
# strata-sycl.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata). The port's binary
# needs the oneAPI runtime of the dev image, so this runs it there with stdin/stdout attached (the serve protocol is
# lines on those pipes) and stderr going to the server's log. Paths in the config's args are the container's:
# the data root is mounted at /work.
#   STRATA_SYCL_ROOT   host directory mounted at /work        (default: two levels above this script's repo)
#   STRATA_SYCL_IMAGE  the runtime image                      (default: strata-sycl-dev)
#   STRATA_SYCL_BIN    the engine binary, relative to the repo (default: build-sycl-aot/strata)
#   STRATA_SYCL_NAME   the container's name                   (default: strata-sycl-serve)
#   STRATA_SYCL_NATIVE 1 runs the binary on the host instead of in the image; this also happens by itself when the
#                      machine has no docker, provided oneAPI is installed (/opt/intel/oneapi/setvars.sh is sourced)
# Passed into the container (or into the native process): every variable of this script's environment that starts
# with STRATA_ (the config's "env" block arrives that way: the server puts it in the engine's environment), ONEAPI_
# (level_zero:* for a two-card split, #423), UR_, IGC_, SYCL_ or ZES_. The port's own defaults (below) are
# overridden the same way, e.g. "env": {"STRATA_VERIFY_NO_HOST": "0"} in the config.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
root=${STRATA_SYCL_ROOT:-$(dirname "$here")}
repo_in=/work/$(basename "$here")
name=${STRATA_SYCL_NAME:-strata-sycl-serve}

# The port's run-time switches, set the same way in both modes (the config's "env" block arrives through the
# environment): the device-built verify plan, and no host handshakes in it.  NO_HOST is for a card on the xe driver
# (the host<->GPU flag stores are not visible there); an i915 card (the A-series, an integrated GPU) runs without it.
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=12)
while IFS= read -r k; do
    case "$k" in
        STRATA_SYCL_*) ;;                                 # this script's own settings
        STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|NEOReadDebugKeys|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
    esac
done < <(compgen -e)

# NATIVE MODE (STRATA_SYCL_NATIVE=1, or no docker on the machine): run the binary on the host instead of in the dev
# image.  Docker is the only way to get the port's oneAPI runtime without installing it, but a machine that already
# has oneAPI (the documented "How to run it by hand", docs/INTEL.md) should not need a container as well - and
# integrated parts are usually desktops where installing Docker is the larger ask.  The only difference is paths: the
# config's args name the container's /work, so each is rewritten to the host's root and the binary is started from the
# repo.  OneAPI's setvars.sh is sourced when it is there and the environment is not already set up.
if [ "${STRATA_SYCL_NATIVE:-0}" = "1" ] || ! command -v docker >/dev/null 2>&1; then
    if [ -z "${ONEAPI_ROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
    fi
    out=()
    for a in "$@"; do
        case "$a" in
            /work/*) out+=("${a/\/work/$root}") ;;
            *) out+=("$a") ;;
        esac
    done
    for k in "${!setting[@]}"; do
        case "${setting[$k]}" in 0|"") ;; *) export "$k=${setting[$k]}" ;; esac
    done
    cd "$here"
    exec "${STRATA_SYCL_BIN:-build-sycl-aot/strata}" "${out[@]}"
fi

docker rm -f "$name" >/dev/null 2>&1 || true              # a container left behind by a killed server
args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
envs=()
for k in "${!setting[@]}"; do
    # the engine tests the switches by presence: a value of 0 (or empty) means "not set", so it is not passed on
    case "${setting[$k]}" in 0|"") ;; *) envs+=(-e "$k=${setting[$k]}") ;; esac
done
exec docker run --rm -i --name "$name" --device /dev/dri --oom-score-adj 1000 --stop-timeout 30 --no-healthcheck \
    --ulimit memlock=-1 \
    -v "$root:/work" \
    "${envs[@]}" \
    "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" \
    "cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
