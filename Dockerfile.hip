# syntax=docker/dockerfile:1
#
# Strata on AMD Radeon: the HIP backend.
#
# This is the AMD counterpart of the CUDA `Dockerfile`. Same shape - the engine is
# compiled during docker build and lives in the image, so the first container start
# only prepares the model and starts the server - but the backend, the base image and
# the device access differ. The GPU backend here is HIP (docs/AMD_HIP.md), not CUDA.
#
# Card: gfx1100 (RX 7900 XT / XTX) is validated by the maintainers; gfx1201 (RX 9070 /
# RX 9070 XT / Radeon AI PRO R9700) likewise. gfx1101 (RX 7800 XT) and gfx1200 (RX 9060 XT)
# are community-validated. cmake/hip_backend.cmake refuses anything else.
#
# Build (needs network: llama.cpp is fetched at its pinned commit):
#   docker build -f Dockerfile.hip -t strata:hip .
#   docker build -f Dockerfile.hip --build-arg HIP_ARCHITECTURES=gfx1100;gfx1201 -t strata:hip .
#
# Run: see the compose setup; the short form is
#   docker run --rm --device /dev/kfd --device /dev/dri --ulimit memlock=-1 \
#     -p 8080:8080 -v <Strata-data>:/data -e STRATA_API_KEY=<secret> strata:hip
#
# Why the base image is pinned to the host's ROCm: the tuned dense-prefill path uses
# tools/hip/<arch>-hipblaslt-<version>.txt, and setup applies that table only when BOTH the
# card architecture and the hipBLASLt version match. ROCm 7.2.4 carries hipBLASLt 1.2.2, so
# tools/hip/gfx1100-hipblaslt-100202.txt applies. Drift the base tag and the tuned path
# quietly stops being used - docs/AMD_HIP.md measured 656 -> 1,164 tok/s for the analogous
# gfx1201 case. The build below fails loudly instead (EXPECTED_HIPBLASLT).
#
# The engine's CPU kernels are compiled native to the machine that builds the image
# (AVX-512 on a Ryzen 7950X), so this image is not portable to other CPUs. The entry point
# warns when the host CPU lacks AVX2.
#
# There is no HIP image encoder yet (setup.py's build_engine_hip says so), so BUILD_VISION
# is "none" or "cpu" here; "cpu" builds the CPU encoder from tools/vision without CUDA.

ARG ROCM_VERSION=7.2.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1 and
# docker stop's SIGTERM reaches it (see setup.start).
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        ninja-build pkg-config \
        python3 python3-pip python3-venv unzip \
        hipblas-dev hipblaslt-dev \
    && rm -rf /var/lib/apt/lists/*
# rocm/dev-ubuntu-24.04 is the HIP/LLVM dev image: hipcc, hip-dev (hip-lang's CMake config) and
# rocm-llvm, but NO BLAS at all. setup.py's rocm_root() needs libhipblas.so in the system ROCm or
# it silently falls back to AMD's TheRock wheels in .venv - a different hipBLASLt and ~10 GB of
# downloads. hipblas-dev pulls hipblas 3.2.0 + rocblas 5.2.0, and hipblaslt-dev gives hipBLASLt
# 1.2.2 (1.2.2.70204), which is what the host's /opt/rocm 7.2.4 carries and what
# tools/hip/gfx1100-hipblaslt-100202.txt was calibrated for. Verified against the image's own
# repo; the guard below fails the build if a future base tag drifts it.

WORKDIR /opt/strata
COPY . .

# No GPU during docker build, so the AMD architecture is a build arg rather than read from
# the card - the same reasoning as CUDA_ARCHITECTURES in the CUDA Dockerfile.
ARG HIP_ARCHITECTURES=gfx1100
ARG BUILD_VISION=none            # none | cpu
# The hipBLASLt the shipped tuning table was calibrated for. The build stops if the base
# image carries a different one, rather than silently losing the tuned prefill path.
ARG EXPECTED_HIPBLASLT=1.2.2

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# Fail loudly, naming the check that failed, on a base image whose hipBLASLt has no shipped
# tuning table, or whose ROCm lacks the files setup.py needs (it would otherwise fall back to
# AMD's TheRock wheels in .venv: a different hipBLASLt and a ~10 GB build-time download).
# hipBLASLt's version encodes into the table name as MAJOR*100000 + MINOR*100 + PATCH, so
# 1.2.2 is gfx1100-hipblaslt-100202.txt (verified against the shipped list: 1.1.0 -> 100100,
# 1.2.0 -> 100200, 1.4.1 -> 100401, 1.5.0 -> 100500).
RUN set -e; \
    fail() { echo "ERROR: $1" >&2; exit 1; }; \
    test -x /opt/rocm/bin/hipcc || fail "/opt/rocm/bin/hipcc is missing from the base image"; \
    ls /opt/rocm/lib/libhipblas.so* >/dev/null || fail "/opt/rocm/lib/libhipblas.so* is missing: install hipblas-dev"; \
    h=/opt/rocm/include/hipblaslt/hipblaslt-version.h; \
    test -f "$h" || fail "$h is missing: install hipblaslt-dev"; \
    maj=$(awk '/define HIPBLASLT_VERSION_MAJOR/ {print $3}' "$h"); \
    min=$(awk '/define HIPBLASLT_VERSION_MINOR/ {print $3}' "$h"); \
    pat=$(awk '/define HIPBLASLT_VERSION_PATCH/ {print $3}' "$h"); \
    code=$(( maj*100000 + min*100 + pat )); \
    arch=$(echo "${HIP_ARCHITECTURES}" | cut -d';' -f1); \
    echo "hipBLASLt $maj.$min.$pat (code $code) -> table tools/hip/${arch}-hipblaslt-${code}.txt"; \
    test "$maj.$min.$pat" = "${EXPECTED_HIPBLASLT}" || \
      fail "hipBLASLt is $maj.$min.$pat but this build expects ${EXPECTED_HIPBLASLT} (the version the host's tuned config was measured with); pass --build-arg EXPECTED_HIPBLASLT=$maj.$min.$pat if the new one is intended"; \
    test -f "tools/hip/${arch}-hipblaslt-${code}.txt" || \
      fail "no shipped hipBLASLt tuning table for $arch at $maj.$min.$pat - the tuned dense-prefill path would silently stop applying; pick a ROCM_VERSION whose hipBLASLt has a table, or add one"

# setup.py picks ROCm itself: a system ROCm with hipcc, libhipblas.so and the HIP development
# files, else AMD's TheRock wheels installed into .venv. The wheels are a different hipBLASLt
# and ~10 GB of downloads, so a silent switch here would change the tuned path without saying
# so - check which one it picks, and stop the build if it is not the system ROCm.
RUN HIP_ARCHITECTURES="$HIP_ARCHITECTURES" .venv/bin/python - <<'PYEOF'
import os, setup

archs = [a for a in os.environ.get("HIP_ARCHITECTURES", "gfx1100").replace(",", ";").split(";") if a]
root, dirs = setup.rocm_root(archs)
print("ROCm Strata will build against:", root, dirs)
if str(root) != "/opt/rocm":
    raise SystemExit(
        f"ERROR: setup.py chose {root} (AMD's TheRock wheels) over the system ROCm. The base image "
        "is missing something rocm_root() requires: /opt/rocm/bin/hipcc, /opt/rocm/lib/libhipblas.so*, "
        "cmake/hip-lang/hip-lang-config.cmake or include/hip/hip_runtime.h.")
PYEOF

# Compile the engine exactly the way setup.py does on a Linux AMD PC: build_engine_hip owns
# the cmake argument list (STRATA_ENABLE_HIP, STRATA_PREFILL_MMQ, the rocm-path and
# rocm-device-lib-path HIP flags, the pinned llama.cpp as STRATA_GGML_DIR) and writes
# engine/BUILD.json with backend "hip", the architecture list and the source hash - which
# is what makes the first container start reuse the engine instead of recompiling it.
# The HIP build links the MMQ/K-quant kernels statically (libstrata_mmq.a,
# libstrata_prefill.a), so unlike the CUDA path there is no libstrata_kq.so to copy.
RUN HIP_ARCHITECTURES="$HIP_ARCHITECTURES" BUILD_VISION="$BUILD_VISION" .venv/bin/python - <<'PYEOF'
import os, setup

archs = [a for a in os.environ.get("HIP_ARCHITECTURES", "gfx1100").replace(",", ";").split(";") if a]
vision = os.environ.get("BUILD_VISION", "none")
assert vision in ("none", "cpu"), "BUILD_VISION on AMD is none or cpu (no HIP encoder yet)"

llama = setup.get_llama_cpp()
gpu = {"arch": archs[0], "archs": archs}
setup.build_engine_hip(gpu, llama, vision=vision)
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build-hip build-vision build && test -x engine/strata && cat engine/BUILD.json

# Run as the host user (compose passes user: ${UID}:${GID}) so the bind-mounted model tree
# is writable and the container is not root. /opt/strata is written to at start (the config
# symlink the entry point makes, and the engine log), so it is chowned to that user; the
# named volume at /opt/strata/logs inherits this ownership on first use.
ARG STRATA_UID=1003
ARG STRATA_GID=100
RUN groupadd -g "${STRATA_GID}" strata 2>/dev/null || true \
    && useradd -u "${STRATA_UID}" -g "${STRATA_GID}" -m -d /opt/strata -s /bin/sh strata 2>/dev/null || true \
    && mkdir -p /opt/strata/logs \
    && chown -R "${STRATA_UID}:${STRATA_GID}" /opt/strata

VOLUME ["/data", "/opt/strata/logs"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one. The port
# only opens after the model loads (the IQ3_S arena alone is ~47 GiB at ~5.7 GiB/s), so the
# start period is generous: a short one marks a still-loading container unhealthy and a
# restart policy would kill it mid-load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
