# What this fork carries

Base: upstream tag `v0.2.15` (`c77572806`). Fork tags are `<upstream>+icedream.<n>`; the image tag is
`<upstream>-icedream.<n>-hip-gfx1100`. Bump `n` whenever a patch changes, never reuse a tag for
different bytes.

## Carried

| Patch | Upstream reference | Base | Why we carry it | Can drop when |
| --- | --- | --- | --- | --- |
| `5af25366` HIP elastic cache: per-device VMM probe, segmented expert cache, `--vram-elastic` on Linux HIP | PR Niko1221/Strata#2399, commit `5af25366`, still open and draft | `v0.2.15` | the RX 7900 XTX member runs `--vram-elastic`, and its vision encoder borrows 2 GiB through `POST /v1/vram`; upstream refuses the flag on HIP | #2399 merges and the flag works on HIP without it |

Local change on top of it: `vmm_segment_unmap(va, bytes)` and `vmm_segment_release(handle)` are two
functions, not one. Upstream's own `tests/core/expert_cache_segmented_test.cpp` and
`include/strata/core/expert_cache_refuse.hpp` drive a refused unmap and a refused release separately,
so the fused call could not satisfy them. `ExpertCache::shrink` keeps upstream's two-step flow and its
`ec_test::refuse_segment_call` hooks; only the driver calls behind them are ours.

## Deliberately not carried

| Commit | What it does | Why not |
| --- | --- | --- |
| `71ef01ec` share the growing K/V arena with `--vram-elastic` | extracts `kvg_take` / `kvg_give_back` so the K/V and the expert cache hand chunks to each other | `--kv-grow` is off whenever `--vram-elastic` or `--kv-resident` is set (`src/program/generate.cpp`, the `kvg.on` gate), and our member config sets both, so the code can never run. It also rewrites `kvg_ensure` / `kvg_trim`, which upstream changed after our base (`kvg.from_cache`, the `#1128` park hold, `kv_grow_target`), so it would need re-merging every release for nothing |

Revisit only if a member starts using `--kv-grow` instead of `--vram-elastic`.

## Our own, not upstream patches

- `Dockerfile.hip`, `Dockerfile.hip-tests`: build the HIP engine from source in the image. Upstream's
  ready-made `ghcr.io/niko1221/strata:<version>-rocm-gfx110x` stops at 0.2.13 and does not carry the
  `serve/server.py --config` start path below.
- `docker-entrypoint-serve.sh`: starts `/opt/strata/serve/server.py --config <file>` instead of
  upstream's `docker/entrypoint.sh`, which plans and tunes a model through `strata launch`.

## Build recipe in service

```
context:  git checkout of this tag (Dockerfile.hip tracked)
base:     bitsea:5102/docker.io/rocm/dev-ubuntu-24.04:7.14.1-full (pin by digest)
args:     HIP_ARCHITECTURES=gfx1100  EXPECTED_HIPBLASLT=1.4.1  BUILD_VISION=gpu
builder:  docker-container, cpus=6 memory=20g swap=30g
image:    bitsea:5100/icedream/strata:0.2.15-icedream.1-hip-gfx1100
```

`EXPECTED_HIPBLASLT=1.4.1` matches `tools/hip/gfx1100-hipblaslt-100401.txt`, which is the table the
member config points `STRATA_HIPBLASLT_TUNING` at. A base image whose hipBLASLt drifts from that fails
the build instead of silently losing the tuned prefill path.
