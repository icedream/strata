# What this fork carries

Base: upstream tag `v0.2.22` (`b28ca95e`). Fork tags are `<upstream>+icedream.<n>`, the image tag is
`<upstream>-icedream.<n>-hip-gfx1100`. Bump `n` when a patch changes; a tag is never reused for
different bytes.

## Carried

| Patch | Upstream reference | Why we carry it | Can drop when |
| --- | --- | --- | --- |
| HIP virtual memory: a device probe, `hipMem*` behind the CUDA VMM API, `--vram-elastic` on Linux | PR Niko1221/Strata#2399 (`63476e48`, re-ported to current main), open and draft | the gfx1100 member runs `--vram-elastic` and its vision encoder borrows VRAM through `POST /v1/vram`; upstream compiles the VMM layer for CUDA only and refuses the flag on HIP | #2399 merges and the flag works on HIP without it |

Files: `engine/src/core/vmm.cpp` and `engine/include/strata/core/vmm.hpp` (the HIP side of `api()`,
`VmmRange`, `vmm_off_reason()` and the segment API), `engine/src/program/generate.cpp` (the
`--vram-elastic` gate: HIP on Linux is allowed when the probe passes, and names the probe's reason
when it does not), `engine/tests/core/vmm_test.cpp`.

None of those files changed upstream between v0.2.20 and v0.2.22, so the 0.2.22 rebase took them
unchanged; only the gate hunk in `generate.cpp` was re-applied. Upstream 0.2.22 still defines
`STRATA_EC_NO_VMM` for every `STRATA_USE_HIP` build and still answers `--vram-elastic` on HIP with
`"it is NVIDIA-only for now"`.

Upstream 0.2.20 has the segmented expert cache itself, so `engine/src/core/expert_cache.cpp` is
unpatched here and this carry is the driver layer only. The segment API it exposes
(`vmm_reserve`, `vmm_segment_map`, `vmm_segment_unmap`, `vmm_segment_release`) has no caller on
0.2.20 or 0.2.22, where the cache uses `VmmRange`; it is kept because the upstream patch ships it.

## No longer carried

The 0.2.15 line carried the segmented expert cache itself, its `ec_test::refuse_segment_call`
hooks and a split `vmm_segment_unmap`/`vmm_segment_release`. Upstream 0.2.20 ships the cache and the
hooks (`engine/include/strata/core/expert_cache_refuse.hpp`), so taking its `expert_cache.cpp`
whole replaced all of it.

## Deliberately not carried

| Commit | What it does | Why not |
| --- | --- | --- |
| `71ef01ec` share the growing K/V arena with `--vram-elastic` | lets the K/V and the expert cache hand chunks to each other | `--kv-grow` is off whenever `--kv-resident`, `--vram-elastic` or `--batch` is set (`engine/src/program/generate.cpp`, the `kvg.on` gate), and the member config sets all three |

## Our own, not upstream patches

- `Dockerfile.hip`, `Dockerfile.hip-tests`: compile the HIP engine and the GPU picture reader inside
  the image. Upstream's ready-made `-rocm-gfx110x` images start through `strata launch`, which plans
  and tunes a model at every start, and they are not built for this card's ROCm.
- `BUILD_INKLING=1` additionally builds `strata-inkling` (Inkling-2-Mini, `.strata v1.1`, safetensors
  or GGUF) from the same configured build tree. Upstream's engine/CMakeLists.txt already includes
  the target when `STRATA_ENABLE_HIP` is on; the app normally fetches a prebuilt shape engine instead.
- `docker-entrypoint-serve.sh`: starts `terminal/serve/server.py --config <file>` from a finished
  config instead. The cluster's start path passes the same arguments to that file directly.

## Where the 0.2.20 layout differs from 0.2.15

- `setup.py`, `serve/` and the shipped `data/` moved under `terminal/`. The image sets
  `PYTHONPATH=/opt/strata/terminal` for the `serve` import. The picture reader moved with them, to
  `terminal/tools/vision`; `rocm/dev-ubuntu-24.04:7.14.1-full` ships hipBLASLt 1.4.1, whose table is
  `terminal/tools/hip/gfx1100-hipblaslt-100401.txt`.
- The member configs name `--expert-profile /opt/strata/data/expert-profile.bin`. `Dockerfile.hip`
  keeps `/opt/strata/data` as a symlink to `terminal/data`, so those configs read as they are.

## Build recipe

`kubed/images` builds the `strata-hip` service from a git URL at a pinned commit, so the cluster
manifest, the `.env` ref and the fork tag have to agree:

```
context:    https://github.com/icedream/strata.git#<commit of the fork tag>
dockerfile: Dockerfile.hip
args:       HIP_ARCHITECTURES=gfx1100  EXPECTED_HIPBLASLT=1.4.1  BUILD_VISION=gpu  BUILD_INKLING=1
```

`BUILD_VISION=gpu` is required, not a preference: the member config starts
`engine/strata-vision-hip` for its `--vision` block, and the build fails without the reader.

`EXPECTED_HIPBLASLT` must match a table in `terminal/tools/hip/` (`<arch>-hipblaslt-<code>.txt`,
where the version encodes as MAJOR*100000 + MINOR*100 + PATCH). That table is the tuned
dense-prefill path, and setup applies it only when the card architecture and the hipBLASLt version
both match; the build fails rather than lose it silently.

The engine's CPU kernels are compiled for the building machine's CPU, so the build runs on the host
that serves the model.
