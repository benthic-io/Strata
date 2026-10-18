# Local testing: Intel UHD 770 (Alder Lake-S GT1) + Cyber-Tiel Q4

This page records a local attempt to build and run Strata on this machine. It is
kept for the maintainer's own hardware notes and for anyone with similar (or
better) Intel silicon. It is not a claim that Strata supports this GPU.

## Hardware

- CPU: Intel Core i5-13500 (20 threads)
- GPU: Intel UHD Graphics 770 (Alder Lake-S GT1, PCI 8086:4680, i915 driver)
- RAM: 62 GiB
- OS: Fedora Linux 44 (kernel 7.2.7)
- oneAPI: 2026.1 (compiler + oneMKL), Level-Zero loader 1.33.1 (extracted, not installed system-wide)

## Build (SYCL engine)

The SYCL port builds cleanly here:

```sh
cd /home/otherdrums/strata
source /opt/intel/oneapi/setvars.sh
export PATH=/tmp/opencode/ninja/usr/bin:$PATH
export LD_LIBRARY_PATH=/tmp/opencode/ze/usr/lib64:$LD_LIBRARY_PATH
cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/home/otherdrums/llama.cpp
cmake --build build-sycl -j 8
```

Result: `build-sycl/strata` (43 MB), 0 compile errors. `tools/build_sycl_local.sh`
wraps this.

Level-Zero for the UHD 770 was made available by extracting
`oneapi-level-zero` and `intel-level-zero` RPMs into `/tmp/opencode/ze` and
`/tmp/opencode/ninja`; `sycl-ls` then reports
`[level_zero:gpu] Intel(R) UHD Graphics 770`.

## Kernel parity on this GPU

`ctest --test-dir build-sycl` with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
gave 17 failures out of 34. Two distinct causes:

- `Required aspect fp64 is not supported on the device` (shared_expert, gdn,
  quantize_act, s_gemv, elementwise, qsa, kv_*, verify, gr, ...).
- `The number of work-items ... cannot exceed {512,512,512}` (sycl_router_fast,
  sycl_payload_pairs).

The UHD 770 lacks the fp64 aspect and the work-group sizes the port assumes
for Arc (Xe2). The port is written for discrete Arc (B-series / Alchemist)
and is not usable on this integrated part. Conclusion: **Intel GPU path is
not possible on this machine.**

## CPU fallback

`ONEAPI_DEVICE_SELECTOR=opencl:cpu` runs the parity tests on Intel's OpenCL CPU
runtime; the docs report 14 of 25 pass there, with failures from tight float
tolerances and timeouts. That is a compile/compute check, not a usable
inference path. Strata's own engine has no CPU-only inference mode; the
qwen35moe (Qwen3.6 / Cyber-Tiel) family is CUDA/HIP only.

## Model under test

Cyber-Tiel-Coder-35B-A3B (Qwen3.5 MoE, `qwen35moe` architecture, same family
Strata already supports via `OWN_PATH_FAMILIES`):

- Source: https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP
- File: `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`
  (22,749,880,640 bytes,
  sha256 `0bbcf3cc9be4c976bad20e641baf629dad9c178d39ebdc9cd72129179943c06a`)
- mmproj: `mmproj-BF16.gguf`
  (902,822,016 bytes,
  sha256 `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837`)
- Stored locally at `/home/otherdrums/models/cyber-tiel/` (not in the repo).

Running it requires a CUDA (NVIDIA) or HIP (AMD) card; on this machine the
model cannot be executed. The same method applies to the other members of the
`qwen35moe` family Strata already supports (Qwen3.6-35B-A3B).

## Notes for a future PR

- Adding Cyber-Tiel as an own-path family would mirror the `qwen36` entry in
  `setup.py` (`OWN_PATH_FAMILIES`), with its own HF repo, file pattern and the
  hashes above. This is left as a local note rather than a code change, because
  it cannot be validated end to end on this hardware.
- The SYCL port's fp64 / work-group assumptions are a portability gap worth
  raising upstream if broader Intel iGPU support is desired.
