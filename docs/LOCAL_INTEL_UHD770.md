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

## iGPU bring-up progress

- Removed device-side `double` accumulations and `sycl::aspect::fp64` capability
  gates from the SYCL kernel files (`sycl/src/kernels/cuda/*.dp.cpp`), so kernels
  no longer require FP64 on the UHD 770.
- gdn parity now passes; remaining failures are work-group sizes of 1024 (device
  max 512) in doorbell/sampler/verify/qsa/native kernels, plus numerical
  mismatches in rope/gr/shared_expert/ple.
- Next: split 1024-wide launches into 512 (or smaller) groups and adjust grid
  sizes and per-thread loops, then re-run parity.

## Latest parity (with IGC FP64 emulation, work groups reduced to 512)

Passing: bf16_gemv, cvec, dequant_s2, elementwise, gdn, kv_q4, kv_q8, kv_steps,
native_grouped, native_multi, qsa, quantize_act, rope, router_top10, s2_gemv,
s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert, verify, conversation_snapshot,
sycl_router_fast, sycl_payload_pairs.

Remaining failures:
- `gr_parity`: fused GR read + q8_1 (STRATA_QFUSE) direct/graph replays not written.
- `kv_stream_parity`: work-group size of 1024 exceeds the device max of 512; the
  resolve kernel needs a 512-thread rewrite that preserves the 1024-slot clock sweep.
- `ple_parity`: skipped (needs the Q2_0 shard 2 GGUF).
- `s2_expert_grouped_parity`: 4 numerical failures.

## Running Cyber-Tiel (CUDA/HIP only)

This machine has no CUDA/HIP GPU, so the model cannot be executed here. On a supported
NVIDIA/AMD card the same own-path flow as Qwen3.6-35B-A3B applies:

```sh
# example only; adjust paths and quant
./setup.sh --family qwen36 ...   # see OWN_PATH_FAMILIES in setup.py
```

Cyber-Tiel file used for this note:
- `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`
- `mmproj-BF16.gguf`
- local path: `/home/otherdrums/models/cyber-tiel/`

The Intel iGPU path above is experimental and currently passes most kernel parity
tests, but not yet the full engine. The remaining failures are documented above.

## The qwen35moe family on an Intel GPU (Cyber-Tiel 35B)

The family Strata already supports for `Qwen3.6-35B-A3B` (`OWN_PATH_FAMILIES`, `src/qwen35/`) is the
architecture **Cyber-Tiel-Coder-35B-A3B** is built on (`qwen35moe`), and it was CUDA/HIP only: the
SYCL build refused it outright because `src/qwen35/q36_engine.cu` is a CUDA translation unit.

It is now migrated into `sycl/src/qwen35/` and **runs on this machine's UHD 770**:

```sh
cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_GGML_DIR=/home/otherdrums/llama.cpp
cmake --build build-sycl --target strata-q36 -j 8
IGC_EnableDPEmulation=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build-sycl/strata-q36 \
  --model /home/otherdrums/models/cyber-tiel/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --no-mtp-block --max-context 2048 --chunk 256 --tokens 1,2,3 --n 8
```

measured (UHD 770, i5-13500, 62 GB, oneAPI 2026.1, Level Zero 1.17):

| | |
|---|---|
| load | 753 tensors, 4.32 GiB of dense weights placed, every routed expert (18.33 GiB) in the shared memory, 5.4 s |
| warm-up (one chunk + one decode) | 11.5 s |
| prompt, 3 tokens | 269 ms (**11.2 tok/s**) |
| decode, 8 tokens | 912 ms (**8.8 tok/s**) |

**The answers are real now.** The NaN was not in any kernel: `upload_f32` (the f32 weight uploader)
returned the device pointer before dpct's asynchronous `memcpy` finished, so the GDN layer's
`ssm_dt.bias` / `ssm_a` (32 floats each) were read as 8.4e-45 and the state step overflowed. One
`.wait()` on the memcpy fixes it; the model now emits real tokens:

  prompt 3 tokens in 277 ms (10.8 tok/s), 4 generated in 392 ms (10.2 tok/s)
  prompt 3 tokens in 277 ms (10.9 tok/s), 16 generated in 1952 ms (8.2 tok/s)

The remaining 300 verify mismatches are the `q8_small` (180) and `shared_gu q8` (120) checks, the
same FMA-ordering class as the pre-existing `s2_expert_grouped_parity` failure (within 1e-7 of the
reference, fails on the CPU SYCL build too). Parity suite: 31/34, unchanged from the pre-existing
three (`iq_parity`, `ple_parity`, `s2_expert_grouped_parity`).

Flags the port added on the way:

- `strata-q36` (the family's own entry point) as a build target, as in the CUDA build.
- `--kv-q8` for int8 K/V cache (benthic-io PR #3084), FP16 by default.

The MTP draft block (Q3_K experts) now runs natively via the new `vec_dot_q3_K_q8_1` kernel, so
`--no-mtp-block` is no longer needed for Cyber-Tiel's UD-Q4_K_XL.  The kernel is scalar (no vmmq
integer chain) - correct first, fast later. `--no-mtp-block` remains as an opt-out for the draft head.

## K/V cache: FP16 by default, int8 with `--kv-q8`

The port now carries benthic-io's PR #3084 (FP16 K/V cache by default, int8 with `--kv-q8`).
The qwen35 engine stored its K/V cache in FP32 with no option; FP16 is now the default and
`--kv-q8` stores int8 codes with one FP16 scale per 64 dims (the main engine's `kv_q8.hpp`
scheme).  On a 16 GiB card that is the difference between 192K and 262144 cells.

Both paths run and produce real tokens on the UHD 770; the decode rate is essentially the same
(8.8-10.2 tok/s) because the iGPU is compute-bound on the expert GEMMs, not on the cache
bandwidth - the saving shows up in how much context fits, not in tok/s.  Measuring the expert/iGPU
split that the shared memory bus actually wants is the next piece of work.
