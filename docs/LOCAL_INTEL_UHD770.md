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

## Performance: what the expert split actually costs, and the full 262144 context

Measured on an idle machine (i5-13500, 62 GB, oneAPI 2026.1, `strata-q36`, three bench
repetitions, the first discarded as warm-up).  **Run benchmarks one at a time**: these numbers
came out of a first pass where four of them ran together, and the shared memory bus and the
page cache turned 27 tok/s prompts into 20 and the free-VRAM figure into nonsense (6.68 GiB
reported where an idle machine reports 28).

### The expert/iGPU split does not move decode

| slots in VRAM (of 256) | CPU pool | pp512 tok/s | tg128 tok/s |
|---|---|---|---|
| 64  | 68.8% on GPU | **27.5** | 7.3 |
| 96  | 74.8%        | **27.5** | 7.2 |
| 128 | 80.8%        | 27.1     | 7.3 |
| 160 | 87.5%        | 25.6     | 7.3 |
| 214 | 95.4%        | 24.8     | 7.3 |
| all | none         | 24.9     | **7.4** |

Decode is flat: the iGPU is the bottleneck whether the CPU pool takes 5% or 31% of the rows, so
moving experts between them buys nothing.  The prompt is the other way round - the CPU's AVX2
helps the big prompt batches, so a few experts on the CPU (64-96 slots) is the best prompt
throughput, ~10% over all-in-VRAM.  All-in-VRAM needs no CPU pool at all and is the simplest
setup; the engine's `--expert-cache auto` picks it when the machine is idle and falls back to
the hybrid when the RAM is busy (that is what the "6.68 GiB free" above was).

### The whole 262144 context fits

Cyber-Tiel's native context is 262144 cells.  With every routed expert in VRAM:

| context | K/V cache | free VRAM after the experts | tg32 tok/s |
|---|---|---|---|
| 4096    | FP16 | 9.7 GiB   | 7.4 |
| 32768   | FP16 | 9.0 GiB   | 7.4 |
| 131072  | FP16 | 6.5 GiB   | 7.3 |
| 131072  | q8   | 7.8 GiB   | 7.4 |
| 262144  | FP16 | 3.0 GiB   | 7.2 |
| 262144  | **q8** | **5.7 GiB** | 7.3 |

Both formats run at the full 262144; `--kv-q8` leaves ~2.7 GiB more headroom and costs nothing
in throughput (decode is latency-bound, not bandwidth-bound).  `--kv-q8` is the setting to ship
for long context.

### What does not help

- `STRATA_GDN_XE=1` (16-lane sub-group GDN): within noise on every shape (24.9 / 7.4 both
  ways); the Xe recurrence is actually slower at T=1/7/1000 and faster only at T=256.  Leave off.
- `--no-graph`: the decode graph is essential; without it 512+128 tokens did not finish in
  seven minutes against ~100 s with it.
- Dropping `IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`: the run is not faster and
  the greedy text changes (the rope and QK paths use doubles), so the flags stay.

### Speculative decoding: the block computes now, the draft loop does not

The Q3_K expert kernel was wrong in a way that made the MTP block produce noise (see the section
below).  With that fixed, the block's own predictions are right: fed a constant prompt it agrees
with the main model on **507 of 511** rows (it was 0 of 511 before the fix), and the `hmtp` /
logits / argmax chain is consistent.

The draft loop around it still is not usable.  On real text the first draft is usually wrong, and
the chained drafts freeze at the same four tokens (598, 11113, 81842, 79874) on every step even
though `tok_m`, `pos0_m` and the seed all change; with `--mtp 4` that is 0 of 124 accepted and
1.9 tok/s against 7.9 without.  A debug build that waits on the queue inside the step gets 5 of 40
(12%), so this is a synchronization/state bug in the MTP path (the block's own K/V or the chained
seed), not in the expert kernel.  Leave `--mtp` off; the plain greedy decode is unaffected.

### Q3_K: the grouped expert dot was wrong

`vec_dot_q3_K_q8_1` in the SYCL port unpacked Q3_K linearly (`qs[p/4]`, `hmask[p/8]`,
`scales[p/16]`) with `QI3_K` hardcoded to 8.  ggml's layout is not linear: a 256-value block is two
128-halves, each in four 2-bit shift groups, the high bit comes from `hmask[l]` bit `4n+j`, and the
16 six-bit scales index `8n+2j+(l/16)`.  `QI3_K` is `QK_K/(4*QR3_K)` with `QR3_K` 4, i.e. **16**,
and the per-call unit is 16 values, not 32.

The port now runs llama.cpp's `vecdotq.cuh` MMVQ kernel.  Measured on a real
`blk.40.ffn_gate_exps.weight` expert, one group and one entry, against `iq_dequant_f32`'s
dequantised reference:

| stage | before | after |
|---|---|---|
| gate | 1.4 rel | **9.2e-8** |
| up | 1.4 rel | **9.3e-8** |
| MoE output | 32.5 rel | **7.3e-3** |

The 7.3e-3 is the SwiGLU intermediate's q8_1 rounding; Q4_K/Q5_K measure 5.2e-3 on the same test.
The dense `native_mmvq` has its own Q3_K kernel and was already right, which is why the error only
showed in the grouped expert path (the MTP block and any Q3_K-expert model).
