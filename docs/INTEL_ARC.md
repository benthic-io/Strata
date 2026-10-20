# Intel Arc (experimental)

Strata 0.1.39 includes an **experimental Intel Arc engine**: Strata's own engine ported to SYCL (Intel oneAPI).
maxfridbe wrote it in [#423](https://github.com/Niko1221/Strata/pull/423), with fixes from the people testing it.
The code is in `sycl/`, and the port's own notes, measurements and maintenance procedure are in
[docs/INTEL.md](INTEL.md). This page covers what you need, how to build it, and what has been tested.

**Experimental means:** the Strata maintainers have no Intel GPU. We compile the port and run its kernel tests, but
we have not run it on an Arc. Every result on Arc hardware below comes from the community. The NVIDIA and AMD
engines are unchanged: the Intel build is a separate CMake target, off by default.

There is **no ready-made Intel engine** in the release zips. You build it from source on Linux.

## What has been run, and by whom

| | Hardware | Result | Reported in |
|---|---|---|---|
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04 | Coder IQ1_M: 70-78 tok/s decode, ~790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context measured | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt (with the `stage_room` fix that is now in 0.1.39) | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M ~23 tok/s, IQ2_XS ~25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS ~21 tok/s, Coder ~15 tok/s; **device loss also seen** | #423 |
| Community | 2x Arc Pro B60 24 GB, `--layer-split` | Coder IQ1_M: 56.6 tok/s decode, 428 tok/s prompt (2,129 tokens); Flash-Next IQ2_XS: 58.6-61.3 tok/s decode, 436 tok/s prompt; all experts in VRAM; 8K context | [bench/results/2026-10-04-community-2x-arc-pro-b60](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| Community | one Arc Pro B60 24 GB | Coder IQ1_M with 4,042 of 12,288 experts mirrored in RAM: 11.9 tok/s decode (needs the ring-wait fix from the same report) | same |
| Strata maintainers (0.1.40.2 and later) | Arc Pro B70 32 GB (xe) and Arc A750 8 GB (i915), test machines | Flash-Next IQ3_S on the B70 with 11.5k of 24.6k experts resident: 10-prompt gate passes, prompt 980-1,000 tok/s on 4K tokens, decode 30 (prose) to 41 (code) tok/s; the xe aliased-pages fix and the `NO_HOST` requirement are in [INTEL.md](INTEL.md) | INTEL.md, "Arc Pro B70 with a model that does not fit" |

The 0.1.39 port re-migrates 0.1.38's port onto the 0.1.39 engine sources (the #606 NaN fix, the #649 verify
trace, the new prompt paths). 0.1.39's `sycl/` did not compile against 0.1.39's own engine sources (#784: the
`ThreadAffinity` type of #626 and the layer range of `NativeDense::load` from #559), and its ring waits never saw a
slow layer as still running (#866, #867). Both are fixed in 0.1.40, and the two B60 reports below were measured with
exactly those two fixes on top of 0.1.39. **No one has run an unpatched 0.1.39 port on an Arc**, and the other rows were
measured on earlier versions.

The two B60 rows ran `6f32ec0` plus two small `sycl/` fixes (the compile fix and the ring-wait fix), AOT `bmg-g21`, on Ubuntu 24.04 with
`xe`, Level Zero V2, NEO 26.09.37435.12 and oneAPI 2026.1.1, without Docker. Host: Ryzen 5 5600, 64 GB RAM, PCIe 3.0 x8 per card.
Full flags, per-request timings and engine logs are in the report linked in the table.

## Integrated GPUs (a UHD 770 and the other Intel iGPUs)

`./setup.sh --backend sycl` finds an **integrated** part as well as a discrete Arc, and runs it through the same
i915 path an Arc A750 takes. It is recognised by where it sits, not by its id: Intel puts the platform's own
graphics at **PCI 00:02.0** on every client part, and a discrete card hangs off a PCIe bridge. Its memory BARs do
not tell it apart - a UHD 770 publishes the same 4 GiB aperture an Arc does. `STRATA_EXPERIMENTAL_INTEL_IGPU=1` takes
any other 8086 id as integrated, the same opt-in the AMD APU ids use.

An iGPU has no VRAM of its own, so:

- it is sized from the **shared pool** (half the RAM, less the OS's share - setup.py's APU rule), and setup does not
  count that as room *beside* the RAM: `--check` says every size "fits, the experts the card does not hold are
  computed on the CPU";
- the experts live in a **RAM arena** and the CPU computes the ones the GPU does not hold (`STRATA_VERIFY_NO_HOST=0`),
  so the GPU's own memory holds only the dense weights, the KV and the prompt buffers;
- FP64 is emulated, as on the A-series (`IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`);
- the n-gram table (28.8 GB) is only locked in RAM when the RAM has room for it beside the arena; otherwise it is
  read from the SSD, and setup says so;
- it is **not** given the small-card 300 MiB reserve, which is advice for a card with real VRAM;
- `strata-sycl.sh` runs the engine **without Docker** when the machine has none (or `STRATA_SYCL_NATIVE=1`), sourcing
  oneAPI's `setvars.sh` and rewriting the config's `/work` paths - an iGPU is usually a desktop where installing
  Docker is the larger ask.

**What this costs.** The GPU is the small part: on a UHD 770 the engine keeps under 4 GiB of experts (the driver's cap
on a single allocation, which `--expert-cache auto` now respects) and the CPU computes the rest, so the speed is a CPU
number with the dense projections, the attention and the sampling offloaded to the iGPU. When the RAM is large enough
for the whole model the experts stay on the GPU and the CPU is not used. The parity suite passes on a
UHD 770 **32 of 35**; the three that do not are:

| test | why |
|---|---|
| `ple_parity` | needs the 28.8 GB Q2_0 shard-2 GGUF as a fixture (not a failure) |
| `iq_parity` | Q3_K's multi-column mmvq is not bitwise equal to its one-column calls: Q3_K's single-column call goes through a dedicated kernel and the multi-column one through the generic kernel, so the sums run in a different order. No Strata model uses Q3_K. |
| `s2_expert_grouped_parity` | the grouped expert kernel sums in a different order than the per-hit kernel it replaces (FMA contraction differs on this compiler). Both are within 1e-7 of the test's own double-precision reference, and the engine picks one path per run, so a run is deterministic - it just does not produce CUDA's last bits. |

**Measured end to end** on a UHD 770 (i5-13500, 62 GB of RAM, Fedora 44, oneAPI 2026.1, all 256 experts in device
memory): Cyber-Tiel-Coder-35B-A3B (`qwen35moe`, UD-Q4_K_XL, 22.7 GiB) decodes at **8.2 tok/s** and processes a prompt
at **24.9 tok/s** with `--kv-q8` and a 262144-token context. `--mtp 1 --draft-vocab 65536` adds the model's
speculative block: **8.8 tok/s**, 88% of drafts accepted, and the same tokens as plain greedy. A layer ablation puts
51-64% of a step in the routed experts, which run at ~10.5 GB/s against the 31 GB/s the dense projections reach; that
is the next lever, and it is why the totals are where they are.

The 35B `qwen35moe` family (Cyber-Tiel, Qwen3.6-35B-A3B) runs on Intel through the port's own copy of its engine
(`sycl/src/qwen35/`), which `strata` selects for a `qwen35moe` model; the numbers above are that path, and
`strata-q36` is its standalone entry point. The fast paths behind them are the ones this port measured: the
multi-column Q8_0 MMVQ (2.1-2.9x at 2-8 columns, bitwise identical to the one-column calls), the grouped Q3_K
expert dot (the port's `QI3_K` was 8 where ggml's is 16, so the dot unpacked the block wrongly), and the shared
expert's separate products in place of the fused kernel (7.7 -> 8.1 tok/s: the fused kernel's grid is 16 blocks).

**Not supported:** an iGPU that the Level Zero runtime does not list (`sycl-ls` shows nothing), and Windows.

## What was tested here (0.1.39)

- **Compiles:** Ubuntu 24.04 (WSL2), Intel oneAPI DPC++ 2026.1.1 + oneMKL 2026.1. The whole `sycl/` project
  builds with 0 errors: the `strata` engine plus all 157 targets (the kernel parity tests and benches). The build is
  SPIR-V (JIT). The AOT build (`STRATA_SYCL_AOT`) was not built here, because it needs `ocloc`.
- **Kernel parity tests on a CPU** (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`, Intel's OpenCL CPU runtime, AMD Ryzen 5 7600):
  14 of 25 pass. That is the same set that PR #423's own 0.1.38 port passes on that device: the failures are tight
  float tolerances on the CPU's math (rel 3e-6 against a 1e-6 limit), model fixtures that are not present, and two
  tests that time out on a CPU. This is a check that the kernels compile and compute, not a test of an Arc.
- **Setup:** `--backend sycl` warns, then hands over to `sycl/setup_intel.py` on Linux (unit tests:
  `tools/test_setup_sycl.py`).
- **Unchanged:** the CUDA engine's greedy output, checked byte-identical against the gated 0.1.39 build (Q2_0 and Coder).
  The HIP build is not affected (the option is off by default).

**Not tested by anyone yet:** Windows (no native build path; see below), Arc on WSL2 for the 0.1.39 port, the Alchemist
A-series, and images (not wired on Intel). Integrated GPUs are described above: they run, their kernel parity is
measured, and a 35B `qwen35moe` model now has an end-to-end number on one.

## What you need

- **Linux**, e.g. Ubuntu 24.04, with Intel's GPU driver (the `xe` or `i915` kernel driver plus the compute runtime /
  Level Zero; on Ubuntu, `intel-opencl-icd libze1 libze-intel-gpu1`, or Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html)).
- **Intel oneAPI**: the DPC++ compiler (`icpx`, 2025.3 or newer; 2026.1 is what was built here) and **oneMKL**. About 5 GB.
- `cmake` 3.24+, `ninja`, `git` (the build fetches ggml unless you point `STRATA_GGML_DIR` at a llama.cpp checkout),
  Python 3.
- For `./setup.sh --backend sycl` today: **Docker**, and nothing else from the list above except the GPU driver.
  `sycl/setup_intel.py` runs the engine in the `strata-sycl-dev` image built from `sycl/tools/Dockerfile`, and the
  engine is built in that image too ("Build with Docker" below). On Ubuntu: `sudo apt install docker.io git`, then
  `sudo usermod -aG docker,render $USER` and log in again. Note that the Dockerfile starts from a community
  llama.cpp SYCL image (`ghcr.io/snailium/...`), not an Intel or Strata image.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS.
  Smaller cards mirror part of the experts in RAM and are slower.

## Build with Docker (what `setup --backend sycl` uses)

Two commands, the same as in [INTEL.md](INTEL.md), "How to build it" (about 10 minutes the first time). From the checkout,
called `Strata` here:

```sh
cd Strata
docker build -t strata-sycl-dev sycl/tools            # once: ~6 minutes, a 13 GB image
H=$(basename "$PWD")
# Arc B-series: bmg-g31 (Pro B70), bmg-g21 (B580, B570, Pro B60). Arc A-series (Alchemist): leave out AOT (JIT) and use build-sycl
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

Then `./setup.sh --backend sycl` (below). Do not run `python3 sycl/setup_intel.py` by hand: on Ubuntu it fails with
`externally-managed-environment`; `setup.sh` runs it in its own `.venv`.

## Build without Docker (Linux)

This builds the engine for running it by hand ("How to run it by hand" in [INTEL.md](INTEL.md)); `setup` itself uses the
Docker build above. Install oneAPI from Intel's apt repository (this is what was used here):

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

Then build the engine. Either command works: the first goes through the top-level CMake option, the second
configures the `sycl/` project directly.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

Options:

- `-DSTRATA_SYCL_AOT=bmg-g31` (Arc Pro B70) or `bmg-g21` (B580 / B570 / Pro B60) compiles the GPU code ahead of
  time. This needs `ocloc` (Intel's `intel-ocloc` package). Without it, the first start JIT-compiles every kernel,
  which takes about 47 s.
- `-DSTRATA_SYCL_PARITY=OFF` skips the kernel tests (on by default). Run them with
  `ctest --test-dir build-sycl` on the card.

`setup_intel.py` looks for `build-sycl-aot/strata` or `build-sycl/strata` in the checkout. With the
top-level option the engine is at `build-sycl/sycl/strata`, so either use `-S sycl` or set `STRATA_SYCL_BIN`.

## Setup and running

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

This prints the experimental warning and continues with `sycl/setup_intel.py` (in setup's own Python environment).
That script finds the Arc in sysfs, uses the SYCL engine you built, and writes the config and `run-<model>.sh`.
On a card under 12 GB or an `i915` card (the A-series) it writes the small-card config ([INTEL.md](INTEL.md), "Arc A750
and the other Alchemist cards"). It still downloads and packs the model
the usual way. To run the engine by hand (no Docker), see "How to run it by hand" in [INTEL.md](INTEL.md).

Things that matter on an Arc (details in INTEL.md):

- **Do not ask for more VRAM than the card has.** On the `xe` driver, an allocation past VRAM can push buffers into
  RAM until the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` now passes the
  variable through) and `--layer-split`.
- **`--layer-split auto` on two equal cards (0.1.44, #1440):** when several splits hold all the experts and predict the
  same speed, the search now takes the one that leaves its fullest card the most VRAM to spare (within 0.2% of the best
  predicted time, `STRATA_SPLIT_HEADROOM_TOL`). It used to take the first one tried, which filled the second card to its
  last MiB. A full card under `xe` moves buffers to system memory instead of failing: 2x B70 with IQ3_S, auto picked
  K=24 (~11 tok/s) where `--layer-split 30` ran ~54. The startup line names both splits when the choice moves.
  `STRATA_SPLIT_HEADROOM=0` keeps the old pick.
- **Arc Pro B60:** the PCI id is `8086:e211` (`lspci -nn`, the kernel's `xe` id list files it with the BMG-G21 cards), `sycl-ls` prints
  `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, and `ocloc ids bmg-g21` prints 20.1.0, so `-DSTRATA_SYCL_AOT=bmg-g21` is the right target.
  `setup_intel.py` knows both B60 ids, `e211` and `e221` (both seen on B60 cards). On a `--layer-split` the startup line "N experts are neither in VRAM nor mirrored"
  counts the other card's layers; the lines `100% of the experts resident` that follow are the ones to read. Set `STRATA_MIRROR_MIB=0` there,
  or the pinned mirror is allocated and not used.
- **`setvars.sh` and `set -u`:** `source /opt/intel/oneapi/setvars.sh` in a shell with `set -u` stops at `OCL_ICD_FILENAMES: unbound variable`
  (oneAPI 2026.1.1). Source it before `set -u`, or run `set +u` around it.
- **Without Docker:** `sycl/serve/strata-sycl.sh` needs Docker. Natively, `source setvars.sh`, export `SYCL_CACHE_PERSISTENT=0`,
  `ZES_ENABLE_SYSMAN=1` and `STRATA_VERIFY_DEVICE_PLAN=1`, then run `build-sycl-aot/strata` or `sycl/serve/server_intel.py` with an `exe` that does this.
  A normal user is not in the `render` group on Ubuntu, and then `sycl-ls` shows only the CPU device.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh` unless the config's `"env"` says otherwise) is needed on `xe`
  cards when part of the experts is not in VRAM (the host-to-GPU flag stores are not visible there, NaN logits
  otherwise) and is left off on the `i915` Arc A-series: see [INTEL.md](INTEL.md). The config's `"env"` block and any
  `STRATA_*`, `ONEAPI_*`, `UR_*`, `IGC_*`, `SYCL_*` or `ZES_*` variable of the server's environment reach the engine
  through `strata-sycl.sh` (for example `STRATA_MIRROR_MIB`, `IGC_EnableDPEmulation`).

## Windows

There is no Windows path yet. `setup --backend sycl` on Windows stops and points here. oneAPI exists for Windows,
but `sycl/CMakeLists.txt` uses GCC-style flags (`-mavx512f`, `-fp-model=precise`, `-qmkl`) and the runner is a
bash/Docker script, so a native Windows build would need work. Nobody has tried it. WSL2 with an Arc has been
used by one tester (the B580 row above), but setup cannot detect the card there, because it reads `/sys/class/drm`,
which WSL2 does not have.

## Reporting a problem

Open an issue with: the card, the driver version, `sycl-ls` output, the oneAPI version, the model and flags, and the
engine's stderr. Results from real cards are what move this from experimental to supported.
