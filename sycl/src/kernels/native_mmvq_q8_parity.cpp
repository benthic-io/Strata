// sycl/src/kernels/native_mmvq_q8_parity.cpp - Q8_0's multi-column contract, bitwise, plus an in-process
// old/new timing.
//
//     build-sycl/native_mmvq_q8_parity            the bitwise checks
//     build-sycl/native_mmvq_q8_parity --bench    the timing tables, then the checks
//     build-sycl/native_mmvq_q8_parity --paths    the library's single-column path against the old-kernel copy
//                                                 (a diagnostic that also proves the copy faithful)
//
// WHAT IT CHECKS.  `include/strata/kernels/native_mmvq.hpp`: "true (default): the ncols == 1 layout, every column
// bitwise equal to a single-column call".  One ncols = 2..8 call is compared with ncols = 1 calls on the SAME Q8_1
// bytes, column j read from xq + j * (n_in / 32) blocks, outputs memcmp'd.  Shapes: the model's 2048x2048 dense
// matrix (64 blocks, the staging tile exactly), a row longer than one tile (n_in = 2560), a short row with a
// partial last 16-row group (1024x257), and the tiled fallback at n_in = 4096.
//
// THE OLD KERNEL.  `old_wide32` below is the wide32 kernel the Q8_0 multi-column path used before the row-shared
// one (including load16_a2, the 32-lane butterfly and the product rounded before the accumulator add, which is what
// the shipped wide32 emits).  Run this binary with STRATA_MMVQ_Q8_ROWS=0 and the shipped old path must equal it
// column for column; the default run compares the shipped new path against single-column calls.  Its output is
// also the timing baseline, measured in the same process (blocks of old calls then blocks of new) so another
// process's GPU load moves both numbers together.  The link line matters to IGC: with the engine's
// `-ze-opt-greater-than-4GB-buffer-required` the new kernel is ~5-15% slower than under a plain `icpx -fsycl`
// link, so the table also prints the shipped path alone for the same process (the engine's steady state).
//
// THE DATA.  Random Q8_0 weight bytes, except each block's fp16 scale, which is a normal fp16 in +-[2^-10, 2^-5):
// random bytes there would produce inf/NaN, and every NaN has the same bits on the GPU, which would hide a
// difference.  Activations are normal floats quantized by native_quantize_q8_1.  Every output must be finite.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

constexpr int WARP = 32;
constexpr int WARPS = 4;

struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct Q81Block {
    sycl::half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);

// ---- the old wide32 kernel and its load helpers, copied verbatim from native_mmvq.dp.cpp
inline sycl::int4 ld_i4_a4(const int* p) {
    sycl::int4 v;
    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p));
    return v;
}
inline sycl::int4 ld_q8_16(const Q81Block* b, int half) {
    return ld_i4_a4(reinterpret_cast<const int*>(b->qs) + 4 * half);
}
inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dpct::dp4a(a.x(), b.x(), acc);
    acc = dpct::dp4a(a.y(), b.y(), acc);
    acc = dpct::dp4a(a.z(), b.z(), acc);
    return dpct::dp4a(a.w(), b.w(), acc);
}
inline sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;
    const bool half = (a & 2) != 0;
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}

template <int NCOLS>
void old_wide32(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    constexpr int BPS = WARP / 2;
    const int blocks_per_row = n_in / 32;
    const int l = lane % 2, sub = lane / 2;
    const Q80Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        const sycl::int4 q = load16_a2(wr[kb].qs + 16 * l);
        const float d = (float) wr[kb].d;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * blocks_per_row + kb;
            const float t = d * (float) xb->ds[0] * (float) dp4a4(q, ld_q8_16(xb, l), 0);
            acc[j] += t;
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template <int NCOLS>
void launch_old(const Q80Block* w, const Q81Block* x, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    const sycl::range<3> t(1, WARPS, WARP);
    s->parallel_for<dpct_kernel_name<class old_wide32_q8, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * t, t),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { old_wide32<NCOLS>(w, x, y, n_in, n_out); });
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Fixture {
    int n_in = 0, n_out = 0;
    std::vector<uint8_t> w;
    std::vector<float> x;
    uint8_t* dw = nullptr;
    void* xq = nullptr;
    float* dx = nullptr;
    float* multi = nullptr;
    float* single = nullptr;
    float* old = nullptr;
};

bool make_fixture(Fixture& f, int n_in, int n_out, int ncols, dpct::queue_ptr s, std::mt19937& rng) {
    using namespace strata::kernels;
    f.n_in = n_in;
    f.n_out = n_out;
    const std::size_t wb = native_mmvq_weight_bytes(8, n_in, n_out);
    f.w.assign(wb, 0);
    for (std::size_t b = 0; b < wb / 34; ++b) {
        const uint16_t d = sane_half(rng);
        std::memcpy(&f.w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) f.w[b * 34 + 2 + i] = (uint8_t) (rng() & 0xff);
    }
    f.x.resize((std::size_t) n_in * ncols);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : f.x) v = nd(rng);
    const std::size_t yb = (std::size_t) n_out * ncols * 4;
    f.dw = (uint8_t*) sycl::malloc_device(wb, *s);
    f.xq = sycl::malloc_device(native_q8_1_bytes(n_in, ncols), *s);
    f.dx = (float*) sycl::malloc_device(f.x.size() * 4, *s);
    f.multi = (float*) sycl::malloc_device(yb, *s);
    f.single = (float*) sycl::malloc_device(yb, *s);
    f.old = (float*) sycl::malloc_device(yb, *s);
    if (!f.dw || !f.xq || !f.dx || !f.multi || !f.single || !f.old) return false;
    s->memcpy(f.dw, f.w.data(), wb).wait();
    s->memcpy(f.dx, f.x.data(), f.x.size() * 4).wait();
    native_quantize_q8_1(f.dx, f.xq, n_in, ncols, s);
    s->wait();
    return true;
}

// Returns the number of differing outputs; also counts non-finite ones (a test failure: with finite scales and
// activations a NaN would compare equal to itself whatever produced it).
long long compare(const float* a, const float* b, std::size_t n, long long& nonfinite) {
    std::vector<uint32_t> ha(n), hb(n);
    dpct::get_current_device().queues_wait_and_throw();
    auto s = dpct::get_in_order_queue();
    s.memcpy(ha.data(), a, n * 4).wait();
    s.memcpy(hb.data(), b, n * 4).wait();
    long long diff = 0;
    nonfinite = 0;
    double maxrel = 0.0;
    std::size_t first = n;
    for (std::size_t i = 0; i < n; ++i) {
        float fa, fb;
        std::memcpy(&fa, &ha[i], 4);
        std::memcpy(&fb, &hb[i], 4);
        if (!std::isfinite(fa) || !std::isfinite(fb)) ++nonfinite;
        if (ha[i] != hb[i]) {
            if (first == n) first = i;
            ++diff;
            const double rel = std::fabs((double) fa - (double) fb) / std::max(1e-30, std::fabs((double) fa));
            maxrel = std::max(maxrel, rel);
        }
    }
    if (diff && std::getenv("Q8PARITY_DEBUG")) {
        float fa, fb;
        std::memcpy(&fa, &ha[first], 4);
        std::memcpy(&fb, &hb[first], 4);
        std::printf("    first diff at %zu: %.9g vs %.9g (max rel %.3g, %lld diffs)\n", first, fa, fb, maxrel, diff);
    }
    return diff;
}

// ncols = 1 must still be wide32: compare it with the old-kernel copy, column by column.
long long check_single(Fixture& f, dpct::queue_ptr s, long long& nonfinite) {
    using namespace strata::kernels;
    native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, 1, s);
    launch_old<1>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s);
    s->wait();
    long long nf = 0;
    const long long d = compare(f.multi, f.old, (std::size_t) f.n_out, nf);
    nonfinite += nf;
    return d;
}

// One ncols call of the shipped path (multi) against ncols single-column calls (single), and the old kernel
// against the same single-column calls.
template <int NCOLS>
long long check_width(Fixture& f, dpct::queue_ptr s, long long& nonfinite) {
    using namespace strata::kernels;
    const std::size_t qcol = native_q8_1_bytes(f.n_in, 1);
    native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, NCOLS, s);
    for (int j = 0; j < NCOLS; ++j)
        native_mmvq(8, f.dw, (const uint8_t*) f.xq + (std::size_t) j * qcol,
                    f.single + (std::size_t) j * f.n_out, f.n_in, f.n_out, 1, s);
    launch_old<NCOLS>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s);
    s->wait();
    long long nf1 = 0, nf2 = 0;
    const long long d1 = compare(f.multi, f.single, (std::size_t) NCOLS * f.n_out, nf1);
    const long long d2 = compare(f.old, f.single, (std::size_t) NCOLS * f.n_out, nf2);
    nonfinite += nf1 + nf2;
    if (d2 != 0)
        std::printf("    the old-kernel copy differs from single-column calls at ncols %d: %lld outputs "
                    "(the reference kernel itself is wrong)\n", NCOLS, d2);
    return d1;
}

void run_bench(Fixture& f, dpct::queue_ptr s, int maxc) {
    using namespace strata::kernels;
    const std::size_t wb = native_mmvq_weight_bytes(8, f.n_in, f.n_out);
    const double mb = wb / 1e6;
    // Steady state: the engine calls one kernel repeatedly, so this is the number the target is about.  Batched,
    // no per-call wait, 200 calls after a warm-up.
    std::printf("\nshipped path alone, 200 consecutive calls (the engine's steady state)\n");
    std::printf("%5s %10s %9s\n", "ncols", "ms", "GB/s");
    for (int nc : {8, 5, 4, 2}) {
        for (int r = 0; r < 20; ++r) native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, nc, s);
        s->wait();
        double best = 1e9;
        for (int b = 0; b < 3; ++b) {
            const double t0 = now_ms();
            for (int r = 0; r < 200; ++r) native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, nc, s);
            s->wait();
            best = std::min(best, (now_ms() - t0) / 200);
        }
        std::printf("%5d %10.4f %9.2f\n", nc, best, mb / best);
    }
    std::printf("\n2048x2048 Q8_0: old (wide32 copy) against new (shipped path), one process, interleaved calls\n");
    std::printf("%5s %10s %10s %9s %9s %8s\n", "ncols", "old ms", "new ms", "old GB/s", "new GB/s", "speedup");
    // Blocks of calls, old then new, repeated: both run in this process (one GPU load, one clock state) but each
    // block gets its own warm cache instead of the two kernels evicting each other every other call.  The minimum
    // per-call time of each block is what is reported.
    const int reps = 64, blocks = 4;
    auto launch_old_n = [&](int ncols) {
        switch (ncols) {
            case 1: launch_old<1>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 2: launch_old<2>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 3: launch_old<3>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 4: launch_old<4>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 5: launch_old<5>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 6: launch_old<6>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            case 7: launch_old<7>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
            default: launch_old<8>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.old, f.n_in, f.n_out, s); break;
        }
    };
    for (int nc : {1, 2, 3, 4, 5, 6, 7, 8}) {
        launch_old_n(nc);
        native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, nc, s);
        s->wait();
        double bo = 1e9, bn = 1e9;
        for (int b = 0; b < blocks; ++b) {
            const double t0 = now_ms();
            for (int r = 0; r < reps; ++r) launch_old_n(nc);
            s->wait();
            const double t1 = now_ms();
            bo = std::min(bo, (t1 - t0) / reps);
            for (int r = 0; r < reps; ++r)
                native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, nc, s);
            s->wait();
            const double t2 = now_ms();
            bn = std::min(bn, (t2 - t1) / reps);
        }
        std::printf("%5d %10.4f %10.4f %9.2f %9.2f %7.2fx\n", nc, bo, bn, mb / bo, mb / bn, bo / bn);
    }
    (void) maxc;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool paths = argc > 1 && std::string(argv[1]) == "--paths";
    dpct::queue_ptr s = &dpct::get_in_order_queue();
    std::mt19937 rng(2026);
    int fail = 0;

    if (paths) {
        Fixture f;
        if (!make_fixture(f, 2048, 2048, 8, s, rng)) return 1;
        const std::size_t qcol = k::native_q8_1_bytes(f.n_in, 1);
        // library ncols=1 per column into f.single; old copy ncols=1 per column into f.old
        for (int j = 0; j < 8; ++j) {
            k::native_mmvq(8, f.dw, (const uint8_t*) f.xq + (std::size_t) j * qcol,
                           f.single + (std::size_t) j * f.n_out, f.n_in, f.n_out, 1, s);
            launch_old<1>((const Q80Block*) f.dw, (const Q81Block*) (f.xq) + (std::size_t) j * (f.n_in / 32),
                          f.old + (std::size_t) j * f.n_out, f.n_in, f.n_out, s);
        }
        s->wait();
        long long nf = 0;
        std::printf("library ncols=1 vs old-copy ncols=1: %lld diffs\n", compare(f.single, f.old, (std::size_t) 8 * f.n_out, nf));
        {
            std::vector<uint32_t> a((std::size_t) 8 * f.n_out), b(a.size());
            s->memcpy(a.data(), f.single, a.size() * 4).wait();
            s->memcpy(b.data(), f.old, b.size() * 4).wait();
            for (int j = 0; j < 8; ++j) {
                long long d = 0;
                for (int r = 0; r < f.n_out; ++r) d += a[(std::size_t) j * f.n_out + r] != b[(std::size_t) j * f.n_out + r];
                std::printf("  column %d: %lld diffs\n", j, d);
            }
        }
        // old copy ncols=8 vs its own ncols=1 columns
        launch_old<8>((const Q80Block*) f.dw, (const Q81Block*) f.xq, f.multi, f.n_in, f.n_out, s);
        s->wait();
        std::printf("old-copy ncols=8 vs old-copy ncols=1: %lld diffs\n", compare(f.multi, f.old, (std::size_t) 8 * f.n_out, nf));
        k::native_mmvq(8, f.dw, f.xq, f.multi, f.n_in, f.n_out, 8, s);
        s->wait();
        std::printf("library ncols=8 vs library ncols=1: %lld diffs\n", compare(f.multi, f.single, (std::size_t) 8 * f.n_out, nf));
        std::printf("library ncols=8 vs old-copy ncols=8: %lld diffs\n", compare(f.multi, f.old, (std::size_t) 8 * f.n_out, nf));
        return 0;
    }
    if (bench) {
        Fixture f;
        if (!make_fixture(f, 2048, 2048, 8, s, rng)) return 1;
        run_bench(f, s, 8);
        sycl::free(f.dw, *s);
        sycl::free(f.xq, *s);
        sycl::free(f.dx, *s);
        sycl::free(f.multi, *s);
        sycl::free(f.single, *s);
        sycl::free(f.old, *s);
    }

    std::printf("native_mmvq_q8_parity: ncols 2..8 vs ncols 1, bitwise\n");
    struct Shape { int n_in, n_out; const char* why; };
    const Shape shapes[] = {
        {2048, 2048, "the dense projections (64 blocks, one staging tile)"},
        {2560, 512, "row longer than one 64-block staging tile"},
        {1024, 257, "row shorter than one tile, partial last 16-row group"},
        {4096, 640, "row of 128 blocks, the tiled fallback"},
    };
    for (const Shape& sh : shapes) {
        Fixture f;
        if (!make_fixture(f, sh.n_in, sh.n_out, 8, s, rng)) {
            std::printf("allocation failed\n");
            return 1;
        }
        std::printf("%dx%d (%s)\n", sh.n_out, sh.n_in, sh.why);
        long long nonfinite = 0;
        long long mismatches[9] = {};
        const long long single_diff = check_single(f, s, nonfinite);
        std::printf("  ncols 1: %s (%lld of %d outputs differ from the old kernel copy)\n",
                    single_diff == 0 ? "bitwise (unchanged wide32)" : "*** NO ***", single_diff, sh.n_out);
        if (single_diff != 0) ++fail;
        mismatches[2] = check_width<2>(f, s, nonfinite);
        mismatches[3] = check_width<3>(f, s, nonfinite);
        mismatches[4] = check_width<4>(f, s, nonfinite);
        mismatches[5] = check_width<5>(f, s, nonfinite);
        mismatches[6] = check_width<6>(f, s, nonfinite);
        mismatches[7] = check_width<7>(f, s, nonfinite);
        mismatches[8] = check_width<8>(f, s, nonfinite);
        for (int nc = 2; nc <= 8; ++nc) {
            const bool ok = mismatches[nc] == 0;
            std::printf("  ncols %d: %s (%lld of %d outputs differ)\n", nc, ok ? "bitwise" : "*** NO ***",
                        mismatches[nc], nc * sh.n_out);
            if (!ok) ++fail;
        }
        if (nonfinite != 0) {
            std::printf("  *** %lld non-finite outputs: the comparison would not be meaningful ***\n", nonfinite);
            ++fail;
        }
        sycl::free(f.dw, *s);
        sycl::free(f.xq, *s);
        sycl::free(f.dx, *s);
        sycl::free(f.multi, *s);
        sycl::free(f.single, *s);
        sycl::free(f.old, *s);
    }

    std::printf("\nnative_mmvq_q8_parity: %s\n", fail ? "FAILED" : "OK");
    return fail ? 1 : 0;
}
