// sycl/include/strata/sycl_unified.hpp - is this Intel GPU the CPU's own graphics?
//
// The CUDA engine asks the runtime (`cudaDevAttrIntegrated`); SYCL has no such aspect and dpct's `get_integrated()`
// is 0 in a SYCL build, so without this the port would size an expert cache from whatever the driver calls free.
// On an integrated part "free" is the free RAM, and a cache sized that way is both wrong (the experts live in RAM
// there, see sycl/setup_intel.py) and impossible to allocate (Level Zero caps one allocation at 4 GiB on these).
//
// The test is the one setup_intel.py uses, so the installer and the engine agree without either knowing the other's
// number: Intel puts the integrated graphics at PCI 00:02.0 on every client part, and a discrete card hangs off a
// PCIe bridge.  STRATA_UNIFIED_MEMORY=1 forces it on and =0 off, for a part the address does not settle.
//
// SYCL port only: nothing here is upstream's.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata {
namespace sycl_detail {

/// "0000:00:02.0" -> the PCI address the SYCL runtime reports, or "" when it cannot (a CPU device, or a backend
/// without the query).  The Intel extension is guarded: it is not on every SYCL implementation.
inline std::string device_pci_address(const sycl::device& d) {
    try {
        return d.get_info<sycl::ext::intel::info::device::pci_address>();
    } catch (const sycl::exception&) {
        return std::string();
    }
}

/// True when the address is PCI bus 0 device 2, where Intel puts the platform's own graphics.
inline bool is_bus0_dev2(const std::string& pci) {
    unsigned domain = 0, bus = 0, dev = 0, func = 0;
    if (std::sscanf(pci.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) return false;
    return bus == 0 && dev == 0x02;
}

/// Whether device `d` is the platform's integrated graphics (unified memory: its "VRAM" is system RAM).
/// STRATA_UNIFIED_MEMORY=1 / =0 overrides the address test.
inline bool unified_memory_device(const sycl::device& d) {
    const char* forced = std::getenv("STRATA_UNIFIED_MEMORY");
    if (forced != nullptr && forced[0] != '\0') return forced[0] != '0';
    if (d.is_cpu()) return false;
    return is_bus0_dev2(device_pci_address(d));
}

/// The largest single device allocation the driver will give (bytes), 0 when it does not say.
/// Level Zero caps one allocation at 4 GiB on the consumer parts (an Arc A750, a UHD 770), so a buffer is split
/// into several below this; cudaMalloc on CUDA has no such limit, which is why upstream never asks.
inline size_t max_single_alloc_bytes(const sycl::device& d) {
    try {
        return (size_t) d.get_info<sycl::info::device::max_mem_alloc_size>();
    } catch (const sycl::exception&) {
        return 0;
    }
}

}  // namespace sycl_detail
}  // namespace strata