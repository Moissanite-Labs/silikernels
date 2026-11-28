// SPDX-License-Identifier: Apache-2.0
// Compilation unit for V2 Ampere sm_86 kernels (audited via .o, .so, and bench SASS gates).

#include "a16_v2_ampere_kernels.cuh"

extern "C" uint32_t a16_v2_sm86_kernel_abi_version(void) {
    return 0x00020001u;
}
