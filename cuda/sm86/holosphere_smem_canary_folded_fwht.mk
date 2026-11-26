# SPDX-License-Identifier: Apache-2.0
# Standalone Ampere sm_86 build & verification targets for the CTA-SMEM packed .sys
# BER canary and 1 KiB FP16 folded FWHT_256 x bent4 head (Refs Moissanite-Labs/silikernels#2).

NVCC   ?= nvcc
ARCH   ?= sm_86
PYTHON ?= python3
BUILD  ?= build

CUDA_FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v --Werror all-warnings

.PHONY: cuda-smem-canary check-smem-canary-zero-spill verify-smem-canary

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libholosphere_smem_canary_folded_fwht_sm86.so: cuda/sm86/holosphere_smem_canary_folded_fwht_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

cuda-smem-canary: $(BUILD)/libholosphere_smem_canary_folded_fwht_sm86.so

check-smem-canary-zero-spill: | $(BUILD)
	@set -eu; \
	LOG=$(BUILD)/smem_canary_ptxas_spill.log; \
	trap 'rm -f "$$LOG"' EXIT; \
	$(MAKE) -f cuda/sm86/holosphere_smem_canary_folded_fwht.mk -B cuda-smem-canary >"$$LOG" 2>&1 || { cat "$$LOG"; exit 1; }; \
	if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG"; then \
		echo "ERROR: non-zero register/stack spill detected" >&2; \
		exit 1; \
	fi; \
	COUNT=$$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG"); \
	if [ "$$COUNT" -ne 5 ]; then \
		echo "ERROR: expected 5 zero-spill kernels, got $$COUNT" >&2; \
		exit 1; \
	fi; \
	echo "SMEM_CANARY_ZERO_SPILL_VERIFIED=1 ($$COUNT kernels)"

verify-smem-canary:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/verify_smem_canary_folded_fwht.py --check --require-verdict CONFIRMED --json; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/test_smem_canary_folded_fwht_r5.py; \
	done
