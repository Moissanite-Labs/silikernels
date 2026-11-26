# SPDX-License-Identifier: Apache-2.0
# Standalone Ampere sm_86 build & verification targets for the 0 B HBM-table
# 2-Stage Monarch-16x16 + 512 B Byte-Unigram Bias + FWHT_256 x bent4 LM Head
# (Refs Moissanite-Labs/silikernels#2).

NVCC   ?= nvcc
ARCH   ?= sm_86
PYTHON ?= python3
BUILD  ?= build

CUDA_FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v --Werror all-warnings

.PHONY: cuda-monarch2-head check-monarch2-head-zero-spill verify-monarch2-head

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libholosphere_monarch2_fwht_head_sm86.so: cuda/sm86/holosphere_monarch2_fwht_head_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

cuda-monarch2-head: $(BUILD)/libholosphere_monarch2_fwht_head_sm86.so

check-monarch2-head-zero-spill: | $(BUILD)
	@set -eu; \
	LOG=$(BUILD)/monarch2_head_ptxas_spill.log; \
	trap 'rm -f "$$LOG"' EXIT; \
	$(MAKE) -f cuda/sm86/holosphere_monarch2_fwht_head.mk -B cuda-monarch2-head >"$$LOG" 2>&1 || { cat "$$LOG"; exit 1; }; \
	if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG"; then \
		echo "ERROR: non-zero register/stack spill detected" >&2; \
		exit 1; \
	fi; \
	COUNT=$$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG"); \
	if [ "$$COUNT" -ne 4 ]; then \
		echo "ERROR: expected 4 zero-spill kernels, got $$COUNT" >&2; \
		exit 1; \
	fi; \
	echo "MONARCH2_HEAD_ZERO_SPILL_VERIFIED=1 ($$COUNT kernels)"

verify-monarch2-head:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/verify_monarch2_fwht_head.py --check --require-verdict CONFIRMED --json; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/test_monarch2_fwht_head_r5.py; \
	done
