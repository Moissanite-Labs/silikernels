# SPDX-License-Identifier: Apache-2.0
# Bare-metal NVIDIA Ampere sm_86 HoloSphere S116 + S117 + W10-OBS-61 V3 Frontier Targets
# Included cleanly by top-level Makefile (Refs Moissanite-Labs/silikernels#2).

PYTHON ?= python3
CXX    ?= g++

HOLOSPHERE_CUDA_BINS := \
	$(BUILD)/libholosphere_megakernel_sm86.so \
	$(BUILD)/libholosphere_h256_chunk32_sm86.so \
	$(BUILD)/libholosphere_v3_frontier_sm86.so

CUDA_BINS += $(HOLOSPHERE_CUDA_BINS)

.PHONY: cuda-holosphere verify-holosphere check-holosphere-zero-spill verify-razor-abi

cuda-holosphere: $(BUILD) $(HOLOSPHERE_CUDA_BINS)

$(BUILD)/libholosphere_megakernel_sm86.so: cuda/sm86/holosphere_megakernel_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

$(BUILD)/libholosphere_h256_chunk32_sm86.so: cuda/sm86/holosphere_h256_chunk32_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

$(BUILD)/libholosphere_v3_frontier_sm86.so: cuda/sm86/holosphere_v3_frontier_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

verify-razor-abi: | $(BUILD)
	@set -eu; \
	TMP_C=$(BUILD)/_razor_abi_probe.c; TMP_CPP=$(BUILD)/_razor_abi_probe.cpp; \
	trap 'rm -f "$$TMP_C" "$$TMP_CPP" "$(BUILD)/_razor_abi_c" "$(BUILD)/_razor_abi_cpp"' EXIT; \
	printf '#include "razor_abi_4byte_sys.h"\nint main(void){HoloSphereRazorCanary64 c={0};c.sec_count_u32=7u;HoloSphereRazorSnapshot s=holosphere_razor_poll_acquire(&c);HoloSpherePiRollback16 p={1.0f,0.0f,0.0f,0.0f};return (s.sec_count_u32==7u&&p.w==1.0f)?0:1;}\n' > "$$TMP_C"; \
	printf '#include "razor_abi_4byte_sys.h"\nint main(){HoloSphereRazorCanary64 c{};c.sec_count_u32=9u;HoloSphereRazorSnapshot s=holosphere_razor_poll_acquire(&c);return (s.sec_count_u32==9u)?0:1;}\n' > "$$TMP_CPP"; \
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fsanitize=undefined -Iinclude "$$TMP_C" -o $(BUILD)/_razor_abi_c; \
	$(BUILD)/_razor_abi_c; \
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -fsanitize=undefined -Iinclude "$$TMP_CPP" -o $(BUILD)/_razor_abi_cpp; \
	$(BUILD)/_razor_abi_cpp; \
	echo "RAZOR_ABI_C11_CXX17_UBSAN_PASS=1"

verify-holosphere: verify-razor-abi
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/verify_holosphere_receipts.py --check --json; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/test_verify_holosphere_receipts_r5.py; \
	done

check-holosphere-zero-spill: | $(BUILD)
	@set -eu; \
	LOG=$(BUILD)/holosphere_ptxas_spill.log; \
	trap 'rm -f "$$LOG"' EXIT; \
	$(MAKE) -B cuda-holosphere >"$$LOG" 2>&1 || { cat "$$LOG"; exit 1; }; \
	if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG"; then \
		echo "ERROR: non-zero register/stack spill detected in HoloSphere sm_86 kernels" >&2; \
		exit 1; \
	fi; \
	COUNT=$$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG"); \
	if [ "$$COUNT" -ne 23 ]; then \
		echo "ERROR: expected 23 zero-spill HoloSphere kernels, got $$COUNT" >&2; \
		exit 1; \
	fi; \
	echo "HOLOSPHERE_ZERO_SPILL_VERIFIED=1 ($$COUNT kernels)"
