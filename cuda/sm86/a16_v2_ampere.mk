# SPDX-License-Identifier: Apache-2.0
# V2 Zero-Spill Ampere sm_86 + AVX-512 B*=2 SWAR127 Kernel & Verification Targets
# Included cleanly by top-level Makefile.

PYTHON ?= python3

CUDA_V2_NVCC_FLAGS := -O3 -std=c++17 -arch=$(ARCH) -Iinclude --ptxas-options=-v
CPU_V2_AVX512_FLAGS := -O3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni -Wall -Wextra -Werror
CPU_V2_PORTABLE_FLAGS := -O3 -std=c11 -Wall -Wextra -Werror

CPU_V2_BINS := \
	$(BUILD)/liba16_v2_avx512.so \
	$(BUILD)/a16_v2_avx512_swar \
	$(BUILD)/liba16_v2_portable.so \
	$(BUILD)/a16_v2_portable_swar

CUDA_V2_BINS := \
	$(BUILD)/a16_v2_ampere_kernels.o \
	$(BUILD)/liba16_v2_ampere.so \
	$(BUILD)/a16_v2_ampere_bench

CPU_BINS += $(CPU_V2_BINS)
CUDA_BINS += $(BUILD)/liba16_v2_ampere.so $(BUILD)/a16_v2_ampere_bench

.PHONY: cpu-v2 cuda-v2 test-cpu verify-v2-sass

cpu-v2: $(BUILD) $(CPU_V2_BINS)

cuda-v2: $(BUILD) $(CUDA_V2_BINS)

$(BUILD)/liba16_v2_avx512.so: cpu/avx512/a16_v2_avx512_swar.c include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(CC) $(CPU_V2_AVX512_FLAGS) -DSILIKERNELS_V2_LIB -pthread -shared -fPIC -Iinclude $< -lm -o $@

$(BUILD)/a16_v2_avx512_swar: cpu/avx512/a16_v2_avx512_swar.c include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(CC) $(CPU_V2_AVX512_FLAGS) -pthread -Iinclude $< -lm -o $@

$(BUILD)/liba16_v2_portable.so: cpu/avx512/a16_v2_avx512_swar.c include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(CC) $(CPU_V2_PORTABLE_FLAGS) -DSILIKERNELS_V2_LIB -pthread -shared -fPIC -Iinclude $< -lm -o $@

$(BUILD)/a16_v2_portable_swar: cpu/avx512/a16_v2_avx512_swar.c include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(CC) $(CPU_V2_PORTABLE_FLAGS) -pthread -Iinclude $< -lm -o $@

$(BUILD)/a16_v2_ampere_kernels.o: cuda/sm86/a16_v2_ampere_kernels.cu cuda/sm86/a16_v2_ampere_kernels.cuh include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -Xcompiler -fPIC,-Wall,-Wextra -c $< -o $@

$(BUILD)/liba16_v2_ampere.so: cuda/sm86/a16_v2_ampere_bench.cu cuda/sm86/a16_v2_ampere_kernels.cuh include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -DSILIKERNELS_V2_SHARED_LIB --shared -Xcompiler -fPIC,-Wall,-Wextra $< -o $@

$(BUILD)/a16_v2_ampere_bench: cuda/sm86/a16_v2_ampere_bench.cu cuda/sm86/a16_v2_ampere_kernels.cuh include/silikernels_v2_contracts.h include/razor_abi_4byte_sys.h | $(BUILD)
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -Xcompiler -Wall,-Wextra $< -o $@

test-cpu: cpu-v2 verify-razor-abi
	@set -eu; \
	CUDA_VISIBLE_DEVICES="" ./$(BUILD)/a16_v2_portable_swar --self-test; \
	if grep -qE '\bavx512f\b' /proc/cpuinfo 2>/dev/null; then \
		CUDA_VISIBLE_DEVICES="" ./$(BUILD)/a16_v2_avx512_swar --self-test; \
	fi; \
	out1="$$(mktemp)"; out2="$$(mktemp)"; out3="$$(mktemp)"; \
	u1="$$(mktemp)"; u2="$$(mktemp)"; u3="$$(mktemp)"; \
	trap 'rm -f "$${out1:?}" "$${out2:?}" "$${out3:?}" "$${u1:?}" "$${u2:?}" "$${u3:?}"' EXIT; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B verify/verify_v2_sass_and_cpu.py --cpu-only > "$$out1"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -O -B verify/verify_v2_sass_and_cpu.py --cpu-only > "$$out2"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -W error verify/verify_v2_sass_and_cpu.py --cpu-only > "$$out3"; \
	cmp -s "$$out1" "$$out2"; cmp -s "$$out1" "$$out3"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_silikernels_v2_cpu.py > "$$u1"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -O -B tests/test_silikernels_v2_cpu.py > "$$u2"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -W error tests/test_silikernels_v2_cpu.py > "$$u3"; \
	cmp -s "$$u1" "$$u2"; cmp -s "$$u1" "$$u3"; \
	cat "$$out1"; \
	cat "$$u1"; \
	echo "SILIKERNELS_V2_TEST_CPU_3X_PASS=1"

verify-v2-sass: cpu-v2 | $(BUILD)
	@set -eu; \
	LOG_O=$(BUILD)/_v2_ptxas_o.log; \
	LOG_SO=$(BUILD)/_v2_ptxas_so.log; \
	LOG_BIN=$(BUILD)/_v2_ptxas_bench.log; \
	SASS_O=$(BUILD)/_v2_kernels_o.sass; \
	SASS_SO=$(BUILD)/_v2_kernels_so.sass; \
	SASS_BIN=$(BUILD)/_v2_kernels_bench.sass; \
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -Xcompiler -fPIC,-Wall,-Wextra -c cuda/sm86/a16_v2_ampere_kernels.cu -o $(BUILD)/a16_v2_ampere_kernels.o >"$$LOG_O" 2>&1 || { cat "$$LOG_O"; exit 1; }; \
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -DSILIKERNELS_V2_SHARED_LIB --shared -Xcompiler -fPIC,-Wall,-Wextra cuda/sm86/a16_v2_ampere_bench.cu -o $(BUILD)/liba16_v2_ampere.so >"$$LOG_SO" 2>&1 || { cat "$$LOG_SO"; exit 1; }; \
	$(NVCC) $(CUDA_V2_NVCC_FLAGS) -Xcompiler -Wall,-Wextra cuda/sm86/a16_v2_ampere_bench.cu -o $(BUILD)/a16_v2_ampere_bench >"$$LOG_BIN" 2>&1 || { cat "$$LOG_BIN"; exit 1; }; \
	cuobjdump --dump-sass $(BUILD)/a16_v2_ampere_kernels.o > "$$SASS_O"; \
	cuobjdump --dump-sass $(BUILD)/liba16_v2_ampere.so > "$$SASS_SO"; \
	cuobjdump --dump-sass $(BUILD)/a16_v2_ampere_bench > "$$SASS_BIN"; \
	CUDA_VISIBLE_DEVICES="" ./$(BUILD)/a16_v2_ampere_bench --cpu-selftest; \
	out1="$$(mktemp)"; out2="$$(mktemp)"; out3="$$(mktemp)"; \
	trap 'rm -f "$${out1:?}" "$${out2:?}" "$${out3:?}"' EXIT; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B verify/verify_v2_sass_and_cpu.py --verify-sass > "$$out1"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -O -B verify/verify_v2_sass_and_cpu.py --verify-sass > "$$out2"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -W error verify/verify_v2_sass_and_cpu.py --verify-sass > "$$out3"; \
	cmp -s "$$out1" "$$out2"; cmp -s "$$out1" "$$out3"; \
	cat "$$out1"; \
	echo "SILIKERNELS_V2_SASS_3X_PASS=1"
