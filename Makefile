CC       ?= gcc
NVCC     ?= /opt/cuda/bin/nvcc
ARCH     ?= sm_86
BUILD    := build

AVX512_FLAGS := -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni -mf16c -mfma
AVX2_FLAGS   := -O3 -mavx2 -mfma -Wall -Wextra -Werror -std=c11
RAPL_FLAGS   := -O3 -Wall -Wextra -Werror
CUDA_FLAGS   := -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v

CPU_BINS := \
	$(BUILD)/libqwen25_avx512_a16_engine.so \
	$(BUILD)/aletheion_avx512_a16_decoder \
	$(BUILD)/s71_proximity_bench \
	$(BUILD)/hf_reference_avx512_a16_blockh4 \
	$(BUILD)/libe8had_simd.so \
	$(BUILD)/libe8_stage1.so \
	$(BUILD)/rapl_landauer_microkernels \
	$(BUILD)/granular_silicon_microprobe \
	$(BUILD)/two_way_silicon_probe

CUDA_BINS := \
	$(BUILD)/libqwen25_sm86_a16_engine.so \
	$(BUILD)/aletheion_sm86_ptx_foundry \
	$(BUILD)/sm86_a16_blockh4_kernel \
	$(BUILD)/sm86_s69_sorted_blockh4_kernel \
	$(BUILD)/rtx3080_live_llm_transcode_kernels \
	$(BUILD)/rtx3080_qwen3b_kernels \
	$(BUILD)/hf_reference_sm86_a16_blockh4 \
	$(BUILD)/rtx3080_siliquant_a16_kernels \
	$(BUILD)/rtx3080_qwen15b_kernels \
	$(BUILD)/libsu2_holonomy_scan_sm86.so

.PHONY: all cpu cuda clean check-zero-spill check-rope-angle check-avx2-smoke

all: cpu cuda

cpu: $(BUILD) $(CPU_BINS)

include cuda/sm86/holosphere_s116_s117.mk

cuda: $(BUILD) $(CUDA_BINS)

$(BUILD):
	@mkdir -p $(BUILD)

# --- Intel AVX-512 (VBMI / VNNI / VPOPCNTDQ) Targets ---

$(BUILD)/libqwen25_avx512_a16_engine.so: cpu/avx512/qwen25_avx512_a16_engine.c | $(BUILD)
	$(CC) $(AVX512_FLAGS) -fopenmp -shared -fPIC $< -lm -o $@

$(BUILD)/aletheion_avx512_a16_decoder: cpu/avx512/aletheion_avx512_a16_decoder.c | $(BUILD)
	$(CC) $(AVX512_FLAGS) -fopenmp $< -lm -o $@

$(BUILD)/s71_proximity_bench: cpu/avx512/s71_proximity_bench.c | $(BUILD)
	$(CC) $(AVX512_FLAGS) -pthread $< -lm -o $@

$(BUILD)/hf_reference_avx512_a16_blockh4: cpu/avx512/hf_reference_avx512_a16_blockh4.c | $(BUILD)
	$(CC) -O3 -march=native $< -lm -o $@

# --- Intel AVX2 / FMA Zero-Division E_8 Lattice Targets ---

$(BUILD)/libe8had_simd.so: cpu/avx2/e8had_simd_bench.c | $(BUILD)
	$(CC) $(AVX2_FLAGS) -shared -fPIC -pthread $< -lm -o $@

$(BUILD)/libe8_stage1.so: cpu/avx2/e8_stage1_bench.c | $(BUILD)
	$(CC) $(AVX2_FLAGS) -shared -fPIC -pthread $< -lm -o $@

# --- Linux intel-rapl / Landauer / RDTSC Silicon Microprobes ---

$(BUILD)/rapl_landauer_microkernels: cpu/rapl/rapl_landauer_microkernels.c | $(BUILD)
	$(CC) $(RAPL_FLAGS) -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma -pthread $< -lm -o $@

$(BUILD)/granular_silicon_microprobe: cpu/rapl/granular_silicon_microprobe.c | $(BUILD)
	$(CC) $(RAPL_FLAGS) -pthread $< -lm -o $@

$(BUILD)/two_way_silicon_probe: cpu/rapl/two_way_silicon_probe.c | $(BUILD)
	$(CC) $(RAPL_FLAGS) -pthread $< -lm -o $@

# --- NVIDIA Ampere sm_86 Bare-Metal PTX/SASS Targets ---

$(BUILD)/libqwen25_sm86_a16_engine.so: cuda/sm86/qwen25_sm86_a16_engine.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC $< -o $@

$(BUILD)/aletheion_sm86_ptx_foundry: cuda/sm86/aletheion_sm86_ptx_foundry.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD)/sm86_a16_blockh4_kernel: cuda/sm86/sm86_a16_blockh4_kernel.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -ldl -o $@

$(BUILD)/sm86_s69_sorted_blockh4_kernel: cuda/sm86/sm86_s69_sorted_blockh4_kernel.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -ldl -o $@

$(BUILD)/rtx3080_live_llm_transcode_kernels: cuda/sm86/rtx3080_live_llm_transcode_kernels.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD)/rtx3080_qwen3b_kernels: cuda/sm86/rtx3080_qwen3b_kernels.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -ldl -o $@

$(BUILD)/hf_reference_sm86_a16_blockh4: cuda/sm86/hf_reference_sm86_a16_blockh4.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD)/rtx3080_siliquant_a16_kernels: cuda/sm86/rtx3080_siliquant_a16_kernels.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD)/rtx3080_qwen15b_kernels: cuda/sm86/rtx3080_qwen15b_kernels.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD)/libsu2_holonomy_scan_sm86.so: cuda/sm86/su2_holonomy_scan_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC $< -o $@

# Expected repo-wide sm_86 kernel instantiations parsed from ptxas -v output.
# Counted per file on CUDA 13.2.86 / 13.0.88 (see README): qwen25 engine 16,
# aletheion foundry 15, sm86_a16_blockh4 7, sm86_s69 9, live_llm 10, qwen3b 10,
# hf_reference 2, siliquant_a16 24, qwen15b 10, su2_holonomy 27,
# holosphere_megakernel 10, holosphere_h256_chunk32 11, holosphere_v3_frontier 2.
EXPECTED_KERNELS ?= 153

# Fail-closed zero-spill gate: requires that the number of ptxas entries parsed
# equals EXPECTED_KERNELS (> 0) and that EVERY entry reports 0 bytes stack frame,
# 0 spill stores, 0 spill loads. With NVCC=true (or any compiler that emits no
# ptxas diagnostics) the parsed count is 0 != EXPECTED_KERNELS, so the gate fails.
check-zero-spill:
	@rm -rf ./build && mkdir -p ./build
	@LOG=./build/ptxas_spill.log && \
	$(MAKE) cuda >"$$LOG" 2>&1 || { cat "$$LOG"; echo "ERROR: cuda build failed" >&2; exit 1; }; \
	ENTRIES=$$(grep -c 'bytes stack frame' "$$LOG" || true); \
	ZERO=$$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG" || true); \
	BAD=$$(grep -cE '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG" || true); \
	echo "ptxas entries parsed: $$ENTRIES (expected $(EXPECTED_KERNELS)), zero-spill: $$ZERO, non-zero spill: $$BAD"; \
	if [ "$$ENTRIES" -ne "$(EXPECTED_KERNELS)" ] || [ "$$ZERO" -ne "$(EXPECTED_KERNELS)" ] || [ "$$BAD" -ne 0 ]; then \
		grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG" || true; \
		echo "ERROR: zero-spill gate FAILED (fail-closed): entries=$$ENTRIES zero=$$ZERO bad=$$BAD expected=$(EXPECTED_KERNELS)" >&2; \
		exit 1; \
	fi; \
	echo "ZERO_SPILL_VERIFIED=1 ($$ZERO kernel instantiations, all 0 B stack frame / 0 B spill)"; \
	rm -f "$$LOG"

# Host-side accuracy check for the rope_kv_append_kernel RoPE angle reduction
# (old cosf/sinf vs new fmod+__sincosf, against a double reference).
check-rope-angle: $(BUILD) verify/rope_angle_check.c | $(BUILD)
	$(CC) -O2 -std=c11 -Wall -Wextra -Werror verify/rope_angle_check.c -lm -o $(BUILD)/rope_angle_check
	$(BUILD)/rope_angle_check

# Functional smoke tests for the AVX2/FMA libraries (cpu/avx2/tests/):
# init, unrank/decode equivalence, GEMV equivalence vs an independent double
# reference, rejected-rank / unaligned-cols / zero-point edge cases. Each test
# is run against the AVX2 .so and a scalar (-mno-avx2) build of the same source;
# the DECODE_DIGEST lines must match (AVX2 and scalar decode byte-identically).
AVX2_TEST_FLAGS := -O2 -std=c11 -Wall -Wextra -Werror -pthread
SCALAR_LIB_FLAGS := -O2 -mno-avx2 -mno-fma -std=c11 -shared -fPIC -pthread

$(BUILD)/test_e8had_smoke: cpu/avx2/tests/test_e8had_smoke.c cpu/avx2/tests/e8_tables_gen.h | $(BUILD)
	$(CC) $(AVX2_TEST_FLAGS) $< -ldl -lm -o $@

$(BUILD)/test_e8_stage1_smoke: cpu/avx2/tests/test_e8_stage1_smoke.c cpu/avx2/tests/e8_tables_gen.h | $(BUILD)
	$(CC) $(AVX2_TEST_FLAGS) $< -ldl -lm -o $@

$(BUILD)/libe8had_simd_scalar.so: cpu/avx2/e8had_simd_bench.c | $(BUILD)
	$(CC) $(SCALAR_LIB_FLAGS) $< -lm -o $@

$(BUILD)/libe8_stage1_scalar.so: cpu/avx2/e8_stage1_bench.c | $(BUILD)
	$(CC) $(SCALAR_LIB_FLAGS) $< -lm -o $@

check-avx2-smoke: $(BUILD)/libe8had_simd.so $(BUILD)/libe8_stage1.so \
                   $(BUILD)/libe8had_simd_scalar.so $(BUILD)/libe8_stage1_scalar.so \
                   $(BUILD)/test_e8had_smoke $(BUILD)/test_e8_stage1_smoke
	$(BUILD)/test_e8had_smoke $(BUILD)/libe8had_simd.so > $(BUILD)/smoke_e8had_avx2.txt
	$(BUILD)/test_e8had_smoke $(BUILD)/libe8had_simd_scalar.so > $(BUILD)/smoke_e8had_scalar.txt
	$(BUILD)/test_e8_stage1_smoke $(BUILD)/libe8_stage1.so > $(BUILD)/smoke_e8_stage1_avx2.txt
	$(BUILD)/test_e8_stage1_smoke $(BUILD)/libe8_stage1_scalar.so > $(BUILD)/smoke_e8_stage1_scalar.txt
	grep -q SMOKE_PASS $(BUILD)/smoke_e8had_avx2.txt
	grep -q SMOKE_PASS $(BUILD)/smoke_e8had_scalar.txt
	grep -q SMOKE_PASS $(BUILD)/smoke_e8_stage1_avx2.txt
	grep -q SMOKE_PASS $(BUILD)/smoke_e8_stage1_scalar.txt
	@D1=$$(grep '^DECODE_DIGEST=' $(BUILD)/smoke_e8had_avx2.txt); \
	 D2=$$(grep '^DECODE_DIGEST=' $(BUILD)/smoke_e8had_scalar.txt); \
	 test "$$D1" = "$$D2" || { echo "ERROR: e8had AVX2 vs scalar decode digest mismatch: $$D1 vs $$D2" >&2; exit 1; }; \
	 echo "e8had decode digest AVX2==scalar: $$D1"
	@D1=$$(grep '^DECODE_DIGEST_M16=' $(BUILD)/smoke_e8_stage1_avx2.txt); \
	 D2=$$(grep '^DECODE_DIGEST_M16=' $(BUILD)/smoke_e8_stage1_scalar.txt); \
	 D3=$$(grep '^DECODE_DIGEST_M65=' $(BUILD)/smoke_e8_stage1_avx2.txt); \
	 D4=$$(grep '^DECODE_DIGEST_M65=' $(BUILD)/smoke_e8_stage1_scalar.txt); \
	 test "$$D1" = "$$D2" -a "$$D3" = "$$D4" || { echo "ERROR: e8_stage1 AVX2 vs scalar digest mismatch" >&2; exit 1; }; \
	 echo "e8_stage1 decode digests AVX2==scalar: $$D1 $$D3"
	@echo "AVX2_SMOKE_PASS=1"

clean:
	rm -rf ./build

include cpu/avx512/holosphere_v3_avx512.mk
