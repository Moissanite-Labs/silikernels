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
	$(BUILD)/rtx3080_qwen15b_kernels

.PHONY: all cpu cuda clean check-zero-spill

all: cpu cuda

cpu: $(BUILD) $(CPU_BINS)

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

check-zero-spill:
	@rm -rf ./build && mkdir -p ./build
	@LOG=./build/ptxas_spill.log && \
	$(MAKE) cuda >"$$LOG" 2>&1 || { cat "$$LOG"; exit 1; }; \
	if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG"; then \
		echo "ERROR: non-zero register/stack spill detected" >&2; \
		exit 1; \
	fi; \
	echo "ZERO_SPILL_VERIFIED=1 ($$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG") kernel instantiations)"; \
	rm -f "$$LOG"

clean:
	rm -rf ./build
