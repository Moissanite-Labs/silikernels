CC       ?= gcc
NVCC     ?= /opt/cuda/bin/nvcc
ARCH     ?= sm_86
BUILD    := build

AVX512_FLAGS := -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni
CUDA_FLAGS   := -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v

CPU_BINS := \
	$(BUILD)/libqwen25_avx512_a16_engine.so \
	$(BUILD)/aletheion_avx512_a16_decoder \
	$(BUILD)/s71_proximity_bench \
	$(BUILD)/hf_reference_avx512_a16_blockh4

CUDA_BINS := \
	$(BUILD)/libqwen25_sm86_a16_engine.so \
	$(BUILD)/aletheion_sm86_ptx_foundry \
	$(BUILD)/sm86_a16_blockh4_kernel \
	$(BUILD)/sm86_s69_sorted_blockh4_kernel \
	$(BUILD)/rtx3080_live_llm_transcode_kernels \
	$(BUILD)/rtx3080_qwen3b_kernels \
	$(BUILD)/hf_reference_sm86_a16_blockh4 \
	$(BUILD)/libsu2_holonomy_scan_sm86.so

.PHONY: all cpu cuda clean check-zero-spill test-cpu verify-v2-sass

all: cpu cuda

cpu: $(BUILD) $(CPU_BINS)

include cuda/sm86/holosphere_s116_s117.mk
include cuda/sm86/a16_v2_ampere.mk

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

$(BUILD)/libsu2_holonomy_scan_sm86.so: cuda/sm86/su2_holonomy_scan_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC $< -o $@

clean:
	rm -rf ./build

include cpu/avx512/holosphere_v3_avx512.mk
