# SPDX-License-Identifier: Apache-2.0
# Bare-metal Intel AVX-512 VNNI L3-Resident HoloSphere-V3 15M CPU Engine (Track S125)
# Included cleanly at the bottom of top-level Makefile (zero conflicts with PR #1 / PR #4).

PYTHON ?= python3

.PHONY: cpu-holosphere-avx512 verify-holosphere-avx512

cpu-holosphere-avx512: $(BUILD)/holosphere_v3_avx512

$(BUILD)/holosphere_v3_avx512: cpu/avx512/holosphere_v3_avx512.c | $(BUILD)
	$(CC) -O3 -march=native -std=c11 -Wall -Wextra -Werror -pthread $< -lm -o $@

verify-holosphere-avx512:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/verify_s125_avx512_l3.py --check --json; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags verify/test_s125_avx512_l3_r5.py; \
	done
