# ==========================================
# CUDA
# ==========================================
CUDA_HOME ?= /usr/local/cuda-12
NVCC      := $(CUDA_HOME)/bin/nvcc
CUDA_INC  := -I$(CUDA_HOME)/include
CUDA_LIB  := -L$(CUDA_HOME)/lib64
NVCC_ARCH ?= -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_86,code=compute_86

# ==========================================
# Compiler
# ==========================================
CXX := g++
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# ==========================================
# Debug toggle
# Usage: make DEBUG=1
# ==========================================
DEBUG ?= 0
USE_DOUBLE ?= 1
LINEINFO ?= 0
BUILD_DIR ?= build
# ==========================================
# libpll
# ==========================================
PLL_INC_DIR ?= /usr/local/include
PLL_LIB_DIR ?= /usr/local/lib
USE_EPA_VENDOR_PLL ?= 1
EPA_PLL_ROOT ?= $(abspath $(REPO_ROOT)/../../DIPPER_MLIPPER/DIPPER/reference/epa-ng_DBG)
EPA_PLL_INC_DIR ?= $(EPA_PLL_ROOT)/libs/pll-modules/libs/libpll/src
EPA_PLL_LIB_DIR ?= $(EPA_PLL_ROOT)/build/libs/pll-modules/libs/libpll/src
PLL_LINK_FLAGS := -L$(PLL_LIB_DIR) -lpll

# ==========================================
# System LAPACK / BLAS
# ==========================================
LAPACK_LIBS := -L/usr/lib/x86_64-linux-gnu \
               -llapack -lblas -lm

# ==========================================
# Include paths
# ==========================================
CLI11_INC_DIR ?= third_party/CLI11/include

INCLUDES := -I. \
            -Itree_generation \
            -Ipmatrix \
            -Ipartial_CUDA \
            -Icore_CUDA \
            -I$(CLI11_INC_DIR) \
            -I$(PLL_INC_DIR) \
            -I$(CUDA_HOME)/include \
            -I$(CONDA_PREFIX)/include

# ==========================================
# Flags
# ==========================================
CXXFLAGS := -O3 -std=c++17 -fno-omit-frame-pointer $(INCLUDES)

NVCCFLAGS := -O3 -std=c++17 -ccbin $(CXX) \
             -Xcompiler -fno-omit-frame-pointer -rdc=true \
             $(INCLUDES) $(CUDA_INC) $(NVCC_ARCH)

ifneq ($(and $(wildcard $(EPA_PLL_INC_DIR)/pll.h),$(wildcard $(EPA_PLL_LIB_DIR)/libpll.a)),)
HAVE_EPA_VENDOR_PLL := 1
else
HAVE_EPA_VENDOR_PLL := 0
endif

ifeq ($(USE_EPA_VENDOR_PLL),1)
ifeq ($(HAVE_EPA_VENDOR_PLL),1)
    INCLUDES := -I. \
                -Itree_generation \
                -Ipmatrix \
                -Ipartial_CUDA \
                -Icore_CUDA \
                -I$(CLI11_INC_DIR) \
                -I$(EPA_PLL_INC_DIR) \
                -I$(CUDA_HOME)/include \
                -I$(CONDA_PREFIX)/include
    CXXFLAGS := -O3 -std=c++17 -fno-omit-frame-pointer $(INCLUDES)
    NVCCFLAGS := -O3 -std=c++17 -ccbin $(CXX) \
                 -Xcompiler -fno-omit-frame-pointer -rdc=true \
                 $(INCLUDES) $(CUDA_INC) $(NVCC_ARCH)
    CXXFLAGS  += -DMLIPPER_USE_VENDOR_PLL
    NVCCFLAGS += -DMLIPPER_USE_VENDOR_PLL
    PLL_LIB_DIR := $(EPA_PLL_LIB_DIR)
    PLL_LINK_FLAGS := $(EPA_PLL_LIB_DIR)/libpll.a
else
    $(warning USE_EPA_VENDOR_PLL=1 but vendor libpll was not found; falling back to system libpll)
endif
endif

ifeq ($(DEBUG),1)
    CXXFLAGS  += -g -O0
    NVCCFLAGS += -G -O0 -lineinfo
endif

ifeq ($(LINEINFO),1)
    NVCCFLAGS += -lineinfo
endif

ifeq ($(USE_DOUBLE),1)
    CXXFLAGS  += -DMLIPPER_USE_DOUBLE
    NVCCFLAGS += -DMLIPPER_USE_DOUBLE
endif

# ==========================================
# Sources
# ==========================================
CPU_SRCS  := pmatrix/pmat.cpp \
             tree_generation/tree_generation.cpp \
             tree_generation/parse_file.cpp \
             tree_generation/tree.cpp \
             tree_generation/seq_preproc.cpp

CUDA_SRCS := tree_generation/tree_generation_device.cu \
             tree_generation/tree_placement.cu \
             tree_generation/derivative.cu \
             pmatrix/pmat_gpu.cu \
             partial_CUDA/partial_likelihood.cu \
             tree_generation/root_likelihood.cu

CPU_OBJS  := $(CPU_SRCS:.cpp=.o)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
CPU_OBJS  := $(patsubst %,$(BUILD_DIR)/%,$(CPU_OBJS))
CUDA_OBJS := $(patsubst %,$(BUILD_DIR)/%,$(CUDA_OBJS))
OBJS      := $(CPU_OBJS) $(CUDA_OBJS)

TARGET := MLIPPER
RUN_ARGS ?=
.DEFAULT_GOAL := $(TARGET)

# ==========================================
# Rules
# ==========================================
# Build-config stamp: forces recompilation when toggling Makefile options (e.g. USE_DOUBLE/DEBUG)
BUILD_CONFIG := $(BUILD_DIR)/.build_config

FORCE:
.PHONY: FORCE

$(BUILD_CONFIG): FORCE
	@mkdir -p $(dir $(BUILD_CONFIG))
	@printf "DEBUG=%s\nUSE_DOUBLE=%s\nLINEINFO=%s\nCUDA_HOME=%s\nUSE_EPA_VENDOR_PLL=%s\nHAVE_EPA_VENDOR_PLL=%s\nPLL_LIB_DIR=%s\nEPA_PLL_ROOT=%s\n" "$(DEBUG)" "$(USE_DOUBLE)" "$(LINEINFO)" "$(CUDA_HOME)" "$(USE_EPA_VENDOR_PLL)" "$(HAVE_EPA_VENDOR_PLL)" "$(PLL_LIB_DIR)" "$(EPA_PLL_ROOT)" > $(BUILD_CONFIG).tmp
	@cmp -s $(BUILD_CONFIG).tmp $(BUILD_CONFIG) 2>/dev/null || mv $(BUILD_CONFIG).tmp $(BUILD_CONFIG)
	@rm -f $(BUILD_CONFIG).tmp

# ---- C++ -> .o ----
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- CUDA -> .o ----
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ---- Link ----
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) -ccbin $(CXX) $(NVCC_ARCH) -o $@ $^ \
		$(PLL_LINK_FLAGS) \
		$(CUDA_LIB) -lcudart -lcuda \
		-ltbb \
		$(LAPACK_LIBS)

# Crude dependency tracking: force rebuild when core headers change.
$(OBJS): $(BUILD_CONFIG) tree_generation/tree.hpp tree_generation/precision.hpp mlipper_util.h tree_generation/derivative.cuh tree_generation/root_likelihood.cuh tree_generation/tree_placement.cuh
debug:
	$(MAKE) clean
	$(MAKE) DEBUG=1 $(TARGET)

run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

cuda-gdb: $(TARGET)
	cuda-gdb --args ./$(TARGET) $(RUN_ARGS)

clean:
	rm -rf $(BUILD_DIR)

double:
	$(MAKE) clean
	$(MAKE) USE_DOUBLE=1 $(TARGET)

float:
	$(MAKE) clean
	$(MAKE) USE_DOUBLE=0 $(TARGET)

.PHONY: clean debug run cuda-gdb double float
