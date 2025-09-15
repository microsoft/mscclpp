CUDA_HOME ?= /usr/local/cuda
ROCM_HOME ?= /opt/rocm

# Check if nvcc exists, otherwise use hipcc
ifeq ($(shell which $(CUDA_HOME)/bin/nvcc 2>/dev/null),)
    COMPILER := $(ROCM_HOME)/bin/hipcc
    ARCH_FLAG := -D__HIP_PLATFORM_AMD__=1
else
    COMPILER := $(CUDA_HOME)/bin/nvcc
    ARCH_FLAG := -arch=native
endif

TARGET = customized_allgather
SRC = customized_allgather.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) -o $@ $< -lmscclpp -lnccl

clean:
	rm -f $(TARGET)
