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

TARGET = peer_access_test
SRC = peer_access_test.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) -o $@ $<

clean:
	rm -f $(TARGET)
