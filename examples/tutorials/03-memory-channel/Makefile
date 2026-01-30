CUDA_HOME ?= /usr/local/cuda
ROCM_HOME ?= /opt/rocm

NVML_FLAG := 
# Check if nvcc exists, otherwise use hipcc
ifeq ($(shell which $(CUDA_HOME)/bin/nvcc 2>/dev/null),)
    COMPILER := $(ROCM_HOME)/bin/hipcc
    ARCH_FLAG := -D__HIP_PLATFORM_AMD__=1
else
    COMPILER := $(CUDA_HOME)/bin/nvcc
    ARCH_FLAG := -arch=native
endif

TARGET = bidir_memory_channel
SRC = bidir_memory_channel.cu
COMPILE_FLAG := -I$(PWD)/../../../include/ -I$(PWD)/../../../build/include -L$(PWD)/../../../build/lib/

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) $(COMPILE_FLAG) -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
