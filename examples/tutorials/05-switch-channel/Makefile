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
    NVML_FLAG := -lnvidia-ml
    MPI_FLAG := -L/opt/hpcx-v2.24.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/lib/ -I/opt/hpcx-v2.24.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/include/ -lmpi
endif

TARGET = bidir_switch_channel
SRC = bidir_switch_channel.cu
COMPILE_FLAG := -I$(PWD)/../../../include/ -I$(PWD)/../../../build/include -L$(PWD)/../../../build/lib/

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) $(COMPILE_FLAG) $(NVML_FLAG) $(MPI_FLAG) -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
