CUDA_HOME ?= /usr/local/cuda
ROCM_HOME ?= /opt/rocm
MSCCLPP_HOME ?= $(CURDIR)/../../..

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
INC = -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_HOME)/apps/nccl/include
LIB = -L$(MSCCLPP_HOME)/build

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) $(INC) $(LIB) -o $@ $< -lmscclpp -lnccl

clean:
	rm -f $(TARGET)
