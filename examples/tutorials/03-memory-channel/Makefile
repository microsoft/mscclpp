CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET = perf_memory_channel
SRC = perf_memory_channel.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -arch=native -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
