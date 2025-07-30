CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET = gpu_ping_pong_mp
SRC = gpu_ping_pong_mp.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -arch=native -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
