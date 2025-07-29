CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET = gpu_ping_pong
SRC = gpu_ping_pong.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -arch=native -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
