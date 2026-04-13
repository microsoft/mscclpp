CUDA_HOME ?= /usr/local/cuda

COMPILER := $(CUDA_HOME)/bin/nvcc
ARCH_FLAG := -arch=native

TARGET = bidir_switch_channel
SRC = bidir_switch_channel.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)
