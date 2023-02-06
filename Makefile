DEBUG ?= 0
VERBOSE ?= 1
TRACE ?= 0

######## CUDA
CUDA_HOME ?= /usr/local/cuda
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
# You should define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
CUDA8_GENCODE = -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61
ifeq ($(shell test "0$(CUDA_MAJOR)" -lt 12; echo $$?),0)
# SM35 is deprecated from CUDA12.0 onwards
CUDA8_GENCODE += -gencode=arch=compute_35,code=sm_35
endif
CUDA9_GENCODE = -gencode=arch=compute_70,code=sm_70
CUDA11_GENCODE = -gencode=arch=compute_80,code=sm_80
CUDA12_GENCODE = -gencode=arch=compute_90,code=sm_90

CUDA8_PTX     = -gencode=arch=compute_61,code=compute_61
CUDA9_PTX     = -gencode=arch=compute_70,code=compute_70
CUDA11_PTX    = -gencode=arch=compute_80,code=compute_80
CUDA12_PTX    = -gencode=arch=compute_90,code=compute_90

######## CXX/NVCC
CXX := g++
NVTX ?= 1

ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 11 -a "0$(CUDA_MINOR)" -ge 8 -o "0$(CUDA_MAJOR)" -gt 11; echo $$?),0)
# Include Hopper support if we're using CUDA11.8 or above
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA11_GENCODE) $(CUDA12_GENCODE) $(CUDA12_PTX)
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA11_GENCODE) $(CUDA11_PTX)
# Include Volta support if we're using CUDA9 or above
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 9; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA9_PTX)
else
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA8_PTX)
endif
$(info NVCC_GENCODE is ${NVCC_GENCODE})

CXXFLAGS   := -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -fPIC -fvisibility=hidden \
              -Wall -Wno-unused-function -Wno-sign-compare -std=c++14 -Wvla \
              -I $(CUDA_INC) \
              $(CXXFLAGS)

ifneq ($(TRACE), 0)
CXXFLAGS  += -DENABLE_TRACE
endif
# Maxrregcount needs to be set accordingly to MSCCLPP_MAX_NTHREADS (otherwise it will cause kernel launch errors)
# 512 : 120, 640 : 96, 768 : 80, 1024 : 60
# We would not have to set this if we used __launch_bounds__, but this only works on kernels, not on functions.
NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all
# Use addprefix so that we can specify more than one path
NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
CXXFLAGS  += -Wall -Wextra
else
.SILENT:
endif

ifeq ($(NVTX), 0)
CXXFLAGS  += -DNVTX_DISABLE
endif

#### MSCCL++
BUILDDIR ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILDDIR))

BUILDSRCS := init.cc debug.cc bootstrap.cc utils.cc param.cc socket.cc proxy.cc
BUILDOBJS := $(patsubst %.cc,$(ABSBUILDDIR)/src/%.o,$(BUILDSRCS))

TESTSSRCS := init_test.cc
TESTSOBJS := $(patsubst %.cc,$(ABSBUILDDIR)/src/%.o,$(TESTSSRCS))
TESTBINS  := $(patsubst %.cc,$(ABSBUILDDIR)/src/%,$(TESTSSRCS))

INCLUDE := -Isrc -Isrc/include

.PHONY: all build tests clean

all: build tests

build: $(BUILDOBJS)
tests: $(TESTBINS)

$(ABSBUILDDIR)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) -o $@ $(INCLUDE) $(CXXFLAGS) -c $<

$(TESTBINS): %: %.o $(BUILDOBJS)
	@mkdir -p $(@D)
	$(NVCC) -o $@ $^ $(NVLDFLAGS)

clean:
	rm -rf $(ABSBUILDDIR)
