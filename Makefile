######## VERSION
MSCCLPP_MAJOR := 0
MSCCLPP_MINOR := 1

######## COMPILE OPTIONS
DEBUG ?= 0
VERBOSE ?= 1
TRACE ?= 0

######## CUDA
CUDA_HOME ?= /usr/local/cuda
CUDA_LIB ?= $(CUDA_HOME)/lib64
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
NVLDFLAGS  := -L$(CUDA_LIB) -lcudart -lrt

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

#### MPI (only for test code)
MPI_HOME    ?= /usr/local/mpi
MPI_INC     := -I$(MPI_HOME)/include
MPI_LDFLAGS := -L$(MPI_HOME)/lib -lmpi

#### MSCCL++
BUILDDIR ?= $(abspath ./build)
INCDIR := include
LIBDIR := lib
OBJDIR := obj
BINDIR := bin

LIBSRCS := $(addprefix src/,debug.cc utils.cc param.cc)
LIBSRCS += $(addprefix src/bootstrap/,init.cc bootstrap.cc socket.cc proxy.cc)
LIBOBJS := $(patsubst %.cc,%.o,$(LIBSRCS))
LIBOBJTARGETS := $(LIBOBJS:%=$(BUILDDIR)/$(OBJDIR)/%)

INCEXPORTS := mscclpp.h mscclpp_net.h
INCTARGETS := $(INCEXPORTS:%=$(BUILDDIR)/$(INCDIR)/%)

LIBNAME   := libmscclpp.so
LIBSONAME := $(LIBNAME).$(MSCCLPP_MAJOR)
LIBTARGET := $(BUILDDIR)/$(LIBDIR)/$(LIBNAME).$(MSCCLPP_MAJOR).$(MSCCLPP_MINOR)

TESTSDIR  := tests
TESTSSRCS := $(addprefix $(TESTSDIR)/,bootstrap_test.cc bootstrap_test_mpi.cc p2p_test_mpi.cu)
TESTSOBJS := $(patsubst %.cc,%.o,$(TESTSSRCS)) $(patsubst %.cu,%.o,$(TESTSSRCS))
TESTSOBJTARGETS := $(TESTSOBJS:%=$(BUILDDIR)/$(OBJDIR)/%)
TESTSBINS       := $(patsubst %.o,$(BUILDDIR)/$(BINDIR)/%,$(TESTSOBJS))

INCLUDE := -Isrc -Isrc/include

.PHONY: all build lib tests clean

all: build

build: lib tests

lib: $(LIBOBJTARGETS) $(INCTARGETS) $(LIBTARGET)

tests: $(TESTSBINS)

# Compile libobjs
$(BUILDDIR)/$(OBJDIR)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) -o $@ $(INCLUDE) $(CXXFLAGS) -c $<

$(BUILDDIR)/$(INCDIR)/%.h: src/$(INCDIR)/%.h
	@mkdir -p $(@D)
	cp $< $@

$(LIBTARGET): $(LIBOBJTARGETS)
	@mkdir -p $(@D)
	$(CXX) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)
	ln -sf $(LIBTARGET) $(BUILDDIR)/$(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(BUILDDIR)/$(LIBDIR)/$(LIBSONAME)

# Compile .cc tests
$(BUILDDIR)/$(OBJDIR)/$(TESTSDIR)/%.o: $(TESTSDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) -o $@ -I$(BUILDDIR)/$(INCDIR) $(MPI_INC) $(CXXFLAGS) -c $<

# Compile .cu tests
$(BUILDDIR)/$(OBJDIR)/$(TESTSDIR)/%.o: $(TESTSDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) -o $@ -I$(BUILDDIR)/$(INCDIR) $(MPI_INC) $(NVCUFLAGS) -c $<

# Test bins
$(BUILDDIR)/$(BINDIR)/%: $(BUILDDIR)/$(OBJDIR)/%.o $(LIBTARGET)
	@mkdir -p $(@D)
	$(NVCC) -o $@ $< $(NVLDFLAGS) $(MPI_LDFLAGS) -L$(BUILDDIR)/$(LIBDIR) -lmscclpp

clean:
	rm -rf $(BUILDDIR)
