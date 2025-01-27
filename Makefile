CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu
SM          = 86
CCFLAG      = -std=c++14 -DNDEBUG -O3 -gencode=arch=compute_$(SM),code=sm_$(SM)
SOFLAG      = $(CCFLAG) -shared
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -lz -L$(CUDA_PATH)/lib64 -lcudart -lcuda -L$(TRT_PATH)/lib -lnvinfer


SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_CPP  = $(shell find . -name '*.cpp')

CU_OBJ      = $(SOURCE_CU:.cu=.cu.o)
CPP_OBJ     = $(SOURCE_CPP:.cpp=.cpp.o)

all: $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) $(SOFLAG) -o ./layernorm_plugin.so $^ $(LDFLAG)
	
%.cpp.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

%.cu.o: %.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<
	
.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan

