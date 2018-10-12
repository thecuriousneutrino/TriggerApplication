
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math
# NVCCFLAGS	:= -lineinfo -arch=sm_50 --ptxas-options=-v --use_fast_math

all:	daq_code daq_nhits inspect_gpu

daq_code:	daq_code.cu library_daq.h Makefile
	nvcc daq_code.cu -o daq_code $(INC) $(NVCCFLAGS) $(LIB)

daq_nhits:	daq_nhits.cu library_daq.h Makefile
	nvcc daq_nhits.cu -o daq_nhits $(INC) $(NVCCFLAGS) $(LIB)

inspect_gpu:	inspect_gpu.cu Makefile
	nvcc inspect_gpu.cu -o inspect_gpu $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f daq_code daq_nhits inspect_gpu
