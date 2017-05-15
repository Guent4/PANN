OBJECTS = par.o matrix.o pfeed_forward.o
EXEC = pann

ALL_INCLUDES = -L/usr/local/cuda-8.0/lib64/
ALL_LIBS = -lm -lcublas -lcudart

C_INCLUDES = -I/usr/local/cuda-8.0/include/
C_LIBS = -lcublas -lcudart
C_FLAGS =

COMPUTE_VER ?= sm_52
COMPUTE = -arch=$(COMPUTE_VER)

NVCC= /usr/local/cuda-8.0/bin/nvcc


all: $(OBJECTS) gpuCode.o
	mpic++ $(OBJECTS) gpuCode.o $(ALL_INCLUDES) $(ALL_LIBS) -o $(EXEC)

%.o: %.c
	mpicc -std=gnu99 -c $(C_FLAGS) -O3 -Wall $(C_INCLUDES) $(C_LIBS) $< -o $@

%.o: %.cpp
	mpic++ -std=c++11 -c $(C_FLAGS) -O3 -Wall $(C_INCLUDES) $(C_LIBS) $< -o $@

%.o: %.cu
	$(NVCC) $(COMPUTE) -O3 -dc -D_FORCE_INLINES $< -o $@

gpuCode.o : $(OBJECTS)
	$(NVCC) $(COMPUTE) -O3 -dlink $^ -o $@

clean:
	rm -f *.o $(EXEC)
