OBJECTS = par.o matrix.o
EXEC = pann

ALL_INCLUDES = -L/usr/local/cuda-8.0/lib64/
ALL_LIBS = -lm -lcublas -lcudart

C_INCLUDES = -I/usr/local/cuda-8.0/include/
C_LIBS = -lcublas -lcudart

COMPUTE_VER ?= sm_52
COMPUTE = -arch=$(COMPUTE_VER)

NVCC= /usr/local/cuda-8.0/bin/nvcc
CC= gcc

#COMPUTE = -arch=sm_30
# gcc -std=gnu99 -c -O3 -Wall a4.c -o main.o
# nvcc p1_mult.cu -arch=sm_30 -dc
# nvcc -arch=sm_30 -dlink main.o p1_mult.o -o gpuCode.o
# g++ main.o p1_mult.o gpuCode.o -I/usr/local/cuda-8.0/lib64/ -lcudart -o a4

all: $(OBJECTS) gpuCode.o
	g++ $(OBJECTS) gpuCode.o $(ALL_INCLUDES) $(ALL_LIBS) -o $(EXEC)

%.o: %.c
	gcc -std=gnu99 -c -O3 -Wall $(C_INCLUDES) $(C_LIBS) $< -o $@

%.o: %.cu
	$(NVCC) $(COMPUTE) -O3 -dc -D_FORCE_INLINES $< -o $@

gpuCode.o : $(OBJECTS)
	$(NVCC) $(COMPUTE) -O3 -dlink $^ -o $@

clean:
	rm -f *.o $(EXEC)
