CC= gcc
CXX=g++
CUDA=nvcc -O3 -arch=sm_75
CFLAGS=   -O3 -Wall -Wextra -Werror -pedantic -Wconversion -std=c99     -I../libs
CXXFLAGS= -O3 -Wall -Wextra -Werror           -Wconversion -std=c++17 -I../libs
# for debug:  -fsanitize=address,undefined,float-divide-by-zero
binaries=AHOMetropolis AHOWolff AHOMulticluster

all: $(binaries)

pcg32.o:         ../libs/pcg/pcg_basic.c ../libs/pcg/pcg_basic.h
	$(CC) $(CFLAGS)    -c ../libs/pcg/pcg_basic.c -o $@

progressBar.o:   ../libs/progressBar.cpp ../libs/progressBar.h
	$(CXX) $(CXXFLAGS) -c ../libs/progressBar.cpp -o $@

AHOWolff:        ../src/AHOWolff.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h progressBar.o
	$(CXX) $(CXXFLAGS) -c ../src/AHOWolff.cpp -o AHOWolff.o
	$(CXX) $(CXXFLAGS) AHOWolff.o pcg32.o progressBar.o -o $@ -lm

AHOMetropolis:   ../src/AHOMetropolis.cu     ../libs/pcg/pcg_basic.c ../libs/progressBar.h progressBar.o ../src/AHOGpuCommon.cu
	$(CUDA) -Xcompiler "$(CXXFLAGS)" ../src/AHOMetropolis.cu progressBar.o -o $@ -lm

AHOMulticluster: ../src/AHOMulticluster.cu   ../libs/pcg/pcg_basic.c ../libs/progressBar.h progressBar.o ../src/AHOGpuCommon.cu
	$(CUDA) -Xcompiler "$(CXXFLAGS)" ../src/AHOMulticluster.cu progressBar.o -o $@ -lm

.PHONY: clean
clean:
	rm -f $(binaries) *.o *.air *.metallib

cleanobj:
	rm -f *.o *.air
