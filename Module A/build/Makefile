CC=gcc
CXX=clang++
CFLAGS=-O3 -Wall -Wextra -Werror -pedantic -Wconversion -std=c99
CXXFLAGS=-O3 -Wno-gnu-statement-expression -Wall -Wextra -Werror -pedantic -Wconversion -std=gnu++20 -I../libs/metal-cpp
FRAMEWORKS=-framework Metal -framework Foundation -framework MetalKit

binaries=IsingWolff IsingWolffTri IsingWolffHex IsingMetropolis IsingMetropolisTri IsingMetropolisHex IsingMetropolisCPU IsingMetropolisCPUTri IsingMetropolisCPUHex

all: $(binaries)

pcg32.o:         ../libs/pcg/pcg_basic.c ../libs/pcg/pcg_basic.h
	$(CC) $(CFLAGS) -c ../libs/pcg/pcg_basic.c -o $@

progressBar.o:   ../libs/progressBar.cpp ../libs/progressBar.h
	$(CXX) $(CXXFLAGS) -c ../libs/progressBar.cpp -o $@ 

IsingWolff:      ../src/IsingWolff.c pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h ../libs/progressBar.h
	$(CC) $(CFLAGS) -D LATTICE=1 -c ../src/IsingWolff.c -o IsingWolff.o
	$(CC) $(CFLAGS) IsingWolff.o pcg32.o -o $@ -lm

IsingWolffTri:   ../src/IsingWolff.c pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h ../libs/progressBar.h
		$(CC) $(CFLAGS) -D LATTICE=2 -c ../src/IsingWolff.c -o IsingWolff.o
		$(CC) $(CFLAGS) IsingWolff.o pcg32.o -o $@ -lm

IsingWolffHex:   ../src/IsingWolff.c pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h ../libs/progressBar.h
		$(CC) $(CFLAGS) -D LATTICE=3 -c ../src/IsingWolff.c -o IsingWolff.o
		$(CC) $(CFLAGS) IsingWolff.o pcg32.o -o $@ -lm

IsingMetropolis.metallib:    ../src/MetalIsingMetropolis.metal
	xcrun -sdk macosx metal -D LATTICE=1 -c ../src/MetalIsingMetropolis.metal -o MetalIsingMetropolisLibrary.air
	xcrun -sdk macosx metallib MetalIsingMetropolisLibrary.air -o $@

IsingMetropolisTri.metallib: ../src/MetalIsingMetropolis.metal
	xcrun -sdk macosx metal -D LATTICE=2 -c ../src/MetalIsingMetropolis.metal -o MetalIsingMetropolisLibrary.air
	xcrun -sdk macosx metallib MetalIsingMetropolisLibrary.air -o $@

IsingMetropolisHex.metallib: ../src/MetalIsingMetropolis.metal
	xcrun -sdk macosx metal -D LATTICE=3 -c ../src/MetalIsingMetropolis.metal -o MetalIsingMetropolisLibrary.air
	xcrun -sdk macosx metallib MetalIsingMetropolisLibrary.air -o $@

MetalIsingMetropolis.o:     ../src/MetalIsingMetropolis.cpp ../src/MetalIsingMetropolis.hpp ../libs/geometry.h ../libs/progressBar.h
	$(CXX) $(CXXFLAGS) -D LATTICE=1 -c ../src/MetalIsingMetropolis.cpp -o $@

MetalIsingMetropolisTri.o:  ../src/MetalIsingMetropolis.cpp ../src/MetalIsingMetropolis.hpp ../libs/geometry.h ../libs/progressBar.h
	$(CXX) $(CXXFLAGS) -D LATTICE=2 -c ../src/MetalIsingMetropolis.cpp -o $@

MetalIsingMetropolisHex.o:  ../src/MetalIsingMetropolis.cpp ../src/MetalIsingMetropolis.hpp ../libs/geometry.h ../libs/progressBar.h
	$(CXX) $(CXXFLAGS) -D LATTICE=3 -c ../src/MetalIsingMetropolis.cpp -o $@

IsingMetropolis:    ../src/IsingMetropolis.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h IsingMetropolis.metallib    ../libs/geometry.h ../libs/progressBar.h progressBar.o MetalIsingMetropolis.o
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=1 ../src/IsingMetropolis.cpp progressBar.o MetalIsingMetropolis.o pcg32.o -o $@ -lm

IsingMetropolisTri: ../src/IsingMetropolis.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h IsingMetropolisTri.metallib ../libs/geometry.h ../libs/progressBar.h progressBar.o MetalIsingMetropolisTri.o
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=2 ../src/IsingMetropolis.cpp progressBar.o MetalIsingMetropolisTri.o pcg32.o -o $@ -lm

IsingMetropolisHex: ../src/IsingMetropolis.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h IsingMetropolisHex.metallib ../libs/geometry.h ../libs/progressBar.h progressBar.o MetalIsingMetropolisHex.o
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=3 ../src/IsingMetropolis.cpp progressBar.o MetalIsingMetropolisHex.o pcg32.o -o $@ -lm

IsingMetropolisCPU:    ../src/IsingMetropolisCPU.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=1 ../src/IsingMetropolisCPU.cpp pcg32.o -o $@ -lm

IsingMetropolisCPUTri: ../src/IsingMetropolisCPU.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=2 ../src/IsingMetropolisCPU.cpp pcg32.o -o $@ -lm

IsingMetropolisCPUHex: ../src/IsingMetropolisCPU.cpp pcg32.o ../libs/pcg/pcg_basic.h ../libs/progressBar.h ../libs/geometry.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -D LATTICE=3 ../src/IsingMetropolisCPU.cpp pcg32.o -o $@ -lm

.PHONY: clean
clean:
	rm -f $(binaries) *.o *.air *.metallib

cleanobj:
	rm -f *.o *.air
