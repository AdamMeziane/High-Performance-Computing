CXX = mpicxx  -fopenmp
CXXFLAGS = -std=c++11 -Wall -o3 -ftree-vectorize

HDRS = LidDrivenCavity.h SolverCG.h mpi.h omp.h
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o

TESTS = UnitTest.o SolverCG.o LidDrivenCavity.o

LIBS = -lblas -lboost_program_options -lboost_mpi -lboost_serialization -lboost_unit_test_framework

DOXYFILE = Doxyfile
DOXYGEN = doxygen

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

UnitTest: $(TESTS) 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

doc: $(DOXYFILE)
	$(DOXYGEN) $(DOXYFILE)

$(DOXYFILE):
	$(DOXYGEN) -g $(DOXYFILE)

clean-doc:
	rm -rf html/ latex/

mpirun: solver
	time mpirun -np 16  -mca oorte_base_help_aggregate 0  -v ./solver

mpirun-test: unittests
	mpirun -np 4  -mca oorte_base_help_aggregate 0  -v ./UnitTest

all: solver doc

.PHONY: clean clean-doc

clean:
	rm -f *.o LidDrivenCavitySolver UnitTest

clean-all: clean clean-doc