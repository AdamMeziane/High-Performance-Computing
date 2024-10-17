#include "SolverCG.h"
#include "LidDrivenCavity.h"
#define BOOST_TEST_MODULE SolverCGTest
#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>

using namespace std;
#define IDX(I, J) ((J) * Nx + (I))

/*We now need to use the boost unit test frame work to
test other methods in the code*/

// check the output array is similar to the expected output

struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);



bool CheckArrayEquality(double* a, double* b, int size, double tolerance)
{
    for(int i = 0; i < size; i++) {
        if(std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

BOOST_AUTO_TEST_CASE(Advance){
    bool check = true;
    int size = 0;
    int world_rank = 0;
    int grid_rank = 0;
    int dims = 2;
    int Coords[dims];
    bool test = false;

    // Get the comm size on each process.
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //Check the number of processors is a square number

    double sqrt_size = std::sqrt(static_cast<double>(size));

    if (floor(sqrt_size) != sqrt_size){
        if (world_rank == 0){
                cout << "The number of process's must be a square number" << endl;
        }
        MPI_Finalize();
        check = false;
    }
    //Generate the grid

    int side = floor(sqrt_size);
    int sizes[dims] = {side,side};
    int periods[dims] = {0,0};
    int reorder = 1;
    MPI_Comm Mygrid;
    MPI_Cart_create(MPI_COMM_WORLD,dims,sizes,periods,reorder, &Mygrid);
    MPI_Comm_rank(Mygrid, &grid_rank);
    MPI_Cart_coords(Mygrid, grid_rank, dims, Coords);

    int Nx = 9;
    int Ny = 9;
    //Local sides
    int r_x     = Nx % side;           // x remainder

    int r_y     = Ny % side;           // y remainder

    int k_x     = (Nx - r_x) / side;   // minimum size of chunk in x

    int k_y     = (Ny - r_y) / side;   // minimum size of chunk in y

    int Start_x = 0;                   // start index of chunk in x
    int Start_y = 0;                   // start index of chunk in y
    int End_x   = 0;                   // end index of chunk in x
    int End_y   = 0;                   // end index of chunk in y

    if (Coords[1] < r_x) {            // for ranks < r, chunk is size k + 1
        k_x++;
        Start_x = k_x * Coords[1];
        End_x   = k_x * (Coords[1] + 1)-1;
    }
    else {                              // for ranks > r, chunk size is k
        Start_x = (k_x+1) * r_x + k_x * (Coords[1] - r_x);
        End_x   = (k_x+1) * r_x + k_x * (Coords[1] - r_x + 1)-1;
    }

    if ((side-1)-Coords[0] < r_y) {            // for ranks < r, chunk is size k + 1
        k_y++;
        Start_y = k_y * ((side-1)-Coords[0]);
        End_y   = k_y * ((side-1)-Coords[0] + 1)-1;
    }
    else 
    {                       
        Start_y = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y);
        End_y   = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y + 1)-1;
    }

    int X_cell = k_x;
    int Y_cell = k_y;


    int Left_rank, Right_rank, Top_rank, Bottom_rank;
    MPI_Cart_shift(Mygrid, 1, 1, &Left_rank, &Right_rank);
    MPI_Cart_shift(Mygrid, 0, 1, &Top_rank, &Bottom_rank);

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(1.0, 1.0);
    solver->SetGridSize(9, 9);
    solver->SetTimeStep(0.01);
    solver->SetFinalTime(1.0);
    solver->SetReynoldsNumber(10.0);
    solver->MPIParameters(Mygrid,Coords,Left_rank,Right_rank,Top_rank,Bottom_rank,Start_x,End_x,Start_y,End_y,X_cell,Y_cell, grid_rank, side,test);

    if(grid_rank == 0){
        solver->PrintConfiguration();
    }

    solver->Initialise(); //makes new variables and a new object of solver class

    solver->WriteSolution("InitialTesting.txt");

    solver->Integrate();

    solver->WriteSolution("FinalTesting.txt");

    // MPI_Finalize();

    if (grid_rank == 0){
        
    //open the files and compare the results
    ifstream file1("RealFinal.txt"), file2("FinalTesting.txt"), file3("InitialTesting.txt"), file4("RealInitial.txt");
    string line1, line2, line3, line4;
    double val1,val2;


    if(!file1 || !file2 || !file3 || !file4) {
        cerr << "Error opening files." << endl;
        check = false;
    }

    //compare the files
    while(getline(file1, line1) && getline(file2, line2)) {
        stringstream ss1(line1);
        stringstream ss2(line2);
        while(ss1 >> val1 && ss2 >> val2){
            if(abs(val1 - val2) > 1e-1) {
                cout << "Final files are different" << endl;
                check = false;
            }

        }

    }

    //compare the files
    while(getline(file3, line3) && getline(file4, line4)) {
        stringstream ss1(line3);
        stringstream ss2(line4);
        while(ss1 >> val1 && ss2 >> val2){
            if(abs(val1 - val2) > 1e-1) {
                cout << "Final files are different" << endl;
                check = false;
            }
        }

    }


    file1.close();
    file2.close();
    file3.close();
    file4.close();

    BOOST_CHECK(check);
    }
}

BOOST_AUTO_TEST_CASE(solve)
{
    bool test = true;
    bool check = true;
    int size = 0;
    int world_rank = 0;
    int grid_rank = 0;
    int dims = 2;
    int Coords[dims];
    
    // Get the comm size on each process.
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //Check the number of processors is a square number

    double sqrt_size = std::sqrt(static_cast<double>(size));

    if (floor(sqrt_size) != sqrt_size){
        if (world_rank == 0){
                cout << "The number of process's must be a square number" << endl;
        }
        MPI_Finalize();
        check = false;
    }
    //Generate the grid

    int side = floor(sqrt_size);
    int sizes[dims] = {side,side};
    int periods[dims] = {0,0};
    int reorder = 1;
    MPI_Comm Mygrid;
    MPI_Cart_create(MPI_COMM_WORLD,dims,sizes,periods,reorder, &Mygrid);
    MPI_Comm_rank(Mygrid, &grid_rank);
    MPI_Cart_coords(Mygrid, grid_rank, dims, Coords);

    int Nx = 9;
    int Ny = 9;
    //Local sides
    int r_x     = Nx % side;           // x remainder

    int r_y     = Ny % side;           // y remainder

    int k_x     = (Nx - r_x) / side;   // minimum size of chunk in x

    int k_y     = (Ny - r_y) / side;   // minimum size of chunk in y

    int Start_x = 0;                   // start index of chunk in x
    int Start_y = 0;                   // start index of chunk in y
    int End_x   = 0;                   // end index of chunk in x
    int End_y   = 0;                   // end index of chunk in y

    if (Coords[1] < r_x) {            // for ranks < r, chunk is size k + 1
        k_x++;
        Start_x = k_x * Coords[1];
        End_x   = k_x * (Coords[1] + 1)-1;
    }
    else {                              // for ranks > r, chunk size is k
        Start_x = (k_x+1) * r_x + k_x * (Coords[1] - r_x);
        End_x   = (k_x+1) * r_x + k_x * (Coords[1] - r_x + 1)-1;
    }

    if ((side-1)-Coords[0] < r_y) {            // for ranks < r, chunk is size k + 1
        k_y++;
        Start_y = k_y * ((side-1)-Coords[0]);
        End_y   = k_y * ((side-1)-Coords[0] + 1)-1;
    }
    else 
    {                       
        Start_y = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y);
        End_y   = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y + 1)-1;
    }

    int X_cell = k_x;
    int Y_cell = k_y;


    int Left_rank, Right_rank, Top_rank, Bottom_rank;
    MPI_Cart_shift(Mygrid, 1, 1, &Left_rank, &Right_rank);
    MPI_Cart_shift(Mygrid, 0, 1, &Top_rank, &Bottom_rank);

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(1.0, 1.0);
    solver->SetGridSize(9, 9);
    solver->SetTimeStep(0.01);
    solver->SetFinalTime(1.0);
    solver->SetReynoldsNumber(10.0);
    solver->MPIParameters(Mygrid,Coords,Left_rank,Right_rank,Top_rank,Bottom_rank,Start_x,End_x,Start_y,End_y,X_cell,Y_cell, grid_rank, side,test);

    if(grid_rank == 0){
        solver->PrintConfiguration();
    }
    solver->Initialise(); //makes new variables and a new object of solver class

    solver->Integrate();

    solver->SolveCGTestData("SolverCGRun.txt");

    //MPI_Finalize();

    //open the files and compare the results
    ifstream file1("SolverCGRun.txt"), file2("SolverCGUnitTest.txt");
    string line1, line2;
    double val1,val2;


    if(!file1) {
        cerr << "Error opening file 1." << endl;
        check = false;
    }

    if(!file2) {
        cerr << "Error opening file 2." << endl;
        check = false;
    }

    //compare the files
    while(getline(file1, line1) && getline(file2, line2)) {
        stringstream ss1(line1);
        stringstream ss2(line2);
        while(ss1 >> val1 && ss2 >> val2){
            if(abs(val1 - val2) > 1e-1) {
                cout << "Final files are different" << endl;
                check = false;
            }

        }

    }

    file1.close();
    file2.close();


    BOOST_CHECK(check);
}

