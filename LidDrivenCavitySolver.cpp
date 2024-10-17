#include <iostream>
#include <mpi.h>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"

int main(int argc, char **argv)
{
    //initialise variables
    int size = 0;
    int world_rank = 0;
    int grid_rank = 0;
    int dims = 2;
    int Coords[dims];
    bool test = false;

    // Initialise MPI.
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) 
    {
        cout << "Failed to initialise MPI" << endl;
        return -1;
    }

    // Get the comm size on each process.
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Check the number of processors is a square number
    double sqrt_size = std::sqrt(static_cast<double>(size));
    int side = floor(sqrt_size);
    if (side != sqrt_size){
        if (world_rank == 0){
                cout << "The number of process's must be a square number" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    //Generate the cartesian grid
    int sizes[dims] = {side,side};
    int periods[dims] = {0,0};
    int reorder = 1;
    MPI_Comm Mygrid;
    MPI_Cart_create(MPI_COMM_WORLD,dims,sizes,periods,reorder, &Mygrid);
    MPI_Comm_rank(Mygrid, &grid_rank);
    MPI_Cart_coords(Mygrid, grid_rank, dims, Coords);

    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    //Local sides
    int r_x     = vm["Nx"].as<int>() % side;           // x remainder
    int r_y     = vm["Ny"].as<int>() % side;           // y remainder
    int k_x     = (vm["Nx"].as<int>() - r_x) / side;   // minimum size of chunk in x
    int k_y     = (vm["Ny"].as<int>() - r_y) / side;   // minimum size of chunk in y

    int Start_x = 0;                   // start index of chunk in x
    int Start_y = 0;                   // start index of chunk in y
    int End_x   = 0;                   // end index of chunk in x
    int End_y   = 0;                   // end index of chunk in y

    if (Coords[1] < r_x) {            // for ranks < r, chunk is size k + 1
        k_x++;
        Start_x = k_x * Coords[1];
        End_x   = k_x * (Coords[1] + 1)-1;
    }
    else {                             
        Start_x = (k_x+1) * r_x + k_x * (Coords[1] - r_x);
        End_x   = (k_x+1) * r_x + k_x * (Coords[1] - r_x + 1)-1;
    }

    if ((side-1)-Coords[0] < r_y) {            // for ranks < r, chunk is size k + 1 but starting from the bottom
        k_y++;
        Start_y = k_y * ((side-1)-Coords[0]);
        End_y   = k_y * ((side-1)-Coords[0] + 1)-1;
    }
    else 
    {                       
        Start_y = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y);
        End_y   = (k_y+1) * r_y + k_y * ((side-1)-Coords[0] - r_y + 1)-1;
    }

    //local grid size
    int X_cell = k_x;
    int Y_cell = k_y;

    // Get the ranks of the neighboring processes
    int Left_rank, Right_rank, Top_rank, Bottom_rank;
    MPI_Cart_shift(Mygrid, 1, 1, &Left_rank, &Right_rank);
    MPI_Cart_shift(Mygrid, 0, 1, &Top_rank, &Bottom_rank);

    //Create a new object of the LidDrivenCavity class and set the parameters
    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    solver->MPIParameters(Mygrid,Coords,Left_rank,Right_rank,Top_rank,Bottom_rank,Start_x,End_x,Start_y,End_y,X_cell,Y_cell, grid_rank, side,test);

    if(grid_rank == 0){
        solver->PrintConfiguration();
    }

    //run simulation
    solver->Initialise(); 

    solver->WriteSolution("ic.txt");

    solver->Integrate();

    solver->WriteSolution("final.txt");

    MPI_Finalize();
    return 0;
}
