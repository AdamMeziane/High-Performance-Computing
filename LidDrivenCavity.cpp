#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>
using namespace std;

#define IDX(I,J) ((J)*Nx + (I))
#define IDX_mini(I,J) ((J)*x_cell + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Constructor for the LidDrivenCavity class
*/
LidDrivenCavity::LidDrivenCavity()
{
}

/**
@brief Destructor for the LidDrivenCavity class
*/
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/**
* @brief Sets the MPI parameters for the LidDrivenCavity class
* @param Mygrid the MPI communicator
* @param Coords the coordinates of the process
* @param Left_rank the rank of the process to the left
* @param Right_rank the rank of the process to the right
* @param Top_rank the rank of the process at the top
* @param Bottom_rank the rank of the process at the bottom
* @param Start_x the starting index in the x direction of the process
* @param End_x the ending index in the x direction of the process
* @param Start_y the starting index in the y direction of the process
* @param End_y the ending index in the y direction of the process
* @param X_cell the number of columns in the process
* @param Y_cell the number of rows in the process
* @param Grid_rank the rank of the process
* @param Side the number of processes in the x and y direction
* @param Test a boolean to determine if the test case is being run
*/
void LidDrivenCavity::MPIParameters(MPI_Comm Mygrid,int Coords[2],int Left_rank,int Right_rank,int Top_rank,int Bottom_rank,int Start_x,int End_x,int Start_y,int End_y, int X_cell, int Y_cell, int Grid_rank, int Side, bool Test)
{
    this->mygrid = Mygrid;
    this->coords = Coords;
    this->left_rank = Left_rank;
    this->right_rank = Right_rank;
    this->top_rank = Top_rank;
    this->bottom_rank = Bottom_rank;
    this->start_x = Start_x;
    this->end_x = End_x; 
    this->start_y = Start_y;
    this->end_y = End_y;
    this->x_cell = X_cell;
    this->y_cell = Y_cell;
    this->grid_rank = Grid_rank;
    this->side = Side;
    this->test = Test;
}

/**
* @brief Collects data from local matracies and sends it to the root process to be printed
* @param pointer pointer to the local matrix
* @param x_cell number of columns
* @param y_cell number of rows
*/
void LidDrivenCavity::Collect(double* pointer, int x_cell, int y_cell, int grid_rank)
{
    MPI_Barrier(mygrid);
    double* temp_check = new double[Npts]();
    //Assemble the global matrix for printing
    for(int j = 0; j < y_cell; j++)
    {
        for (int i = 0; i < x_cell; i++)
        {
            temp_check[IDX(start_x+i,start_y+j)] = pointer[IDX_mini(i,j)];
        }
    }
    double* temp_check_sum = new double[Npts]();

    MPI_Reduce(temp_check, temp_check_sum, Npts, MPI_DOUBLE, MPI_SUM, 0, mygrid);
    if (grid_rank == 0)
{
    PrintMatrix(temp_check_sum,Nx,Ny,grid_rank);
}

    delete[] temp_check;
    delete[] temp_check_sum;
}

/**
* @brief Prints out a matrix in a row major format to the terminal when stored bottom row first
* @param A pointer to the matrix bottom left corner
* @param x_cell number of columns
* @param y_cell number of rows
* @param grid_rank rank of the process
*/
void LidDrivenCavity::PrintMatrix(double* A, int x_cell, int y_cell, int grid_rank)
{
    //this is a col major print scheme
    //n = number of rows
    //m = number of columns

    for(int j = y_cell-1; j >= 0; j--)
    {
        for(int i = 0; i < x_cell ; i++)
        {
            //cout << A[j*x_cell + i] << ":(" << grid_rank << ")"<< setw(5);
            cout << A[j*x_cell + i]<< setw(5);
        }
        cout << endl;
    }
    cout << endl;
}

/**
* @brief Saves the stream function and vorticity to a file for testing
* @param file the name of the file that will store the data
*/
void LidDrivenCavity::SolveCGTestData(std::string file)
{
    if(test == true){
    MPI_Barrier(mygrid);

    //Assemble the global matrix
    for(int j = 0; j < y_cell; j++)
    {
         for (int i = 0; i < x_cell; i++)
        {
             vnew[IDX(start_x+i,start_y+j)] = vnew_local[IDX_mini(i,j)];
             tmp[IDX(start_x+i,start_y+j)] = s_local[IDX_mini(i,j)];
        }
    }

    //Generate the global matrix
    MPI_Reduce(vnew, v, Npts, MPI_DOUBLE, MPI_SUM, 0, mygrid);
    MPI_Reduce(tmp, s, Npts, MPI_DOUBLE, MPI_SUM, 0, mygrid);

    
    if (grid_rank == 0) //Write the data to a file
    {
    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            k = IDX(i, j);

            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();
    }
    }
}

/**
 * @brief Passes the value set for the size of the domain to the private variables
 * @param xlen Domain size in the x axis
 * @param ylen Domain size in the y axis
*/
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

/**
 * @brief Passes the value set for the number of grid points to the private variables
 * @param nx Number of grid points in x direction
 * @param ny Number of grid points in y direction
*/
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}

/**
 * @brief Passes the value set for the time step to the private variables
 * @param deltat time step
*/
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/**
 * @brief Passes the value set for final time step to the private variables
 * @param finalt Final time
*/
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/**
 * @brief Passes the value set for Reynolds number to the private variables
 * @param re Final time
*/
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

/**
 * @brief Assigns the correct amount of space on the heap as well as an object from the SolverCG class
*/
void LidDrivenCavity::Initialise()
{
    CleanUp();

    v   = new double[Npts]();
    vnew= new double[Npts]();
    s   = new double[Npts]();
    tmp = new double[Npts]();

    v_local = new double[x_cell*y_cell]();
    s_local = new double[x_cell*y_cell]();
    vnew_local = new double[x_cell*y_cell]();

    cg  = new SolverCG(Nx,Ny,dx,dy);

    top_temp = new double[x_cell]();
    bottom_temp = new double[x_cell]();

    left_temp = new double[y_cell]();
    right_temp = new double[y_cell]();

    top_temp_v = new double[x_cell]();
    bottom_temp_v = new double[x_cell]();

    left_temp_v = new double[y_cell]();
    right_temp_v = new double[y_cell]();

    left_storage_v = new double[y_cell]();
    right_storage_v = new double[y_cell]();

    left_storage = new double[y_cell]();
    right_storage = new double[y_cell]();

}

/**
 * @brief Calculates the number of time steps requested using the final time and the time step size.
 *  Then integrate loops the advance function to calculate the vorticity and stream function at each time step.
*/
void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);

    if (test == true)
    {
        NSteps = 1;  //To prevent the test case from running wasted cycles
    }

    for (int t = 0; t < NSteps; ++t)
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl; 
        Advance();
    }
}

/**
 * @brief Takes the local variable stream function , s, and calculates horizontal and vertical velocity
 * The velocity for the top boundary condition is then set to U. The data is then written to a file
 * giving the coordinates, then the vorticity, the stream function and finally the velocities
 * @param file the name of the file that will store the data
*/
void LidDrivenCavity::WriteSolution(std::string file)
{

    MPI_Barrier(mygrid);

    for(int j = 0; j < y_cell; j++)
    {
         for (int i = 0; i < x_cell; i++)
        {
             vnew[IDX(start_x+i,start_y+j)] = vnew_local[IDX_mini(i,j)];
             tmp[IDX(start_x+i,start_y+j)] = s_local[IDX_mini(i,j)];
        }
    }

    MPI_Reduce(vnew, v, Npts, MPI_DOUBLE, MPI_SUM, 0, mygrid);
    MPI_Reduce(tmp, s, Npts, MPI_DOUBLE, MPI_SUM, 0, mygrid);

    if (grid_rank == 0)
    {
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy; 
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;
    }

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            k = IDX(i, j);
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
              << " " << u0[k] << " " << u1[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();

    delete[] u0;
    delete[] u1;
    }
}

/**
 * @brief Print out to the terminal the configuration of the simulation
*/
void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

/**
 * @brief Deletes the memory that was allocated on the heap to allow the system to use it again
*/
void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete[] v_local;
        delete[] s_local;
        delete[] vnew_local;

        delete[] left_temp;
        delete[] right_temp;
        delete[] top_temp;
        delete[] bottom_temp;
        delete[] left_temp_v;
        delete[] right_temp_v;
        delete[] top_temp_v;
        delete[] bottom_temp_v;  

        delete[] coords;

        delete[] left_storage;
        delete[] right_storage;
        delete[] left_storage_v;
        delete[] right_storage_v;

        delete cg;
    }
}

/**
 * @brief Calculates the grid spacing and calculates the number of grid points
*/
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx * Ny;
}

/**
 * @brief Calculates the vorticity at each boundary at time t using the stream function at time t. Then the vorticity
 * of the interior grid points is calculated, both of these spacial derivatives are calculated using second-order centeral-difference. 
 * Following this the vorticity of the interior grid points at time t + dt is calculated with a forward time integration scheme. The Solve function
 * from the SolverCG class is called which uses a conjugate gradient algorithm to get stream function at time t + dt.
*/
void LidDrivenCavity::Advance()
{
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    
    
    //Setting the boundary conditions
    if(left_rank == MPI_PROC_NULL || right_rank == MPI_PROC_NULL || top_rank == MPI_PROC_NULL || bottom_rank == MPI_PROC_NULL)
    {
        //Setting the loop limits for the boundary conditions to ensure the corners are skipped
        int lower_x = (start_x == 0) ?  1:0;
        int upper_x = (end_x == Nx-1) ? x_cell-1:x_cell;

        int lower_y = (start_y == 0) ?  1:0;
        int upper_y = (end_y == Ny-1) ? y_cell-1:y_cell;  

        //top
        if(top_rank == MPI_PROC_NULL)
        {
            for(int i = lower_x; i < upper_x; i++)
            {
                 v_local[IDX_mini(i,y_cell-1)] = 2.0 * dy2i * (s_local[IDX_mini(i,y_cell-1)] - s_local[IDX_mini(i,y_cell-2)]) - 2.0 * dyi*U;
            }
        }

        //bottom
        if(bottom_rank == MPI_PROC_NULL)
        {
            for(int i = lower_x; i < upper_x; i++)
            {
                v_local[IDX_mini(i,0)]    = 2.0 * dy2i * (s_local[IDX_mini(i,0)]    - s_local[IDX_mini(i,1)]);
            }
        }

        //left
        if (left_rank == MPI_PROC_NULL)
        {
            for(int j = lower_y; j < upper_y; j++)
            {
                v_local[IDX_mini(0,j)]    = 2.0 * dx2i * (s_local[IDX_mini(0,j)]    - s_local[IDX_mini(1,j)]);    
            }
        }
        //right
        if(right_rank == MPI_PROC_NULL)
        {
            for(int j = lower_y; j < upper_y; j++)
            {
                v_local[IDX_mini(x_cell-1,j)] = 2.0 * dx2i * (s_local[IDX_mini(x_cell-1,j)] - s_local[IDX_mini(x_cell-2,j)]);
            }
        }
    }


    //Sending the neighbouring data
    DataSend_s();
    DataSend_v();

    //Calculating the loop limits for the interior grid points
    startloopx = (start_x == 0) ? 1:0;
    endloopx = (end_x == Nx-1) ? x_cell-1 : x_cell;

    startloopy = (start_y == 0) ? 1:0;
    endloopy = (end_y == Ny-1) ? y_cell-1: y_cell;


    //Calculating the vorticity at the interior grid points
    for(int j = startloopy; j < endloopy; ++j)
    {
        for (int i = startloopx; i < endloopx; ++i)
        {
            //Calculating if need to access neighbouring data or if its on the process already
            left_s = (i == 0) ? left_temp[j]:s_local[IDX_mini(i-1,j)] ;
            right_s = (i == x_cell-1) ? right_temp[j]: s_local[IDX_mini(i+1,j)];
            up_s = (j == y_cell-1) ? top_temp[i]:s_local[IDX_mini(i,j+1)];
            down_s = (j == 0) ? bottom_temp[i]:s_local[IDX_mini(i,j-1)];

            v_local[IDX_mini(i,j)] = dx2i*(
                    2.0 * s_local[IDX_mini(i,j)] - right_s - left_s)
                        + 1.0/dy/dy*( 
                    2.0 * s_local[IDX_mini(i,j)] - up_s - down_s);
        }
    }

       //time advance
    for(int j = startloopy; j < endloopy; j++)
    {
        for (int i = startloopx; i < endloopx; i++)
        {
            //Calculating if need to access neighbouring data or if its on the process already
            left_s = (i == 0) ? left_temp[j]:s_local[IDX_mini(i-1,j)] ;
            right_s = (i == x_cell-1) ? right_temp[j]: s_local[IDX_mini(i+1,j)];
            up_s = (j == y_cell-1) ? top_temp[i]:s_local[IDX_mini(i,j+1)];
            down_s = (j == 0) ? bottom_temp[i]:s_local[IDX_mini(i,j-1)];

            left_v = (i == 0) ? left_temp_v[j]:v_local[IDX_mini(i-1,j)] ;
            right_v = (i == x_cell-1) ? right_temp_v[j]: v_local[IDX_mini(i+1,j)];            
            up_v = (j == y_cell-1) ? top_temp_v[i]:v_local[IDX_mini(i,j+1)];
            down_v = (j == 0) ? bottom_temp_v[i]:v_local[IDX_mini(i,j-1)];

            vnew_local[IDX_mini(i,j)] = v_local[IDX_mini(i,j)] + dt*(
                ( (right_s - left_s) * 0.5 * dxi
                *(up_v - down_v) * 0.5 * dyi)
            - ( (up_s - down_s) * 0.5 * dyi
                *(right_v - left_v) * 0.5 * dxi)
            + nu * (right_v - 2.0 * v_local[IDX_mini(i,j)] + left_v)*dx2i
            + nu * (up_v - 2.0 * v_local[IDX_mini(i,j)] + down_v)*dy2i);
            }
        }


    if(test == true)
    {
        // Sinusoidal test case with analytical solution, which can be used to test
        // the Poisson solver
        const int k = 3;
        const int l = 3;
        for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            vnew[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
        }

     for(int j = 0; j < y_cell; j++)
     {
         for (int i = 0; i < x_cell; i++)
         {
             vnew_local[IDX_mini(i,j)] = vnew[IDX(start_x+i,start_y+j)];
         }
     }  
    }

    //Send the data to the SolverCG class to solve the stream function
    cg->ParameterSends(mygrid, right_rank,left_rank,top_rank,bottom_rank, x_cell, y_cell, grid_rank, start_x, end_x, start_y, end_y);
    cg->Solve(vnew_local, s_local);
}

/**
 * @brief Sends the stream function data to the neighboring processes
*/
void LidDrivenCavity::DataSend_s()
{
   // sending top
    if(top_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(&s_local[IDX_mini(0,y_cell-1)], x_cell, MPI_DOUBLE,
    top_rank, 0, top_temp, x_cell,
    MPI_DOUBLE, top_rank, 0,
    mygrid, MPI_STATUS_IGNORE);

    }

    // sending bottom
    if (bottom_rank != MPI_PROC_NULL) 
    {
    MPI_Sendrecv(&s_local[IDX_mini(0,0)], x_cell, MPI_DOUBLE,
    bottom_rank, 0, bottom_temp, x_cell,
    MPI_DOUBLE, bottom_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }

    //Storing left and right since we store in a row major format
    for (int j = 0; j < y_cell; j++)
    {
        left_storage[j] = s_local[IDX_mini(0,j)];
        right_storage[j] = s_local[IDX_mini(x_cell-1,j)];
    }

    //Sending right
    if (right_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(right_storage, y_cell, MPI_DOUBLE,
    right_rank, 0, right_temp, y_cell,
    MPI_DOUBLE, right_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }

    //Sending left
    if (left_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(left_storage, y_cell, MPI_DOUBLE,
    left_rank, 0, left_temp, y_cell,
    MPI_DOUBLE, left_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }
}

/**
 * @brief Sends the velocity data to the neighboring processes
*/
void LidDrivenCavity::DataSend_v()
{
    //sending top
    if(top_rank != MPI_PROC_NULL)
    {
        MPI_Sendrecv(&v_local[IDX_mini(0,y_cell-1)], x_cell, MPI_DOUBLE,
    top_rank, 0, top_temp_v, x_cell,
    MPI_DOUBLE, top_rank, 0,
    mygrid, MPI_STATUS_IGNORE);

    }

    //Sending bottom
    if (bottom_rank != MPI_PROC_NULL) 
    {
    MPI_Sendrecv(&v_local[IDX_mini(0,0)], x_cell, MPI_DOUBLE,
    bottom_rank, 0, bottom_temp_v, x_cell,
    MPI_DOUBLE, bottom_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }

    //Storing left and right since we store in a row major format
    for (int j = 0; j < y_cell; j++)
    {
        left_storage_v[j] = v_local[IDX_mini(0,j)];
        right_storage_v[j] = v_local[IDX_mini(x_cell-1,j)];
    }

    //Sending right
    if (right_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(right_storage_v, y_cell, MPI_DOUBLE,
    right_rank, 0, right_temp_v, y_cell,
    MPI_DOUBLE, right_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }

    //Sending left
    if (left_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(left_storage_v, y_cell, MPI_DOUBLE,
    left_rank, 0, left_temp_v, y_cell,
    MPI_DOUBLE, left_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }
}