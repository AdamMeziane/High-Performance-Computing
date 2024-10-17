#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <mpi.h>
using namespace std;

#include <cblas.h>

#include "SolverCG.h"

#define IDX(I,J) ((J)*Nx + (I))
#define IDX_mini(I,J) ((J)*x_cell + (I))

/**
 * @brief Constructor for the solveCG class and enters the reqired inputs
 * @param pdx dx
 * @param pdy dy
 * @param pNx Nx: Number of X grid points
 * @param pNy Ny: Number of Y grid points
*/
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
}

/**
 * @brief Destructor for the solveCG class
*/
SolverCG::~SolverCG()
{
    // delete[] r;
    // delete[] p;
    // delete[] z;
    // delete[] t;
}

/**
* @brief Sends the parameters to the solverCG class
* @param Mygrid MPI communicator
* @param Right_rank Rank of the right processor
* @param Left_rank Rank of the left processor
* @param Top_rank Rank of the top processor
* @param Bottom_rank Rank of the bottom processor
* @param X_cell Number of columns for the processor
* @param Y_cell Number of rows for the processor
* @param Grid_rank Rank of the processor
* @param Start_x Start index of the processor in the x direction
* @param End_x End index of the processor in the x direction
* @param Start_y Start index of the processor in the y direction
* @param End_y End index of the processor in the y direction
*/
void SolverCG::ParameterSends(MPI_Comm Mygrid, int Right_rank,int Left_rank,int Top_rank,int Bottom_rank, int X_cell, int Y_cell, int Grid_rank, int Start_x, int End_x, int Start_y, int End_y)
{
    this->mygrid = Mygrid;
    this->right_rank = Right_rank;
    this->left_rank = Left_rank;
    this->top_rank = Top_rank;
    this->bottom_rank = Bottom_rank;
    this->x_cell = X_cell;
    this->y_cell = Y_cell;
    this->grid_rank = Grid_rank;
    this->start_x = Start_x;
    this->end_x = End_x;
    this->start_y = Start_y;
    this->end_y = End_y;
}

/**
 * @brief Solves linear system of equations in form Ax = b using the Conjugate Gradient method
 * @param b vector on the right hand side of the system of equations
 * @param x vector on the left hand side of the system of equations
*/
void SolverCG::Solve(double* b, double* x) {
    unsigned int n = x_cell*y_cell;
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;

    r = new double[n];
    p = new double[n]; 
    z = new double[n];
    t = new double[n]; //temp  


    eps = cblas_dnrm2(n, b, 1); //local eps

    //calculate global eps
    eps = eps*eps; 
    double global_eps;
    MPI_Allreduce(&eps, &global_eps, 1, MPI_DOUBLE, MPI_SUM, mygrid);
    eps = sqrt(global_eps);
    
    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t);               // Laplacian of x gets put into t
    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    ImposeBC(r);                       //sets the BC's to 0.0

    cblas_daxpy(n, -1.0, t, 1, r, 1);  //r - t since t is our Ax0 matrix
    Precondition(r, z);                //make z is a preconditioned r
    cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0

    k = 0;
    do {
        k++;
        // Perform action of Nabla^2 * p
        ApplyOperator(p, t);//problem child

        //sum denominator and numerator for alpha and beta
        alpha = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
        
        //calculate global alpha
        double global_alpha;
        MPI_Allreduce(&alpha, &global_alpha, 1, MPI_DOUBLE, MPI_SUM, mygrid);
        alpha = global_alpha;
        
        double temp1 = cblas_ddot(n, r, 1, z, 1);
        double global_temp1;
        MPI_Allreduce(&temp1, &global_temp1, 1, MPI_DOUBLE, MPI_SUM, mygrid);
        temp1 = global_temp1;

        alpha = temp1 / alpha; // compute alpha_k

        //calculate global beta
        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
        double global_beta;
        MPI_Allreduce(&beta, &global_beta, 1, MPI_DOUBLE, MPI_SUM, mygrid);
        beta = global_beta;

        cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps = cblas_dnrm2(n, r, 1); //Get global eps
        eps = eps*eps;
        double global_eps1;
        MPI_Allreduce(&eps, &global_eps1, 1, MPI_DOUBLE, MPI_SUM, mygrid);
        eps = sqrt(global_eps1);
        //eps again
        if (eps < tol*tol) {
            break;
        }
        Precondition(r, z);

        //Global beta
        double temp2 = cblas_ddot(n, r, 1, z, 1);
        double global_temp2;
        MPI_Allreduce(&temp2, &global_temp2, 1, MPI_DOUBLE, MPI_SUM, mygrid);
        temp2 = global_temp2;
        
        beta = temp2 / beta;

        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    cout << "Converged in " << k << " iterations. eps = " << eps << endl;

    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;

}

/**
 * @brief uses a finite difference approximation of the Laplacian operator (∇^2)
 * @param in  vector of what you want to find the laplacian of
 * @param out Where the calculated laplacian gets put into
*/
void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    
    //Calculate is loop boundaries to access the interior points
    int startloopx = (start_x == 0) ? 1:0;
    int endloopx = (end_x == Nx-1) ? x_cell-1 : x_cell;

    int startloopy = (start_y == 0) ? 1:0;
    int endloopy = (end_y == Ny-1) ? y_cell-1: y_cell;

    //Send neighboring data
    SendData(in);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int j = startloopy; j < endloopy; ++j) {
        for (int i = startloopx; i < endloopx; ++i) {
            double left = (i == 0) ? left_temp[j]:in[IDX_mini(i-1,j)] ;
            double right = (i == x_cell-1) ? right_temp[j]: in[IDX_mini(i+1,j)];
            double up = (j == y_cell-1) ? top_temp[i]:in[IDX_mini(i,j+1)];
            double down = (j == 0) ? bottom_temp[i]:in[IDX_mini(i,j-1)];

            out[IDX_mini(i,j)] = ( -     left
                              + 2.0*in[IDX_mini(i,   j)]
                              -     right)*dx2i
                          + ( -     down
                              + 2.0*in[IDX_mini(i,   j)]
                              -     up)*dy2i;
        }
    }

    delete[] top_temp;
    delete[] bottom_temp;
    delete[] left_temp;
    delete[] right_temp;
    delete[] left_storage;
    delete[] right_storage;
}

/**
 * @brief Preconditions the matrix which makes the iterative method converge faster to a solution. This is Jacobi preconditioning
 * @param in Matrix to precondition
 * @param out The Matrix after preconditioning
*/
void SolverCG::Precondition(double* in, double* out) {
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 2.0*(dx2i + dy2i);
    int i, j;

    //Calculate is loop boundaries to access the interior points
    int startloopx = (start_x == 0) ? 1:0;
    int endloopx = (end_x == Nx-1) ? x_cell-1 : x_cell;

    int startloopy = (start_y == 0) ? 1:0;
    int endloopy = (end_y == Ny-1) ? y_cell-1: y_cell;

    //#pragma omp parallel for schedule(static) collapse(2)
    for (j = startloopy; j < endloopy; ++j) {
        for (i = startloopx; i < endloopx; ++i) {
            out[IDX_mini(i,j)] = in[IDX_mini(i,j)]/factor;
        }
    }

    // Boundaries
    if(top_rank == MPI_PROC_NULL)
    {
        for (i = 0; i < x_cell; ++i) {
            out[IDX_mini(i, y_cell-1)] = in[IDX_mini(i, y_cell-1)];
        }
    }

    if(bottom_rank == MPI_PROC_NULL)
    {
        for (i = 0; i < x_cell; ++i) {
            out[IDX_mini(i, 0)] = in[IDX_mini(i, 0)];
        }
    }

    if(left_rank == MPI_PROC_NULL)
    {
        for (int j = 0; j < y_cell; ++j) {
            out[IDX_mini(0, j)] = in[IDX_mini(0, j)];
        }
    }

    if(right_rank == MPI_PROC_NULL)
    {
        for (int j = 0; j < y_cell; ++j) {
            out[IDX_mini(x_cell-1, j)] = in[IDX_mini(x_cell-1, j)];
        }
    }
}

/**
 * @brief Sets the value of inout to 0.0 at the boundaries of the domain
 * @param inout The variable which will have its boundaries set to zero
*/
void SolverCG::ImposeBC(double* inout) {
    // Calculate Boundaries
    if(left_rank == MPI_PROC_NULL)
    {
        for (int j = 0; j < y_cell; ++j) {
            inout[IDX_mini(0, j)] = 0.0;
        }
    }

    if(right_rank == MPI_PROC_NULL)
    {
        for (int j = 0; j < y_cell; ++j) {
            inout[IDX_mini(x_cell-1, j)] = 0.0;
        }
    }

    if(top_rank == MPI_PROC_NULL)
    {
        for (int i = 0; i < x_cell; ++i) {
            inout[IDX_mini(i, y_cell-1)] = 0.0;
        }
    }

    if(bottom_rank == MPI_PROC_NULL)
    {
        for (int i = 0; i < x_cell; ++i) {
            inout[IDX_mini(i, 0)] = 0.0;
        }
    }
}

/**
* @brief Sends the data to the neighboring processors
* @param in The data to be sent
* @param out The data to be received
*/
void SolverCG::SendData(double* in)
{
    top_temp = new double[x_cell];
    bottom_temp = new double[x_cell];
    left_temp = new double[y_cell];
    right_temp = new double[y_cell];

    left_storage = new double[y_cell];
    right_storage = new double[y_cell];
    
    // sending top
    if(top_rank != MPI_PROC_NULL)
    {
    MPI_Sendrecv(&in[IDX_mini(0,y_cell-1)], x_cell, MPI_DOUBLE,
    top_rank, 0, top_temp, x_cell,
    MPI_DOUBLE, top_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }


    // sending bottom
    if (bottom_rank != MPI_PROC_NULL) 
    {
    MPI_Sendrecv(&in[IDX_mini(0,0)], x_cell, MPI_DOUBLE,
    bottom_rank, 0, bottom_temp, x_cell,
    MPI_DOUBLE, bottom_rank, 0,
    mygrid, MPI_STATUS_IGNORE);
    }

    //Storing left and right since we store in a row major format
    for (int j = 0; j < y_cell; j++)
    {
        left_storage[j] = in[IDX_mini(0,j)];
        right_storage[j] = in[IDX_mini(x_cell-1,j)];
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
