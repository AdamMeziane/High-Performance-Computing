#pragma once
#include <mpi.h>

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();
    void ParameterSends(MPI_Comm Mygrid, int Right_rank,int Left_rank,int Top_rank,int Bottom_rank, int X_cell, int Y_cell, int Grid_rank, int Start_x, int End_x, int Start_y, int End_y);
    void SendData(double* in);
    void Solve(double* b, double* x);

private:
    double dx;
    double dy;
    int Nx;
    int Ny;
    double* r;
    double* p;
    double* z;
    double* t;

    double* left_temp = nullptr;
    double* right_temp = nullptr;
    double* top_temp = nullptr;
    double* bottom_temp = nullptr;

    double* left_storage = nullptr;
    double* right_storage = nullptr;

    MPI_Comm mygrid;
    int right_rank;
    int left_rank;
    int top_rank;
    int bottom_rank;
    int x_cell;
    int y_cell;
    int grid_rank;

    int start_x;
    int end_x;
    int start_y;
    int end_y;

    // double dxi ;
    // double dyi ;
    // double dx2i;
    // double dy2i;
    // double factor;



    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};

