#pragma once

#include <string>
#include <mpi.h>
using namespace std;

class SolverCG;

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    void MPIParameters(MPI_Comm Mygrid,int Coords[2],int Left_rank,int Right_rank,int Top_rank,int Bottom_rank,int Start_x,int End_x,int Start_y,int End_y, int X_cell, int Y_cell, int Grid_rank, int Side,bool Test);
    void PrintMatrix(double* A, int x_cell, int y_cell, int grid_rank);
    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();
    void DataSend_s();
    void DataSend_v();
    void Collect(double* pointer, int x_cell, int y_cell, int grid_rank);
    void SolveCGTestData(std::string file);

private:
    double* v   = nullptr;
    double* vnew= nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;
    double* v_local = nullptr;
    double* s_local = nullptr;
    double* vnew_local = nullptr;

    
    double* left_temp = nullptr;
    double* right_temp = nullptr;
    double* top_temp = nullptr;
    double* bottom_temp = nullptr;

    double* left_storage = nullptr;
    double* right_storage = nullptr;
    double* left_storage_v = nullptr;
    double* right_storage_v = nullptr;

    int* coords = nullptr;
    
    double* left_temp_v = nullptr;
    double* right_temp_v = nullptr;
    double* top_temp_v = nullptr;
    double* bottom_temp_v = nullptr;


    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;

    MPI_Comm mygrid;
    int left_rank,right_rank,top_rank, bottom_rank, start_x, end_x, start_y, end_y,x_cell,y_cell;
    double left_s,right_s,up_s,down_s;
    double left_v,right_v,up_v,down_v;
    int startloopx, endloopx, startloopy,endloopy;
    int grid_rank,side,left_x_cell,right;
    bool test;
    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};

