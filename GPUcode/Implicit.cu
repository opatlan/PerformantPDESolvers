#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <string.h> 


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

struct BS_DiffEq { //struct for parameters of BS
    double _nu1, _nu2, _dt, _sigma, _intRate;
    long long int _NS, _NT;
};

//function to calculate mean squared error
double mean(double *V, double Vf, double Vo, long long int N_S)
{
    double acum=0.0; //acumulator
    double mn=0.0; //mean 
    int inc=N_S/100;
    for (long long int i = 0; i < N_S; i+=inc)
    {
        if(i==0)
        {
            acum += Vo;    //Initial Value
        }

        acum+=V[i];
        
        if(i==N_S-1)
        {
            acum += Vf;    //Last Value
        }

    }
    mn=acum/100; //Normalize
    return mn;
}
__global__
void calc_coef(double* _S, double* A, double* B, double* C, BS_DiffEq* pbs_diff_eq)
{

    int th = threadIdx.x; //thread number in block
    int blk_sz = blockDim.x; //block size
    int blk_id = blockIdx.x; //block number in grid
    int index = blk_sz * blk_id + th; // global index
    double _nu1 = pbs_diff_eq->_nu1; // dt / dS^2
    double _nu2 = pbs_diff_eq->_nu2; // dt / dS
    double _dt = pbs_diff_eq->_dt; // time step size
    double _volatility = pbs_diff_eq->_sigma;
    double _intRate = pbs_diff_eq->_intRate;
    long int N_S = pbs_diff_eq->_NS; //number of stock levels
    long int N_t = pbs_diff_eq->_NT; //number of time levels
    double a_k = 0.5 * (_volatility * _volatility) * (_S[index] * _S[index]);
    double b_k = _intRate * _S[index];
    double c_k = -_intRate;
    double Ak = 0.0;
    double Bk = 0.0;
    double Ck = 0.0;
    if (index < N_S) //conditional for fitting FD column size with GPU grid
    {
        Ak = -_nu1 * a_k + 0.5 * _nu2 * b_k;
        Bk = 2*_nu1 * a_k - _dt * c_k;
        Ck = -_nu1 * a_k - 0.5 * _nu2 * b_k;

        A[index] = Ak;
        B[index] = 1 + Bk;
        C[index] = Ck;

        if (index == N_S - 1)  // lower boundary condition
        {
            A[index] = Ak - Ck;
            B[index] = 1 + Bk + 2 * Ck;
        }                   
    }
}

int main(int argc, char** argv)
//int main(void)
{
    // Host problem definition
    int gridsize = atoi(argv[1]);  // grid size (number of asset levels)
    double volatility = atof(argv[2]);
    double expiration = atof(argv[3]);
    int blksize = atoi(argv[4]); //block size
    int tstsize = atoi(argv[5]);  //number of time levels

    double Vol = volatility, Int_Rate = 0.05, Expiration = expiration, Strike = 100.0; //params of BS

    //double Vol = 0.2, Int_Rate = 0.05, Expiration = 1.0, Strike = 100.0;
    int block_size = blksize;
    long long int N_Sp = gridsize; // total number of asset levels
    long long int N_S = N_Sp - 2; // total number of asset levels in FD matrices without boundary elements

    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum

    double dS = (2 * Strike) / N_Sp; //asset step

    long long int N_t = tstsize;
    double dt = Expiration / N_t; //time step
    const int nrh = 1; // number of right hand sides in algebraic solver
    const float h_one = 1;
    const float h_zero = 0;
    size_t lworkInBytes = 0;
    char* d_work = NULL;


    const int A_num_rows = N_S;
    const int A_num_cols = N_S;
    const int A_nnz = (A_num_cols + 2 * (A_num_cols - 1));

    t0 = clock();
    double* hX = (double*)malloc(N_Sp * sizeof(*hX)); //host V^k array
    double* hY = (double*)malloc(A_num_cols * sizeof(*hY)); //host V^k+1 array
    double* hY_result = (double*)malloc(N_Sp * sizeof(*hY_result));

    double* S = (double*)malloc(N_Sp * sizeof(*S)); // host stock array
    double* A = (double*)malloc(N_S * sizeof(*A)); // host coefficient A array
    double* B = (double*)malloc(N_S * sizeof(*B)); // host coefficient B array
    double* C = (double*)malloc(N_S * sizeof(*C)); // host coefficient C array
    BS_DiffEq* pbs_diff_eq = (BS_DiffEq*)malloc(sizeof(*pbs_diff_eq)); // params structure

    double     alpha = 1.0f; // alpha in y= alpha *Ax + beta*y
    double     beta = 0.0f;  // beta in y= alpha *Ax + beta*y

    //--------------------------------------------------------------------------
    // Device memory management
    double* d_S; // device stock array
    double* d_A; // device coefficient A array
    double* d_B; // device coefficient B array
    double* d_C; // device coefficient C array
    BS_DiffEq* d_pbs_diff_eq;

    long long int* dA_csrOffsets, * dA_columns;
    double* dA_values, * dX, * dY, * dY_result; // device  V^k, V^k+1 arrays

    // memory allocation of all device arrays
    CHECK_CUDA(cudaMalloc((void**)&d_S, N_S * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_A, N_S * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_B, N_S * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_C, N_S * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_pbs_diff_eq, N_S * sizeof(BS_DiffEq)))
    CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&dY_result, N_Sp * sizeof(double)))



        for (int i = 0; i < N_Sp; i++) { // fill in stock value array
            S[i] = i * dS;
        }
    
    printf("%lf\n", S[N_Sp - 1]);



    // set initial condition

    for (int i = 0; i < N_Sp; i++) {  //initial V^k array
        hX[i] = fmaxf(S[i] - Strike, 0.0); //payoff function
    }

    printf("%lf\n", hX[N_Sp - 1]);
 
    double nu1 = (dt / (dS * dS)); // dt / dS^2
    double nu2 = (dt / dS); // dt / dS

    pbs_diff_eq->_nu1 = nu1;
    pbs_diff_eq->_nu2 = nu2;
    pbs_diff_eq->_dt = dt;
    pbs_diff_eq->_sigma = Vol;
    pbs_diff_eq->_intRate = Int_Rate;
    pbs_diff_eq->_NS = N_S;
    pbs_diff_eq->_NT = N_t;

    int numBlocks = (N_S + block_size - 1) / block_size; //number of blocks 
    CHECK_CUDA(cudaMemcpy(d_pbs_diff_eq, pbs_diff_eq, sizeof(BS_DiffEq), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, &hX[1], N_S * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(d_S, &S[1], N_S * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(d_A, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_B, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_C, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dY, &hX[1], N_S * sizeof(double), cudaMemcpyHostToDevice))

    cusparseHandle_t     handle = NULL; //handle to cuSPARSE
    CHECK_CUSPARSE(cusparseCreate(&handle)) //cuSPARSE matrix descriptor


    double V_o = 0.0; // first value in V^k array (upper boundary condition)
    double V_lo = hX[0]; // first value in V^k array first time step
    double* V_fi = (double*)malloc(2*sizeof(double)); // two final values in V^k array required for computing  the lower boundary condition
    double V_f = 0.0; // last value in V^k array (upper boundary condition)
    t1 = clock(); //setup time
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    //launch calculate coefficients kernel
    calc_coef << <numBlocks, block_size >> > (d_S, d_A, d_B, d_C, d_pbs_diff_eq);
    //device syncrhonization after kernel execution
    cudaDeviceSynchronize();

        CHECK_CUSPARSE(cusparseDgtsv2_nopivot_bufferSizeExt(
            handle, N_S,
            nrh, d_A, d_B, d_C, dY, N_S,
            &lworkInBytes))
        CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes))
        
    for (int i = 0; i < N_t; i++) //time step loop
    {


        //solve tridiagonal system using CR-PCR algorithm
            CHECK_CUSPARSE(cusparseDgtsv2_nopivot(
                handle, N_S,
                nrh, d_A, d_B, d_C, dY, N_S,
                d_work))
                
            CHECK_CUDA(cudaDeviceSynchronize())
            
        V_o = V_lo * (1 - Int_Rate * dt); //Calculate upper boundary condition
        V_lo = V_o; // update first value for next iteration
        //--------------------------------------------------------------------------
        // device result check
        //CHECK_CUDA(cudaMemcpy(dX, dY, A_num_rows * sizeof(double), cudaMemcpyDeviceToDevice))
            CHECK_CUDA(cudaMemcpy(V_fi, &dY[N_S-2], 2 * sizeof(double), cudaMemcpyDeviceToHost))
	    V_f = 2 * V_fi[1] - V_fi[0]; // calculate lower boundary conditions

        //CHECK_CUDA(cudaMemset(dY, 0, N_S * sizeof(double)))
    }

    t2 = clock(); // computation time of full solution
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);

    CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(double), cudaMemcpyDeviceToHost)) //copy solution

    printf("%lf\n", V_f); // print final value of V^k
    printf("\n");
    double mn=0.0; //initialize mean squared value variable
    mn=mean(hY,V_f,V_o,N_Sp); // call mean squared value function
    printf("%lf\n",mn); // print mean squared value
    printf("End\n"); // print END

    //--------------------------------------------------------------------------
    // 
    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroy(handle))
    CHECK_CUDA(cudaFree(d_work))
        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))
        CHECK_CUDA(cudaFree(dY_result))
        CHECK_CUDA(cudaFree(d_A))
        CHECK_CUDA(cudaFree(d_B))
        CHECK_CUDA(cudaFree(d_C))
        CHECK_CUDA(cudaFree(d_pbs_diff_eq))

    // host memory deallocation
    free(S);
    free(A);
    free(B);
    free(C);
    free(hX);
    free(hY);
    free(hY_result);
    free(V_fi);
    free(pbs_diff_eq);
    return EXIT_SUCCESS;
}
