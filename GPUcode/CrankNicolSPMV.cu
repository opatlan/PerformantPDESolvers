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

double mean(double *V, double Vf, double Vo, long long int N_S) //function to calculate mean squared error
{
    double acum=0.0; //acumulator
    double mn=0.0;  //mean 
    int inc=N_S/100;
    for (long long int i = 0; i < N_S; i+=inc)
    {
        if(i==0)
        {
            acum += Vo; //Initial Value    
        }

        acum+=V[i];
        
        if(i==N_S-1)
        {
            acum += Vf;  //Last Value     
        }

    }
    mn=acum/100; //Normalize
    return mn;
}
__global__
void calc_coef(double* _S, double* A, double* B, double* C, double* nA, double* nB, double* nC, double* rhc, long long int* ICol, long long int* IRow, BS_DiffEq* pbs_diff_eq) //kernel to calculate coefficient matrices
{

    int th = threadIdx.x; //thread number in block 
    int blk_sz = blockDim.x; //block size
    int blk_id = blockIdx.x; //block number in grid
    int index = blk_sz * blk_id + th; // global index
    double _nu1 = pbs_diff_eq->_nu1; // dt / dS^2
    double _nu2 = pbs_diff_eq->_nu2; // dt / dS
    double _dt = pbs_diff_eq->_dt;  // time step size
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
        Ak = 0.5 * _nu1 * a_k - 0.25 * _nu2 * b_k;
        Bk = -_nu1 * a_k + 0.5 * _dt * c_k;
        Ck = 0.5 * _nu1 * a_k + 0.25 * _nu2 * b_k;

        A[index] = -Ak; //lhs A
        B[index] = 1 - Bk; //lhs B
        C[index] = -Ck; //lhs C

        nA[index] = Ak; //rhs A
        nB[index] = 1 + Bk; //rhs B
        nC[index] = Ck;  //rhs C


        if (index == 0) // lower boundary condition
        {
            /*B[index] = 1 - Bk - 2 * Ak;
            C[index] = -Ck + Ak;

            nB[index] = 1 + Bk + 2 * Ak; 
            nC[index] = Ck - Ak;*/

            rhc[index] = nB[index]; //first value of CSR non zero elements array 
            rhc[index+1] = nC[index]; //second value of CSR non zero elements array 
            ICol[index] = index; // column index of first value CSR
            ICol[index + 1] = index + 1; // column index of second value CSR
            IRow[index] = 0; //number elements before current entries CSR

        }
        else
        {
            if (index == N_S - 1) //lower boundary condition
            {
                A[index] = -Ak + Ck;   //lower boundary condition for lhs A
                B[index] = 1 - Bk -2*Ck; //lower boundary condition for lhs B

                nA[index] = Ak - Ck; //lower boundary condition for rhs A
                nB[index] = 1 + Bk + 2 * Ck; //lower boundary condition for rhs B

                rhc[(index - 1) * 3 + 2] = nA[index]; //penultimate value of CSR non zero elements array 
                rhc[(index - 1) * 3 + 3] = nB[index]; //last value of CSR non zero elements array 
                ICol[(index - 1) * 3 + 2] = index - 1;  // column index of penultimate value CSR
                ICol[(index - 1) * 3 + 3] = index; // column index of last value CSR
                IRow[index] = (index - 1) * 3 + 2; //number elements before last entries CSR
                IRow[index + 1] = (index - 1) * 3 + 4; // total number of non zero elements CSR
                
                


            }
            else
            {
                
                rhc[(index - 1) * 3 + 2] = nA[index]; // rhs A^k_i in CSR non zero elements array
                rhc[(index - 1) * 3 + 3] = nB[index]; // rhs B^k_i in CSR non zero elements array
                rhc[(index - 1) * 3 + 4] = nC[index]; // rhs C^k_i in CSR non zero elements array
                ICol[(index - 1) * 3 + 2] = (index - 1); // column index of rhs A^k_i value CSR
                ICol[(index - 1) * 3 + 3] = (index - 1) + 1; // column index of rhs B^k_i value CSR
                ICol[(index - 1) * 3 + 4] = (index - 1) + 2; // column index of rhs C^k_i value CSR
                IRow[index] = (index - 1) * 3 + 2; //number elements before current entries CSR
                
            }

        }
    }
}

int main(int argc, char** argv)
//int main(void) 
{
    // Host problem definition
    int gridsize = atoi(argv[1]); // grid size (number of asset levels)
    double volatility = atof(argv[2]);
    double expiration = atof(argv[3]);
    int blksize = atoi(argv[4]); //block size
    int tstsize = atoi(argv[5]); //number of time levels

    double Vol = volatility, Int_Rate = 0.05, Expiration = expiration, Strike = 100.0; //params of BS
    int block_size = blksize;
    long long int N_Sp = gridsize; // total number of asset levels
    long long int N_S = N_Sp-2; // total number of asset levels in FD matrices without boundary elements
    
    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum

    
    double dS = (2*Strike)/N_Sp; //asset step
    long long int N_t = tstsize;
    double dt = Expiration / N_t; //time step
    const int nrh = 1; // number of right hand sides in algebraic solver
    const float h_one = 1;
    const float h_zero = 0;
    size_t lworkInBytes = 0; 
    char* d_work = NULL;

    
    long long int A_num_rows = N_S; // number of rows in FD grid without boundary elements
    long long int A_num_cols = N_S; // number of columns in FD grid without boundary elements
    long long int A_nnz = (A_num_cols + 2 * (A_num_cols - 1)); //number of non zero elements

    t0 = clock(); //start timer for setup
    long long int* hA_csrOffsets = (long long int*)malloc((A_num_cols + 1) * sizeof(*hA_csrOffsets)); //row indices array
    long long int* hA_columns = (long long int*)malloc(A_nnz * sizeof(*hA_columns)); //column indices array
    double* hA_values = (double*)malloc(A_nnz * sizeof(*hA_values)); //host non zero values array
    double* hX = (double*)malloc(N_Sp * sizeof(*hX)); //host V^k array
    double* hY = (double*)malloc(A_num_cols * sizeof(*hY)); //host V^k+1 array

    double* S = (double*)malloc(N_Sp * sizeof(*S)); // host stock array
    double* A = (double*)malloc(N_S * sizeof(*A)); // host coefficient lhs A array
    double* B = (double*)malloc(N_S * sizeof(*B)); // host coefficient lhs B array
    double* C = (double*)malloc(N_S * sizeof(*C)); // host coefficient lhs C array
    double* nA = (double*)malloc(N_S * sizeof(*A)); // host coefficient rhs A array
    double* nB = (double*)malloc(N_S * sizeof(*B)); // host coefficient rhs B array
    double* nC = (double*)malloc(N_S * sizeof(*C)); // host coefficient rhs C array
    BS_DiffEq* pbs_diff_eq = (BS_DiffEq*)malloc(sizeof(*pbs_diff_eq)); // params structure

    double     alpha = 1.0f; // alpha in y= alpha *Ax + beta*y
    double     beta = 0.0f; // beta in y= alpha *Ax + beta*y

    //--------------------------------------------------------------------------
    // Device memory management
    double* d_S; // device stock array
    double* d_A; // device coefficient lhs A array
    double* d_B; // device coefficient lhs B array
    double* d_C; // device coefficient lhs C array
    double* d_nA; // device coefficient rhs A array
    double* d_nB; // device coefficient rhs B array
    double* d_nC; // device coefficient rhs C array
    BS_DiffEq* d_pbs_diff_eq;

    long long int* dA_csrOffsets, * dA_columns; // device row and column indices arrays
    double* dA_values, * dX, * dY; // device non zero values, V^k, V^k+1 arrays
    // memory allocation of all device arrays
    CHECK_CUDA(cudaMalloc((void**)&d_S, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_A, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_B, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_C, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_nA, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_nB, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_nC, N_S * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&d_pbs_diff_eq, N_S * sizeof(BS_DiffEq)))
        CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(long long int)))
        CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(long long int)))
        CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(double)))



    for (int i = 0; i < N_Sp; i++) { // fill in stock value array
        S[i] = i*dS;
        //printf("%lf\n", S[i]);
    }
    printf("%lf\n", S[N_Sp - 1]);

    // set initial condition
    for (int i = 0; i < N_Sp; i++) {//initial V^k array
        hX[i] = fmaxf(S[i] - Strike, 0.0); //payoff function
    }
    
    printf("%lf\n", hX[N_Sp - 1]);
    // evaluate coefficients that are needed in finite difference approximation
    double nu1 = (dt / (dS * dS)); // dt / dS^2
    double nu2 = (dt / dS); // dt / dS

    //store in params struct
    pbs_diff_eq->_nu1 = nu1;
    pbs_diff_eq->_nu2 = nu2;
    pbs_diff_eq->_dt = dt;
    pbs_diff_eq->_sigma = Vol;
    pbs_diff_eq->_intRate = Int_Rate;
    pbs_diff_eq->_NS = N_S;
    pbs_diff_eq->_NT = N_t;
    int numBlocks = (N_S + block_size - 1) / block_size; //number of blocks 
    // copy and set initial values for device arrays from host arrays
    CHECK_CUDA(cudaMemcpy(d_pbs_diff_eq, pbs_diff_eq, sizeof(BS_DiffEq), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, &hX[1], N_S * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(d_S, &S[1], N_S * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(d_A, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_B, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_C, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(dY, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_nA, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_nB, 0, N_S * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_nC, 0, N_S * sizeof(double)))
        //printf("%lld\n", N_t);

        cusparseHandle_t     handle = NULL; //handle to cuSPARSE
        cusparseSpMatDescr_t matA; //cuSPARSE matrix descriptor
        cusparseDnVecDescr_t vecX, vecY; // cuSPARSE vector X and Y descriptor
        void* dBuffer = NULL;
        size_t               bufferSize = 0; // size of buffer
        CHECK_CUSPARSE(cusparseCreate(&handle)) // create cuSPARSE handle 

        double V_o = 0.0; // first value in V^k array (upper boundary condition)
        double V_lo = hX[0]; // first value in V^k array first time step
        double V_f = 0.0; // last value in V^k array (upper boundary condition)
        double* V_fi = (double*)malloc(2*sizeof(double)); // two final values in V^k array required for computing  the lower boundary condition
        t1 = clock(); //setup time
        t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("Init took %f seconds.  Begin compute\n", t1sum);
        //launch calculate coefficients kernel
        calc_coef << <numBlocks, block_size >> > (d_S, d_A, d_B, d_C, d_nA, d_nB, d_nC, dA_values, dA_columns, dA_csrOffsets, d_pbs_diff_eq); 
        //device syncrhonization after kernel execution
        cudaDeviceSynchronize();



        // Create rhs sparse matrix A in CSR format
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
            dA_csrOffsets, dA_columns, dA_values,
            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))


            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F))
            // Create dense vector y
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F))
            // allocate  buffersize of product
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_MV_ALG_DEFAULT, &bufferSize))
            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
            // allocate  buffersize of tridiagonal solver
            CHECK_CUSPARSE(cusparseDgtsv2_nopivot_bufferSizeExt(
                handle, N_S,
                nrh, d_A, d_B, d_C, dY, N_S,
                &lworkInBytes))
            CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes))


        for (int i = 0; i < N_t; i++) //time step loop 
        {

            // Create dense vector X

                // execute SpMV
                CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer))

                // synchronize device after product
                CHECK_CUDA(cudaDeviceSynchronize())
                // destroy matrix/vector descriptors
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))


                //solve tridiagonal system using CR-PCR algorithm
                CHECK_CUSPARSE(cusparseDgtsv2_nopivot(
                    handle, N_S,
                    nrh, d_A, d_B, d_C, dY, N_S,
                    d_work))
                // synchronize device after product
                CHECK_CUDA(cudaDeviceSynchronize())
                V_o = V_lo * (1 - Int_Rate * dt); //Calculate upper boundary condition
                V_lo = V_o; // update first value for next iteration
            CHECK_CUDA(cudaMemcpy(dX, dY, A_num_rows * sizeof(double), cudaMemcpyDeviceToDevice)) //update V^k with V^k+1
            //Copy last two values of  V^k to compute upper boundary conditions
            CHECK_CUDA(cudaMemcpy(V_fi, &dY[N_S-2], 2 * sizeof(double), cudaMemcpyDeviceToHost)) 
            if(i!=N_Sp-1) //if not final stock level 
            {
                //Create X and Y vectors with new values
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F))
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F))
            }

            V_f = 2 * V_fi[1] - V_fi[0]; // calculate lower boundary conditions

            //CHECK_CUDA(cudaMemset(dY, 0, N_S * sizeof(double)))
        }
        
        t2 = clock();// computation time of full solution
        t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
        printf("Computing took %f seconds.  Finish to compute\n", t2sum);

    CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(double), cudaMemcpyDeviceToHost)) //copy solution

    printf("%lf\n", V_f); // print final value of V^k
    printf("\n");
    double mn=0.0; //initialize mean squared value variable
    mn=mean(hY,V_f,V_o,N_Sp);// call mean squared value function
    printf("%lf\n",mn); // print mean squared value
    printf("End\n"); // print END

    //--------------------------------------------------------------------------
    // 
    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroy(handle))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(d_work))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))
    CHECK_CUDA(cudaFree(d_A))
    CHECK_CUDA(cudaFree(d_B))
    CHECK_CUDA(cudaFree(d_C))
    CHECK_CUDA(cudaFree(d_nA))
    CHECK_CUDA(cudaFree(d_nB))
    CHECK_CUDA(cudaFree(d_nC))
    CHECK_CUDA(cudaFree(d_pbs_diff_eq))

    // host memory deallocation
    free(S);
    free(A);
    free(B);
    free(C);
    free(nA);
    free(nB);
    free(nC);
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    free(hX);
    free(hY);
    free(V_fi);
    free(pbs_diff_eq);
    return EXIT_SUCCESS;
}
