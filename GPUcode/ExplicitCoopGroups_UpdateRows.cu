#include "cuda_runtime.h"
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>
#include "device_launch_parameters.h"
#include <math.h>
#include<vector>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
using namespace cooperative_groups;

struct BS_DiffEq { // struct for storing the BS params
    double _ak, _bk, _ck, _dk, _dt, _intRate;
    long long int _NS, _NT;
};


// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}


__global__
void grscheme(double* _S, double* V, double *Vp, BS_DiffEq* pbs_diff_eq) // calculate BS by explicit FDM kernel
{

    int th = threadIdx.x; //thread number in block
    int blk_sz = blockDim.x; //block size
    int blk_id = blockIdx.x; //block number in grid
    int index = blk_sz * blk_id + th; // global index
    extern __shared__ double shdV[]; //dynamic shared memory array
    auto g = this_grid(); // grid handle
    double _ak = pbs_diff_eq->_ak;
    double _bk = pbs_diff_eq->_bk;
    double _ck = pbs_diff_eq->_ck;
    double _dk = pbs_diff_eq->_dk;
    double _dt = pbs_diff_eq->_dt;
    double _intRate = pbs_diff_eq->_intRate;
    long long int N_S = pbs_diff_eq->_NS; //stock levels
    long long int N_t = pbs_diff_eq->_NT; // time steps
    int thp1 = th + 1;
    if (index < N_S) { //conditional for fitting FD column size with GPU grid
        for (long long int i = 1; i < N_t; i++) //time stepping inside kernel
        {
            
            shdV[thp1] = V[index]; // each thread stores its current value
            shdV[0] = 0.0; // initialize lower neighbour
            shdV[blk_sz + 1] = 0.0; // initialize upper neighbour

            if (blk_id > 0) // if not in first block
            {
                //lower neighbour comes from lower neighbouring block
                shdV[0] = V[blk_id * blk_sz - 1];
            }


            if (blk_id < (gridDim.x - 1)) // if not in last block
            {
                //upper neighbour comes from upper neighbouring block
                shdV[blk_sz + 1] = V[(blk_id + 1) * blk_sz];
                
            }


            double s = shdV[thp1], sm1 = 0.0, sp1 = 0.0; //define FD stencil
           
            __syncthreads(); //wait for all threads in block to gather their values
            if ((index - 1) > 0)
            {
                sm1 = shdV[thp1 - 1]; //read level i-1 from shared memory
            }
            if ((index + 1) < N_S)
            {
                sp1 = shdV[thp1 + 1]; //read level i+1 from shared memory
            }

            double Delta = _ak * (sp1 - sm1); //calculate delta=dV/dS
            double Gamma = _bk * (sp1 - 2 * s + sm1); //calculate gamma=dV/dS^2
            double Theta = _ck * _S[index] * _S[index] * Gamma - _intRate * _S[index] * Delta + _intRate * s; //calculate theta=dV/dt
            Vp[index] = V[index] - Theta * _dt; //calculate V^k+1 from theta
            if (index == 0) //is first row
            {
                //First row tridiagonal matrix - dense vector product
                Vp[index] = _dk * V[index];
            }
            if (index == N_S - 1) //is last row
            {
                //Last row tridiagonal matrix - dense vector product
                Vp[index] = 2 * Vp[N_S - 2] - Vp[N_S - 3];
            }    
           V[index] = Vp[index]; //Update V^k with V^k+1
           
           g.sync();  //synchronize full grid 
        }
    }
}

//function to calculate mean squared error
double mean(double *V, long long int N_S)
{
    double acum=0.0; //acumulator
    double mn=0.0; //mean 
    int inc=N_S/100;
    for (long long int i = 0; i < N_S; i+=inc)
    {
        acum+=V[i];

    }
    mn=acum/100; //Normalize
    return mn;
}

void solve_BS(int gs ,double vty ,double epn, int bs)
{
    // set problem parameters


    double Vol = vty, Int_Rate = 0.05, Expiration = epn, Strike = 100.0; //params BS
    int block_size = bs; 
    long long  int N_S = gs; //number of stock levels
    //scanf("%d", &N_S);
    double dS = 2 * Strike / N_S; //asset step
    double dt = 0.9f / (Vol * Vol) / (N_S * N_S); //time step
    long long int N_t = ceil(Expiration / dt) + 1;  //number of time steps
    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum
#define V(I,J) V_mat[I + N_S * J] 

    dt = Expiration / N_t; //time step for calculations

    t0 = clock(); // initialize setup
    // initialize stock price grid
    double* S = (double*)malloc(N_S * sizeof(*S)); //host stock price array
    double* V_mat = (double*)malloc(N_S * sizeof(*V_mat));  //host V^k array
    BS_DiffEq* pbs_diff_eq = (BS_DiffEq*)malloc(sizeof(*pbs_diff_eq)); //host struct BS params

    double* d_S; //device stock price array
    double* d_V; //device V^k array
    double* d_Vp; //device V^k+1 array
    BS_DiffEq* d_pbs_diff_eq; //device struct BS params

    
    checkCuda(cudaMalloc((void**)&d_S, N_S * sizeof(double)));
    checkCuda(cudaMalloc((void**)&d_V, N_S * sizeof(double)));
    checkCuda(cudaMalloc((void**)&d_Vp, N_S * sizeof(double)));
    checkCuda(cudaMalloc((void**)&d_pbs_diff_eq, sizeof(BS_DiffEq)));

    S[0] = 10.0f; //initial stock price
    for (int i = 1; i < N_S; i++) { // fill in stock value array
        S[i] = S[i - 1] + dS;
    }
    printf("%8.3f\n", S[N_S-1]);

    // set initial condition
    for (int i = 0; i < N_S; i++) { //initial V^k array
        V(i, 0) = fmaxf(S[i] - Strike, 0.0);  //payoff function
    }

    printf("%8.3f\n", V_mat[N_S-1]);
    // evaluate coefficients that are needed in finite difference approximation
    double ak = (1 / (2 * dS));
    double bk = (1 / dS / dS);
    double ck = (-0.5 * Vol * Vol);
    double dk = (1 - Int_Rate * dt);
    pbs_diff_eq->_ak = ak;
    pbs_diff_eq->_bk = bk;
    pbs_diff_eq->_ck = ck;
    pbs_diff_eq->_dk = dk;
    pbs_diff_eq->_dt = dt;
    pbs_diff_eq->_intRate = Int_Rate;
    pbs_diff_eq->_NS = N_S;
    pbs_diff_eq->_NT = N_t;

    int numBlocks = (N_S + block_size - 1) / block_size; //number of blocks
    checkCuda(cudaMemcpy(d_S, S, N_S * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_pbs_diff_eq, pbs_diff_eq, sizeof(BS_DiffEq), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_V, V_mat, N_S * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_Vp, 0, N_S * sizeof(double)));
    //printf("%lld\n",N_t);
    void* args[] = { (void*)&d_S, (void*)&d_V,  (void*)&d_Vp,(void*)&d_pbs_diff_eq };  //argument array for cooperative kernel
    t1 = clock(); //setup time
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    cudaLaunchCooperativeKernel((void*)grscheme, numBlocks, block_size, args, (block_size + 2) * sizeof(double)); //launch BS explicit FDM cooperative kerne
    cudaDeviceSynchronize(); //device synchronization after all calculations
    checkCuda(cudaMemcpy(V_mat, d_V, N_S  * sizeof(double), cudaMemcpyDeviceToHost)); //copy solution
    t2 = clock(); // computation time of full solution
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);

   printf("%lf\n", V_mat[N_S-1]);
   double mn=0.0;
   mn=mean(V_mat,N_S);
   printf("%lf\n",mn);
   printf("End\n");

    checkCuda(cudaFree(d_S));
    checkCuda(cudaFree(d_V));
    checkCuda(cudaFree(d_pbs_diff_eq));
    //#endif
    free(S);
    free(V_mat);
    free(pbs_diff_eq);
}

int main(int argc, char** argv)
{
    int gridsize = atoi(argv[1]); //stock levels
    double volatility=atof(argv[2]); //volatility
    double expiration=atof(argv[3]); //expiration
    int blksize=atoi(argv[4]); //block size
    //int tstsize=atoi(argv[5]);
    try {
        solve_BS(gridsize,volatility,expiration,blksize);
        //solve_BS(gridsize,volatility,expiration,blksize,tstsize);
    }
    catch (std::runtime_error err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}
