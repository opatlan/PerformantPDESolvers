#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>

struct BS_DiffEq { // struct for storing the BS params
    float _ak, _bk, _ck, _dk, _dt, _intRate;
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
void grscheme(float* _S, float* V, float* Vp, BS_DiffEq* pbs_diff_eq) // calculate BS by explicit FDM kernel
{

    int th = threadIdx.x; //thread number in block 
    int blk_sz = blockDim.x; //block size
    int blk_id = blockIdx.x;  //block number in grid
    int index = blk_sz * blk_id + th; // global index
    extern __shared__ float shdV[]; //dynamic shared memory array 

    float _ak = pbs_diff_eq->_ak;
    float _bk = pbs_diff_eq->_bk;
    float _ck = pbs_diff_eq->_ck;
    float _dk = pbs_diff_eq->_dk;
    float _dt = pbs_diff_eq->_dt;
    float _intRate = pbs_diff_eq->_intRate;
    long long int N_S = pbs_diff_eq->_NS; //stock levels
    long long int N_t = pbs_diff_eq->_NT; // time steps 
    int thp1 = th + 1;

    shdV[thp1] = V[index]; // each thread stores its current value
    shdV[0] = 0.0; // initialize lower neighbour
    shdV[blk_sz + 1] = 0.0; // initialize upper neighbour

    if (blk_id > 0) // if not in first block
    {
        //lower neighbour comes from lower neighbouring block
        shdV[0] = V[blk_id * blk_sz - 1];
    }


    if (blk_id < (gridDim.x - 1))  // if not in last block
    {
        //upper neighbour comes from upper neighbouring block
        shdV[blk_sz + 1] = V[(blk_id + 1) * blk_sz];
        
    }


    float s = shdV[thp1], sm1 = 0.0, sp1 = 0.0; //define FD stencil

    __syncthreads(); //wait for all threads in block to gather their values

 
    if ((index - 1) > 0)
    {
        sm1 = shdV[thp1 - 1]; //read level i-1 from shared memory
    }
    if ((index + 1) < N_S)
    {
        sp1 = shdV[thp1 + 1]; //read level i+1 from shared memory
    }

    float Delta = _ak * (sp1 - sm1);//(V[th + 1] - V[th - 1]); //calculate delta=dV/dS
    float Gamma = _bk * (sp1 - 2 * s + sm1); //V[th + 1] - 2 * V[th] + V[th - 1]); //calculate gamma=dV/dS^2
    float Theta = _ck * _S[index] * _S[index] * Gamma - _intRate * _S[index] * Delta + _intRate * s; //calculate theta=dV/dt
    Vp[index] = V[index] - Theta * _dt; //calculate V^k+1 from theta
    if (index == 0) //is first value in V^k 
    {
        Vp[index] = _dk * V[index]; //Calculate upper boundary condition
    } 
    if (index == N_S - 1) //is last value in V^k 
    {
        Vp[index] = 2 * Vp[N_S - 2] - Vp[N_S - 3]; //Calculate lower boundary condition
    }    
    __syncthreads(); //wait for all values to finish 
}

void solve_BS(int gs, int bs)
{
    // set problem parameters


    float Vol = 0.2, Int_Rate = 0.05, Expiration = 1.0, Strike = 100.0; //params BS
    const int block_size = bs; 
    int N_S=gs; //number of stock levels 
    //scanf("%d", &N_S);
    float dS = 2 * Strike / N_S; //asset step
    float dt = 0.9f / (Vol * Vol) / (N_S * N_S);  //time step
    long long int N_t = ceil(Expiration / dt) + 1; //number of time steps

    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum

    #define V(I,J) V_mat[I + N_S * J] 

    dt = Expiration / N_t; //time step for calculations

    t0 = clock(); // initialize setup
    // initialize stock price grid
    float* S = new float[N_S]; //host stock price array
    float* V_mat = new float[N_S * N_t]; //host V^k array
    BS_DiffEq* pbs_diff_eq = new BS_DiffEq; //host struct BS params

    float* d_S; //device stock price array
    float* d_V; //device V^k array
    float* d_Vp; //device V^k+1 array
    BS_DiffEq* d_pbs_diff_eq; //device struct BS params

    checkCuda(cudaMalloc((void**)&d_S, N_S * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_V, N_S * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_Vp, N_S * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_pbs_diff_eq, sizeof(BS_DiffEq)));

    S[0] = 10.0f; //initial stock price
    for (int i = 1; i < N_S; i++) { // fill in stock value array
        S[i] = S[i - 1] + dS;
    }

    for (int i = 0; i < N_S; i++) { //initial V^k array
        V(i, 0) = fmaxf(S[i] - Strike, 0.0); //payoff function
    }

    // evaluate coefficients that are needed in finite difference approximation
    float ak = (1 / (2 * dS));
    float bk = (1 / dS / dS);
    float ck = (-0.5 * Vol * Vol);
    float dk = (1 - Int_Rate * dt);

    pbs_diff_eq->_ak = ak;
    pbs_diff_eq->_bk = bk;
    pbs_diff_eq->_ck = ck;
    pbs_diff_eq->_dk = dk;
    pbs_diff_eq->_dt = dt;
    pbs_diff_eq->_intRate = Int_Rate;
    pbs_diff_eq->_NS = N_S;
    pbs_diff_eq->_NT = N_t;

    int numBlocks = (N_S + block_size - 1) / block_size; //number of blocks 

    checkCuda(cudaMemcpy(d_S, S, N_S * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_pbs_diff_eq, pbs_diff_eq, sizeof(BS_DiffEq), cudaMemcpyHostToDevice));

    t1 = clock(); //setup time 
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    for (long long int k = 1; k < N_t; k++) //out kernel time step loop
    {
        checkCuda(cudaMemcpy(d_V, &V(0, (k - 1)), N_S * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(d_Vp, 0, N_S * sizeof(float)));
        grscheme <<<numBlocks, block_size, (block_size + 2)*sizeof(float)>>> (d_S, d_V, d_Vp, d_pbs_diff_eq); //dynamic shared memory kernel 
        checkCuda(cudaMemcpy(&V(0, k), d_Vp, N_S * sizeof(float), cudaMemcpyDeviceToHost)); //update V^k with V^k+1
    }
    cudaDeviceSynchronize(); //device synchronization after all calculations

    t2 = clock(); // computation time of full solution
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);


    for (long long int k = 0; k < N_t - 1; k += ceil(N_t / 10) + 1)
    {
        printf("%8.3f", V((N_S - 1), k));
    }

    checkCuda(cudaFree(d_S));
    checkCuda(cudaFree(d_V));
    checkCuda(cudaFree(d_Vp));
    checkCuda(cudaFree(d_pbs_diff_eq));
    free(S);
    free(V_mat);
    free(pbs_diff_eq);
}

int main(int argc, char **argv)
{
    int gridsize=atoi(argv[1]); //stock levels
    int blksize=atoi(argv[2]); //block size
    try {
        solve_BS(gridsize,blksize);
    }
    catch (std::runtime_error err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}
