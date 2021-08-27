#include <math.h>
#include<vector>
#include <stdio.h>
#include<string.h> 
#include <iostream>
#include <time.h>
#include <omp.h>

//same comments as ExplicitOPMP except from the parallel region
static inline int min(int a, int b)
{
  return a < b ? a : b;
}
// define a struct for evaluating V(idx,k) as a function of V(idx,k-1)
struct BS_DiffEq { // struct for storing the BS params
    double _ak, _bk, _ck, _dk,_dt, _intRate;
    long long int _NTS; // number of asset levels 
    double* _S; // stock price array
    void operator()(double* V, double* Vp) const //function to calculate BS by explicit FDM
    {
        
        #pragma omp parallel default(none) shared(V,Vp) // omp parallel region 
        {
        int tid = omp_get_thread_num(); //thread id
        int nthread = omp_get_num_threads(); //number of threads
        int chunk = _NTS / nthread + ((_NTS % nthread) > tid); //chuk size
        int start = tid * (_NTS / nthread) + min(tid, _NTS % nthread); //start of block
        for (int idx = start; idx < start + chunk; idx++) { //gridsize for
        if (idx == 0)
        {
            Vp[idx] = _dk * V[idx];
        }
        else
        {
            if (idx == _NTS - 1)
            {
                Vp[idx] = 2 * Vp[_NTS - 2] - Vp[_NTS - 3];
            }
            else
            {
                double Delta = _ak * (V[idx + 1] - V[idx - 1]);
                double Gamma = _bk * (V[idx + 1] - 2 * V[idx] + V[idx - 1]);
                double Theta = _ck * _S[idx] * _S[idx] * Gamma - _intRate * _S[idx] * Delta + _intRate * V[idx];
                Vp[idx] = V[idx] - Theta * _dt;
            }
        }
        }
        }
    }
};

void solve_BS(long long int gs)
{
    // set problem parameters
    double Vol = 0.05, Int_Rate = 0.05, Expiration = 1.0, Strike = 100.0;
    long long int N_S = gs;
    double dS = 2 * Strike / N_S;
    double dt = 0.9f / (Vol * Vol) / (N_S * N_S);
    long long int N_t = ceil(Expiration / dt) + 1;
    dt = Expiration / N_t;
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;
    int nthread;
    int thread;
    // initialize stock price grid
    t0 = clock();
    double* S = (double*)malloc(N_S * sizeof(*S));
    S[0] = 10.0f;
    for (long long int i = 1; i < N_S; i++) {
        S[i] = S[i - 1] + dS;
    }
    printf("%lf\t", S[N_S - 1]);
    printf("\n");
    // create array for storing stock-price grid and solution
#define V(I,J) V_mat[I + N_S * J]    
    double* V_mat = (double*)malloc((N_S) * sizeof(*V_mat));
    double* V_res = (double*)malloc((N_S) * sizeof(*V_res));
    //memset(V_res, 0, (N_S) * sizeof(*V_res));
    // set initial condition
    for (long long int i = 0; i < N_S; i++) 
    {
        V(i, 0) = fmaxf(S[i] - Strike, 0.0);   
        V_res[i] = 0.0;
    }
    printf("%lf\t", V_mat[N_S-1]);
    printf("\n");

     // evaluate coefficients that are needed in finite difference approximation
    // INSERT CODE HERE
    double ak = (1 / (2 * dS));
    double bk = (1/dS/dS);
    double ck = (-0.5 * Vol * Vol);
    double dk = (1 - Int_Rate *dt);

    // instantiate a struct for storing coefficients and evaluating derivatives
    BS_DiffEq bs_diff_eq;
    // INSERT CODE HERE
    bs_diff_eq._ak = ak;
    bs_diff_eq._bk = bk;
    bs_diff_eq._ck = ck;
    bs_diff_eq._dk = dk;
    bs_diff_eq._dt = dt;
    bs_diff_eq._S = S;
    bs_diff_eq._intRate = Int_Rate;
    bs_diff_eq._NTS = N_S;
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);



    // loop over time
    for (long long int k = 1; k < N_t; k++) 
    {
        // loop over stock-price grid
        bs_diff_eq(V_mat, V_res);
       /* for(long long int i=0; i<N_S; i++)
        {
            printf("%lf\n", V_res[i]);
        }*/
        /*for (long long int i = 0; i < N_S; i++)
        {
            V_mat[i] = V_res[i];
        }*/
        memcpy(V_mat, V_res, N_S * sizeof(double));
        
    }

    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);

    /*for (int i =10; i < N_S; i++) {
        for (int k = 0; k < N_t - 1; k += 2) {
            printf("%8.3f", V(i, k));
        }
        printf("%8.3f\n", V(i, (N_t - 1)));
    }*/
    #pragma omp master
    printf("%lf\n", V_mat[N_S-1]);
}

int main(int argc, char** argv)
//int main()
{
    long long int gridsize = atoi(argv[1]);
    int nth=atoi(argv[2]);
    omp_set_num_threads(nth);
    try {
        solve_BS(gridsize);
    }
    catch (std::runtime_error err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}
