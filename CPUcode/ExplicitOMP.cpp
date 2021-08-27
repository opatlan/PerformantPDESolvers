#include <math.h>
#include<vector>
#include <stdio.h>
#include<string.h> 
#include <iostream>
#include <time.h>
#include <omp.h>

struct BS_DiffEq {// struct for storing the BS params
    double _ak, _bk, _ck, _dk,_dt, _intRate;
    long long int _NTS;// number of asset levels 
    double* _S; // stock price array
    void operator()(double* V, double* Vp, int idx) const //function to calculate BS by explicit FDM
    {
        if (idx == 0) //is first value in V^k 
        {
            Vp[idx] = _dk * V[idx]; //Calculate upper boundary condition
        }
        else
        {
            if (idx == _NTS - 1) //is last value in V^k 
            {
                Vp[idx] = 2 * Vp[_NTS - 2] - Vp[_NTS - 3]; // calculate lower boundary conditions
            }
            else
            {
                double Delta = _ak * (V[idx + 1] - V[idx - 1]); //calculate delta=dV/dS
                double Gamma = _bk * (V[idx + 1] - 2 * V[idx] + V[idx - 1]); //calculate gamma=dV/dS^2
                double Theta = _ck * _S[idx] * _S[idx] * Gamma - _intRate * _S[idx] * Delta + _intRate * V[idx]; //calculate theta=dV/dt
                Vp[idx] = V[idx] - Theta * _dt; //calculate V^k+1 from theta
            }
        }
    }
};

void solve_BS(long long int gs)
{
    // set problem parameters
    double Vol = 0.05, Int_Rate = 0.05, Expiration = 1.0, Strike = 100.0; //params of BS
    long long int N_S = gs; // total number of asset levels
    double dS = 2 * Strike / N_S; //asset step
    double dt = 0.9f / (Vol * Vol) / (N_S * N_S); //time step
    long long int N_t = ceil(Expiration / dt) + 1; //number of time steps
    dt = Expiration / N_t; //time step for calculations
    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum

   
    t0 = clock();  // initialize setup
    double* S = (double*)malloc(N_S * sizeof(*S)); // stock price array
    S[0] = 10.0f; //initial stock price
    for (long long int i = 1; i < N_S; i++) { // fill in stock value array
        S[i] = S[i - 1] + dS;
    }
    printf("%lf\t", S[N_S - 1]);
    printf("\n");
    // create array for storing stock-price grid and solution
#define V(I,J) V_mat[I + N_S * J]    
    double* V_mat = (double*)malloc((N_S) * sizeof(*V_mat)); //V^k array
    double* V_res = (double*)malloc((N_S) * sizeof(*V_res)); //V^k+1 array
    // set initial condition
    for (long long int i = 0; i < N_S; i++) 
    { //initial V^k array
        V(i, 0) = fmaxf(S[i] - Strike, 0.0);   //payoff function
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
    for (long long int k = 1; k < N_t; k++)  //time step loop 
    {
        // loop over stock-price grid
        #pragma omp parallel for default(none), shared(bs_diff_eq, V_mat,V_res,N_S) schedule(static) //parallel for (covering stock levels)
        for (long long int i = 0; i < N_S; i++) 
        {
            bs_diff_eq(V_mat, V_res, i);//call BS explicit FDM solver
        }
        memcpy(V_mat, V_res, N_S * sizeof(double));//update V^k with V^k+1
        
    }

    t2 = clock(); // computation time of full solution
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);

    #pragma omp master
    printf("%lf\n", V_mat[N_S-1]);
}

int main(int argc, char** argv)
//int main()
{
    long long int gridsize = atoi(argv[1]); //number of stock levels 
    int nth=atoi(argv[2]); //number of omp threads
    omp_set_num_threads(nth);
    try {
        solve_BS(gridsize);
    }
    catch (std::runtime_error err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}
