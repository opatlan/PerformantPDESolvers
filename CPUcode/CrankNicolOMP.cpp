#include <math.h>
#include<vector>
#include <iostream>
#include <time.h>
#include <omp.h>

double V_o = 0.0;
double V_lo = 0.0;
double V_f = 0.0;

struct BS_DiffEq { //struct for parameters of BS
    double _nu1, _nu2, _dt, _sigma, _intRate;
    long long int _NS, _NT;
};

void sp_matvec(double* nA, double* nB, double* nC, double *V, double *Vmul, long int N_S)
{
    #pragma omp parallel for default(none) shared(Vmul,V,nA,nB,nC,N_S) schedule(static) //parallel for loop to divide the stock levels between available threads 
    for(long long int i=0; i<N_S; i++) // loop through all stock levels (rows)
    {
        if (i == 0)  //is first row
        {
            //First row tridiagonal matrix - dense vector product
            Vmul[i] = nB[i] * V[i] + nC[i] * V[i + 1];
        }
        else
        {
            if(i==N_S-1) //is last row
            {
                //Last row tridiagonal matrix - dense vector product
                Vmul[i] = nA[i] * V[i - 1] + nB[i] * V[i];
            }
            else
            {
                //Middle row tridiagonal matrix - dense vector product
                Vmul[i] = nA[i] * V[i - 1] + nB[i] * V[i] + nC[i] * V[i + 1];
            }
        }
    }
}
void calc_coef(double* _S, double* A, double* B, double* C, double* nA, double* nB, double* nC, BS_DiffEq* pbs_diff_eq)
{
    double _nu1 = pbs_diff_eq->_nu1; // dt / dS^2
    double _nu2 = pbs_diff_eq->_nu2; // dt / dS
    double _dt = pbs_diff_eq->_dt;  // time step size
    double _volatility = pbs_diff_eq->_sigma;
    double _intRate = pbs_diff_eq->_intRate;
    long int N_S = pbs_diff_eq->_NS; //number of stock levels 
    long int N_t = pbs_diff_eq->_NT; //number of time levels 
    double a_k = 0.0;
    double b_k = 0.0;
    double c_k = 0.0;

    double Ak = 0.0;
    double Bk = 0.0;
    double Ck = 0.0;
    #pragma omp parallel for default(none) shared(a_k,b_k,c_k,Ak,Bk,Ck,A,B,C,nA,nB,nC,_S,_intRate,_volatility,_nu1,_nu2,N_S,_dt) schedule(static) //parallel for loop to divide the rows between available threads 
    for (long long int i=0;i<N_S;i++) // loop through all stock levels (rows)
    {
        a_k = 0.5 * pow(_volatility, 2) * pow(_S[i], 2);
        b_k = _intRate * _S[i];
        c_k = -_intRate;

        Ak = 0.5 * _nu1 * a_k - 0.25 * _nu2 * b_k;
        Bk = -_nu1 * a_k + 0.5 * _dt * c_k;
        Ck = 0.5 * _nu1 * a_k + 0.25 * _nu2 * b_k;

        A[i] = -Ak; //lhs A
        B[i] = 1.0 - Bk; //lhs B
        C[i] = -Ck; //lhs C

        nA[i] = Ak; //rhs A
        nB[i] = 1.0 + Bk; //rhs B
        nC[i] = Ck; //rhs C

        if (i == N_S - 1) // lower boundary condition
        {
            A[i] = -Ak + Ck;  //lower boundary condition for lhs A
            B[i] = 1.0 - Bk - 2 * Ck; //lower boundary condition for lhs B

            nA[i] = Ak - Ck; //lower boundary condition for rhs A
            nB[i] = 1.0 + Bk + 2 * Ck; //lower boundary condition for rhs B
        }
    }
}

double* CN_Serial(double* A, double* B, double* C, double* nA, double* nB, double* nC, double* hX, double* Vmul, long long int N_S,long long int N_t,double Int_Rate, double dt) //Thomas algorithm
{
    double* d = (double*)malloc(N_S * sizeof(*d)); //main diagonal
    double* u = (double*)malloc(N_S * sizeof(*u)); //upper diagonal
    double* l = (double*)malloc(N_S * sizeof(*l)); //lower diagonal 
    double* V = (double*)malloc(N_S * sizeof(*V)); 
    double* w = (double*)malloc(N_S * sizeof(*w));
    double* q = (double*)malloc(N_S * sizeof(*q));

    d[0] = B[0]; //initialize with first value of main diagonal

    // LU decomposition for the coefficient matrix, performed just once
    for (int i = 1; i < N_S; i++) //loop through all rows sequentially
    {
        u[i - 1] = C[i - 1];
        l[i] = A[i] / d[i - 1];
        d[i] = B[i] - l[i] * C[i - 1];
    }
    for (int j = 0; j < N_t; j++) //loop through all stock levels 
    {
        sp_matvec(nA, nB, nC,hX, q, N_S); //matrix vector product 

        w[0] = q[0]; //initial condition
        //Forward Reduction
        for (int i = 1; i < N_S; i++)
        {
            w[i] = q[i] - l[i] * w[i - 1];
        }
        V[N_S - 1] = w[N_S - 1] / d[N_S - 1];
        
        //Backward Substitution
        for (int i = N_S - 2; i>=0; i--)
        {
            V[i] = (w[i] - u[i] * V[i + 1]) / d[i];
        }

        V_o = V_lo * (1 - Int_Rate * dt); //calulate first value from upper boundary condition
        V_lo = V_o; //update first value for next iteration
        V_f = 2 * V[N_S - 1] - V[N_S - 2]; //calulate last value from lower boundary condition
        hX = V; //Update V^k with V^k+1
    }
    return V;
}

int main(int argc, char** argv)
//int main(void) 
{
    // Host problem definition
    int gridsize = atoi(argv[1]); //number of stock levels 
    double volatility = atof(argv[2]);
    double expiration = atof(argv[3]);
    int blksize = atoi(argv[4]);  //number of time levels

    double Vol = volatility, Int_Rate = 0.05, Expiration = expiration, Strike = 100.0;  //params of BS
    int block_size = blksize;
    long long int N_Sp = gridsize; // total number of asset levels
    long long int N_S = N_Sp - 2;  // total number of asset levels in FD matrices without boundary elements
    clock_t t0, t1, t2; //timing variables
    double t1sum = 0.0; //timing sum
    double t2sum = 0.0; //timing sum

    double dS = (2 * Strike) / N_Sp; //asset step

    long long int N_t = blksize;
    double dt = Expiration / N_t;  //time step

    t0 = clock();
    double* hX = (double*)malloc(N_Sp * sizeof(*hX)); // V^k array
    double* hY = (double*)malloc(N_S * sizeof(*hY)); // V^k+1 array

    double* S = (double*)malloc(N_Sp * sizeof(*S)); // stock array
    double* A = (double*)malloc(N_S * sizeof(*A)); // coefficient lhs A array
    double* B = (double*)malloc(N_S * sizeof(*B)); // coefficient lhs B array
    double* C = (double*)malloc(N_S * sizeof(*C)); // coefficient lhs C array
    double* Vmul = (double*)malloc(N_S * sizeof(*Vmul)); //vector product array
    double* nA = (double*)malloc(N_S * sizeof(*A));  // coefficient rhs A array
    double* nB = (double*)malloc(N_S * sizeof(*B)); //  coefficient rhs B array
    double* nC = (double*)malloc(N_S * sizeof(*C)); //  coefficient rhs C array
    double* Vres = (double*)malloc(N_S * sizeof(*Vres));
    BS_DiffEq* pbs_diff_eq = (BS_DiffEq*)malloc(sizeof(*pbs_diff_eq));  // params structure

    for (int i = 0; i < N_Sp; i++) // fill in stock value array
    {
        S[i] = i * dS;
    }

    printf("%lf\n", S[N_Sp - 1]);

    // set initial condition
    for (int i = 0; i < N_Sp; i++) //initial V^k array
    {
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

    V_lo = hX[0]; //initialize first value for first iteration

    t1 = clock(); //setup clock
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    calc_coef(&S[1], A, B, C, nA, nB, nC, pbs_diff_eq); //call calculate coefficient
    Vres = CN_Serial(A, B, C, nA, nB, nC, &hX[1], Vmul, N_S, N_t,Int_Rate,dt); // solve BS by Thomas algorithm
    t2 = clock(); // computation time of full solution
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);
    printf("\n");
    printf("%lf\n", V_f);
    printf("\n");
}
