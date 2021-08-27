#include <math.h>
#include<vector>
#include <iostream>
#include <time.h>

//Same comments as openmp version excepting the parallel for definitions
double V_o = 0.0;
double V_lo = 0.0;
double V_f = 0.0;

struct BS_DiffEq {
    // INSERT CODE HERE
    double _nu1, _nu2, _dt, _sigma, _intRate;
    long long int _NS, _NT;
};

void calc_coef(double* _S, double* A, double* B, double* C, BS_DiffEq* pbs_diff_eq)
{
    double _nu1 = pbs_diff_eq->_nu1;
    double _nu2 = pbs_diff_eq->_nu2;
    double _dt = pbs_diff_eq->_dt;
    double _volatility = pbs_diff_eq->_sigma;
    double _intRate = pbs_diff_eq->_intRate;
    long int N_S = pbs_diff_eq->_NS;
    long int N_t = pbs_diff_eq->_NT;
    double a_k = 0.0;
    double b_k = 0.0;
    double c_k = 0.0;

    double Ak = 0.0;
    double Bk = 0.0;
    double Ck = 0.0;
    for (long long int i = 0; i < N_S; i++)
    {
        a_k = 0.5 * pow(_volatility, 2) * pow(_S[i], 2);
        b_k = _intRate * _S[i];
        c_k = -_intRate;

        Ak = -_nu1 * a_k + 0.5 * _nu2 * b_k;
        Bk = 2*_nu1 * a_k - _dt * c_k;
        Ck = -_nu1 * a_k - 0.5 * _nu2 * b_k;

        A[i] = Ak;
        B[i] = 1.0 + Bk;
        C[i] = Ck;

        if (i == N_S - 1)
        {
            A[i] = Ak - Ck;
            B[i] = 1.0 + Bk + 2 * Ck;
        }
    }
}

double* CN_Serial(double* A, double* B, double* C, double* hX, double* Vmul, long long int N_S, double Int_Rate, double dt)
{
    double* d = (double*)malloc(N_S * sizeof(*d));
    double* u = (double*)malloc(N_S * sizeof(*u));
    double* l = (double*)malloc(N_S * sizeof(*l));
    double* V = (double*)malloc(N_S * sizeof(*V));
    double* w = (double*)malloc(N_S * sizeof(*w));
    double* q = (double*)malloc(N_S * sizeof(*q));

    q= hX;

    d[0] = B[0];

    for (int i = 1; i < N_S; i++)
    {
        u[i - 1] = C[i - 1];
        l[i] = A[i] / d[i - 1];
        d[i] = B[i] - l[i] * C[i - 1];
    }
    for (int j = 0; j < N_S + 2; j++)
    {
        w[0] = q[0];
        for (int i = 1; i < N_S; i++)
        {
            w[i] = q[i] - l[i] * w[i - 1];
        }
        V[N_S - 1] = w[N_S - 1] / d[N_S - 1];

        for (int i = N_S - 2; i >= 0; i--)
        {
            V[i] = (w[i] - u[i] * V[i + 1]) / d[i];
        }

        V_o = V_lo * (1 - Int_Rate * dt);
        V_lo = V_o;
        V_f = 2 * V[N_S - 1] - V[N_S - 2];
        q = V;
    }
    return V;
}

int main(int argc, char** argv)
//int main(void) 
{
    // Host problem definition
    int gridsize = atoi(argv[1]);
    double volatility = atof(argv[2]);
    double expiration = atof(argv[3]);
    int blksize = atoi(argv[4]);

    double Vol = volatility, Int_Rate = 0.05, Expiration = expiration, Strike = 100.0;
    //double Vol = 0.2, Int_Rate = 0.05, Expiration = 1.0, Strike = 100.0;
    int block_size = blksize;
    long long int N_Sp = gridsize;
    long long int N_S = N_Sp - 2;


    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;
    //int block_size = blksize;
    //long long int N_S = gridsize;

    double dS = (2 * Strike) / N_Sp;
    //double dt = 0.9f / (Vol * Vol) / (N_Sp * N_Sp);
    //long long int N_t = ceil(Expiration / dt) + 1;
    long long int N_t = N_Sp;
    double dt = Expiration / N_t;

    t0 = clock();
    double* hX = (double*)malloc(N_Sp * sizeof(*hX));
    double* hY = (double*)malloc(N_S * sizeof(*hY));

    double* S = (double*)malloc(N_Sp * sizeof(*S));
    double* A = (double*)malloc(N_S * sizeof(*A));
    double* B = (double*)malloc(N_S * sizeof(*B));
    double* C = (double*)malloc(N_S * sizeof(*C));
    double* Vmul = (double*)malloc(N_S * sizeof(*Vmul));
    double* Vres = (double*)malloc(N_S * sizeof(*Vres));
    BS_DiffEq* pbs_diff_eq = (BS_DiffEq*)malloc(sizeof(*pbs_diff_eq));

    for (int i = 0; i < N_Sp; i++)
    {
        S[i] = i * dS;
    }

    printf("%8.3f\n", S[N_Sp - 1]);

    for (int i = 0; i < N_Sp; i++)
    {
        hX[i] = fmaxf(S[i] - Strike, 0.0);
    }

    printf("%8.3f\n", hX[N_Sp - 1]);

    double nu1 = (dt / (dS * dS));
    double nu2 = (dt / dS);

    pbs_diff_eq->_nu1 = nu1;
    pbs_diff_eq->_nu2 = nu2;
    pbs_diff_eq->_dt = dt;
    pbs_diff_eq->_sigma = Vol;
    pbs_diff_eq->_intRate = Int_Rate;
    pbs_diff_eq->_NS = N_S;
    pbs_diff_eq->_NT = N_t;

    V_lo = hX[0];

    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    calc_coef(&S[1], A, B, C, pbs_diff_eq);
    Vres = CN_Serial(A, B, C,&hX[1], Vmul, N_S, Int_Rate, dt);
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computing took %f seconds.  Finish to compute\n", t2sum);
    printf("\n");
    printf("%8.3f\n", V_f);
    printf("\n");
}
