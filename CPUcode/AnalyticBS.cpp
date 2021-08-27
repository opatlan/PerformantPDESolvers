#include <math.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


double normalCDF(double value)
{
   return 0.5 * erfc(-value * sqrt(2.0));//Function to evaluate the normal distribution
}

void solve_BS(double t, double value, double int_Rate,double vol,double strike, double T)//Close form solution of BS
{
    double d1=(1/(vol*sqrt(T-t)))*(log(value/strike)+(int_Rate+pow(vol,2)/2)*(T-t));//d+ in close form solution of BS
    double d2= d1-vol*sqrt(T-t);//d- in close form solution of BS
    double dis_factor = exp(-int_Rate*(T-t));//discount factor e^-rtau
    double opt_price=normalCDF(d1)*value-strike*dis_factor*normalCDF(d2);//Analytic Sol of BS
     printf("%lf\n",opt_price);

}
int main(int argc, char** argv)
//int main()
{
    double value = atof(argv[1]);
    try {
        solve_BS(0.0,value,0.05,0.05,100,1.0);
    }
    catch (std::runtime_error err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}
