#include <iostream>
#include <omp.h>
using namespace std;

double test(int i)
{
    int a = 0;
    for (int i =0; i< 10000000; i++){
        a = i + 1;
    }
}


int main() {
    double s_time = omp_get_wtime();
    double sum = 0;
    for (int i = 0; i < 100; ++i) {
        test(i);
    }
    cout << endl << "Time of loop without omp:" << omp_get_wtime() - s_time << endl;

    double s_omp_time = omp_get_wtime();
    sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        test(i);
    }
    cout << endl << "Time of omp loop:" << omp_get_wtime() - s_omp_time << endl;
    return 0;
}
