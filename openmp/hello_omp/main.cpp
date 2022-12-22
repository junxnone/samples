#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    #pragma omp parallel
    {
        printf("hello in thread %d, total thread %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    return 0;
}
