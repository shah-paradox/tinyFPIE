#include <omp.h>
#include <likwid-marker.h>
#include <vector>
#include <iostream>

int main() {
    LIKWID_MARKER_INIT;

    #pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
        LIKWID_MARKER_REGISTER("test_region");
        
        LIKWID_MARKER_START("test_region");
        int sum = 0;
        #pragma omp for
        for (int i = 0; i < 1000000; i++) {
            sum += i;
        }
        LIKWID_MARKER_STOP("test_region");
    }

    LIKWID_MARKER_CLOSE;
    return 0;
}
