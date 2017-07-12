#include <stdio.h>
#include <helper_cuda.h>

#define cudamalloc(p, size) {                                               \
    cudaMalloc(&p, size);                                                   \
    if (p)                                                                  \
        printf("Allocated %zu bytes from %p \n", size, p);                  \
    else                                                                    \
        printf("Failed to allocate %zu bytes\n", size);                     \
}

int main()
{
    size_t step = 0x1000000;
    size_t size = step;
    static size_t best = 0;
    cudaError_t e;
    
    while (step > 0)
    {
        void *p;

        //Try allocating Memory
        cudamalloc(p, size);
        e = cudaGetLastError();

        //Check if successful
        if (e==cudaSuccess) {
            cudaFree(p);
            best = size;
        }
        else {
            step /= 0x10;
        }
        size += step;
    }

    void *p;
    //Confirm
    cudamalloc(p, best);
    e = cudaGetLastError();
    if (e==cudaSuccess)
    {
        printf("\nBest possible allocatable block size is %.4f GB\n",
               (float)best/1000000000.0);
        cudaFree(p);
        return 0;
    }
    else
        return 1;
}
