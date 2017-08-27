#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

using namespace std;

const int GPUs[] = {0,1,2,3,4}; // If left blank all available GPUs will be used.

vector<int> g(GPUs, GPUs + sizeof(GPUs)/sizeof(int));

void configure(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d)
{
    for (int i=0; i<g.size(); i++)
    {
        cudaSetDevice(g[i]);
        cudaMalloc(&buffer_s[i], size);
        cudaMalloc(&buffer_d[i], size);
        for (int j=0; j<g.size(); j++)
        {
            int access;
            if (i!=j)
            {
                cudaDeviceCanAccessPeer(&access, g[i], g[j]);
                if (access)
                {
                    cudaSetDevice(g[i]);
                    cudaDeviceEnablePeerAccess(g[j], 0);
                    cudaDeviceSynchronize();
                    cudaSetDevice(g[j]);
                    cudaDeviceEnablePeerAccess(g[i], 0);
                    cudaDeviceSynchronize();
                }
            }
        }
    }
}

void reset(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d)
{
    for (int i=0; i<g.size(); i++)
    {
        cudaSetDevice(g[i]);
        cudaFree(buffer_s[i]);
        cudaFree(buffer_d[i]);
        for (int j=0; j<g.size(); j++)
        {
            int access;
            if (i!=j)
            {
                cudaDeviceCanAccessPeer(&access, g[i], g[j]);
                if (access)
                {
                    cudaSetDevice(g[i]);
                    cudaDeviceDisablePeerAccess(g[j]);
                    cudaDeviceSynchronize();
                    cudaSetDevice(g[j]);
                    cudaDeviceDisablePeerAccess(g[i]);
                    cudaDeviceSynchronize();
                }
            }
        }
    }
}

void blocked_copy(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d)
{
    float time_taken[g.size()*g.size()], bw[g.size()*g.size()];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("\nBlocked Memory Transfers: Only one memory transfer at a time\n");

    for (int i=0; i<g.size(); i++)
    {
        for (int j=0; j<g.size(); j++)
        {
            if (i!=j)
            {
                printf("Copying from %d to %d\n", g[i], g[j]);
                cudaEventRecord(start);
                cudaMemcpyPeerAsync(buffer_s[i],g[i],buffer_d[j],g[j], size);
                cudaEventRecord(stop);
                cudaDeviceSynchronize();
                cudaEventSynchronize(stop);
                float time_ms=0.0;
                cudaEventElapsedTime(&time_ms, start, stop);
                time_taken[i*g.size()+j] = time_ms*1e3;
                bw[i*g.size()+j] = (float)size*1000/time_ms/(1<<30);
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Time(ms) spent in memcpy\n");
    printf("   D\\D");
    for (int j=0; j<g.size(); j++)
        printf("%10d ", g[j]);

    printf("\n");

    for (int i=0; i<g.size(); i++)
    {
        printf("%6d", g[i]);
        for (int j=0; j<g.size(); j++)
        {
            if (i==j)
                printf("%12.4f", 0.0);
            else
                printf("%12.4f", time_taken[i*g.size()+j]);
        }
        printf("\n");
    }

    printf("bandwidth(Gbps) utilized during memcpy\n");
    printf("   D\\D");
    for (int j=0; j<g.size(); j++)
        printf("%10d ", g[j]);

    printf("\n");

    for (int i=0; i<g.size(); i++)
    {
        printf("%6d", g[i]);
        for (int j=0; j<g.size(); j++)
        if (i==j)
            printf("%12.4f", 0.0);
        else
            printf("%12.4f", bw[i*g.size()+j]);
        printf("\n");
    }
}

void perf_analyze(size_t size)
{
    vector<int*> buffer_s(g.size());
    vector<int*> buffer_d(g.size());

    configure(size, buffer_s, buffer_d);

    // Cyclic
    blocked_copy(size, buffer_s, buffer_d);

    reset(size, buffer_s, buffer_d);
}

int main(int argc, char** argv)
{
    // NVLink D<->D performance
    size_t size = (1<<30);
    if (!g.size())
    {
        int n;
        printf("Using all 8 GPUs\n");
        cudaGetDeviceCount(&n);
        for (int i=0; i<n; i++)
            g.push_back(i);
    }
    //define size
    perf_analyze(size);

    return 0;
}
