#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

using namespace std;

const int GPUs[] = {0,5}; // If left blank all available GPUs will be used.

vector<int> g(GPUs, GPUs + sizeof(GPUs)/sizeof(int));

void configure(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d,
               vector<cudaEvent_t> &start, vector<cudaEvent_t> &stop,
               cudaStream_t stream[])
{
    for (int i=0; i<g.size(); i++)
    {
        cudaSetDevice(g[i]);
        cudaMalloc(&buffer_s[i], size);
        cudaMalloc(&buffer_d[i], size);
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
        cudaStreamCreate(&stream[i]);
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

void reset(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d,
               vector<cudaEvent_t> &start, vector<cudaEvent_t> &stop,
               cudaStream_t stream[])
{
    for (int i=0; i<g.size(); i++)
    {
        cudaSetDevice(g[i]);
        cudaFree(buffer_s[i]);
        cudaFree(buffer_d[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
        cudaStreamDestroy(stream[i]);
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

void cudaMemcpyPoolAsync(
        int* &dst, int  dstDevice, int* &src, int  srcDevice,
        size_t count, int route, size_t chunk)
{
    void* rbuff[2];
    cudaStream_t rstream[2];
    cudaSetDevice(srcDevice);
    cudaDeviceEnablePeerAccess(route, 0);
    cudaStreamCreate(&rstream[0]);
    cudaStreamCreate(&rstream[1]);
    cudaSetDevice(dstDevice);
    cudaStreamCreate(&rstream[0]);
    cudaStreamCreate(&rstream[1]);
    cudaSetDevice(route);
    cudaDeviceEnablePeerAccess(dstDevice, 0);
    cudaMalloc(&rbuff[0], chunk);
    cudaMalloc(&rbuff[1], chunk);
    cudaStreamCreate(&rstream[0]);
    cudaStreamCreate(&rstream[1]);
    int strm=1;

    for(int i=0; i<count; i+=chunk)
    {
        strm^=(0^1);
        cudaMemcpyPeerAsync(rbuff[strm], route, &src[i], srcDevice, chunk, rstream[strm]);
        cudaMemcpyPeerAsync(&dst[i], dstDevice, rbuff[strm], route, chunk, rstream[strm]);
    }
    cudaSetDevice(srcDevice);
    cudaDeviceDisablePeerAccess(route);
    cudaStreamDestroy(rstream[0]);
    cudaStreamDestroy(rstream[1]);
    cudaSetDevice(dstDevice);
    cudaStreamDestroy(rstream[0]);
    cudaStreamDestroy(rstream[1]);
    cudaSetDevice(route);
    cudaStreamDestroy(rstream[0]);
    cudaStreamDestroy(rstream[1]);
    cudaDeviceDisablePeerAccess(dstDevice);
    cudaFree(rbuff[0]);
    cudaFree(rbuff[1]);
}

void blocked_copy(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d,
                  vector<cudaEvent_t> &start, vector<cudaEvent_t> &stop,
                  cudaStream_t stream[])
{
    float time_taken[g.size()*g.size()], bw[g.size()*g.size()];
    printf("\nBlocked Memory Transfers: Only one memory transfer at a time\n");

    configure(size, buffer_s, buffer_d, start, stop, stream);
    for (int i=0; i<g.size(); i++)
    {
        for (int j=0; j<g.size(); j++)
        {
            if (i!=j)
            {
                printf("Copying from %d to %d\n", g[i], g[j]);
                cudaEventRecord(start[i]);
                //cudaMemcpyPeerAsync(buffer_s[i],g[i],buffer_d[j],g[j], size);
                //cudaMemcpyPoolAsync(buffer_s[i],g[i],buffer_d[j],g[j], size, 1,
                        //size);
                cudaEventRecord(stop[i]);
                cudaEventSynchronize(stop[i]);
                cudaDeviceSynchronize();
                float time_ms;
                cudaEventElapsedTime(&time_ms,start[i],stop[i]);
                time_taken[i*g.size()+j] = time_ms*1e3;
                bw[i*g.size()+j] = (float)size*1000/time_ms/(1<<30);
            }
        }
    }

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
                printf("%12.2f", 0.0);
            else
                printf("%12.2f", time_taken[i*g.size()+j]);
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
            printf("%12.2f", 0.0);
        else
            printf("%12.2f", bw[i*g.size()+j]);
        printf("\n");
    }
}

void perf_analyze(size_t size)
{
    vector<int*> buffer_s(g.size());
    vector<int*> buffer_d(g.size());
    vector<cudaEvent_t> start(g.size());
    vector<cudaEvent_t> stop(g.size());
    cudaStream_t stream[g.size()];

    configure(size, buffer_s, buffer_d, start, stop, stream);

    // Cyclic
    blocked_copy(size, buffer_s, buffer_d, start, stop, stream);

    reset(size, buffer_s, buffer_d, start, stop, stream);
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
