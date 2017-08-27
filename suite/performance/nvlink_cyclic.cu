#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

using namespace std;

const int GPUs[] = {0,1,2,3,4}; // If left blank all available GPUs will be used.

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

void cyclic_copy(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d,
                 vector<cudaEvent_t> &start, vector<cudaEvent_t> &stop,
                 cudaStream_t stream[])
{
    float bw[g.size()], time_taken[g.size()];
    printf("\nCyclic Memory Transfers: 0->1->2->3...n->0\n");
    
    configure(size, buffer_s, buffer_d, start, stop, stream);
    for (int i=0; i<g.size(); i++)
    {
        cudaEventRecord(start[i], stream[i]);
        cudaMemcpyPeerAsync(buffer_s[i],g[i],buffer_d[(i+1)%g.size()],
                            g[(i+1)%g.size()], size, stream[i]);
        cudaEventRecord(stop[i], stream[i]);
    }

    for (int i=0; i<g.size(); i++)
    {
        cudaEventSynchronize(stop[i]);
        float time_ms;
        cudaEventElapsedTime(&time_ms,start[i],stop[i]);
        time_taken[i] = time_ms*1e3;
        bw[i] = (float)size*1000/time_ms/(1<<30);
    }

    printf("\nTime(s) spent in memcpy\n");
    for (int i=0; i<g.size(); i++)
        printf("GPU%d -> GPU%d:   %3.5f\n", g[i], g[(i+1)%g.size()],
                time_taken[i]/1e3);

    printf("\nBandwidth(Gbps) utilized in memcpy\n");
    for (int i=0; i<g.size(); i++)
        printf("GPU%d -> GPU%d:   %3.5f\n", g[i], g[(i+1)%g.size()], bw[i]);

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
    cyclic_copy(size, buffer_s, buffer_d, start, stop, stream);

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
