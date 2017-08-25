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
                    cudaStreamCreate(&stream[i*g.size()+j]);
                    cudaDeviceSynchronize();
                    cudaSetDevice(g[j]);
                    cudaDeviceEnablePeerAccess(g[i], 0);
                    cudaStreamCreate(&stream[i*g.size()+j]);
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

void burst_copy(size_t size, vector<int*> &buffer_s, vector<int*> &buffer_d,
                vector<cudaEvent_t> &start, vector<cudaEvent_t> &stop,
                cudaStream_t stream[])
{
    float bw[g.size()], time_taken[g.size()];
    printf("\nBurst copy: Every GPU is memcpy-ing to every other GPU\n");
    printf("%4d%8d%12s\n%4s%8s%12s\n%4s%8s%12s\n",
            1, 2,"n","^","^", "^","|","|","|");
    printf("3<-0->2 4<-1->3 ... %s<-%s->%s\n","q", "m", "p");
    printf("%4s%8s%12s\n%4s%8s%12s\n%4d%8d%12s\n\n",
            "|","|","|","v","v", "v",4,0,"r");
    configure(size, buffer_s, buffer_d, start, stop, stream);
    
    for (int i=0; i<g.size(); i++)
    {
        cudaEventRecord(start[i]);
        for (int j=0; j<g.size(); j++)
            if (i!=j)
                cudaMemcpyPeerAsync(buffer_s[i],g[i],buffer_d[j],g[j], size,
                                    stream[i*g.size()+j]);
        cudaEventRecord(stop[i]);
    }

    for (int i=0; i<g.size(); i++)
    {
        cudaEventSynchronize(stop[i]);
        float time_ms;
        cudaEventElapsedTime(&time_ms,start[i],stop[i]);
        time_taken[i] = time_ms*1e3;
        bw[i] = (float)size*1000/time_ms/(1<<30);
    }
    printf("\t\tTime(ms)\tBandwidth(Gbps)\n");
    for (int i=0; i<g.size(); i++)
        printf("GPU%d\t\t%6.2f\t\t%6.2f\n",g[i], time_taken[i], bw[i]);
}

void perf_analyze(size_t size)
{
    vector<int*> buffer_s(g.size());
    vector<int*> buffer_d(g.size());
    vector<cudaEvent_t> start(g.size());
    vector<cudaEvent_t> stop(g.size());
    cudaStream_t stream[g.size() * g.size()];

    configure(size, buffer_s, buffer_d, start, stop, stream);

    // Cyclic
    burst_copy(size, buffer_s, buffer_d, start, stop, stream);

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
