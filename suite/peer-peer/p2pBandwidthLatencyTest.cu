#include <vector>

#include <helper_cuda.h>

using namespace std;

int numGPUs = 3;
int g[] = {0, 1, 5};

const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                                  \
    cudaError_t e=cudaGetLastError();                                       \
    if(e!=cudaSuccess) {                                                    \
        printf("Cuda failure %s:%d: '%s'\n",                                \
                __FILE__,__LINE__,cudaGetErrorString(e));                   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

__global__ void delay(int * null) {
  float j=threadIdx.x;
  for(int i=1;i<10000;i++)
      j=(j+1)/j;

  if(threadIdx.x == j) null[0] = j;
}

void checkP2Paccess(int numGPUs)
{
    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if (g[i]!=g[j])
            {
                cudaDeviceCanAccessPeer(&access,g[i],g[j]);
                printf("Device=%d %s Access Peer Device=%d\n", g[i],
                        access ? "CAN" : "CANNOT", g[j]);
            }
        }
    }
    printf("\n***NOTE: In case a device doesn't have P2P access to other one, "
           "it falls back to normal memcopy procedure.\nSo you can see lesser "
           "Bandwidth (GB/s) in those cases.\n\n");
}

void outputBandwidthMatrix(int numGPUs, bool p2p)
{
    int numElems=10000000;
    int repeat=10;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,g[i],g[j]);
                if (access)
                {
                    cudaDeviceEnablePeerAccess(g[j],0 );
                    cudaCheckError();
                }
            }

            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],g[i],buffers[j],g[j],
                                    sizeof(int)*numElems);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=numElems*sizeof(int)*repeat/(double)1e9;
            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if (p2p && access)
            {
                cudaDeviceDisablePeerAccess(g[j]);
                cudaCheckError();
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", g[j]);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

void outputBidirectionalBandwidthMatrix(int numGPUs, bool p2p)
{
    int numElems=10000000;
    int repeat=10;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream0(numGPUs);
    vector<cudaStream_t> stream1(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
        cudaStreamCreate(&stream0[d]);
        cudaCheckError();
        cudaStreamCreate(&stream1[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,g[i],g[j]);
                if (access)
                {
                    cudaSetDevice(g[i]);
                    cudaDeviceEnablePeerAccess(g[j],0);
                    cudaCheckError();
                    cudaSetDevice(g[j]);
                    cudaDeviceEnablePeerAccess(g[i],0);
                    cudaCheckError();
                }
            }

            cudaSetDevice(g[i]);
            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],g[i],buffers[j],g[j],
                                    sizeof(int)*numElems,stream0[i]);
                cudaMemcpyPeerAsync(buffers[j],g[j],buffers[i],g[i],
                                    sizeof(int)*numElems,stream1[i]);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;
            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if(p2p && access)
            {
                cudaSetDevice(g[i]);
                cudaDeviceDisablePeerAccess(g[j]);
                cudaSetDevice(g[j]);
                cudaDeviceDisablePeerAccess(g[i]);
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", g[j]);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
        cudaStreamDestroy(stream0[d]);
        cudaCheckError();
        cudaStreamDestroy(stream1[d]);
        cudaCheckError();
    }
}

void outputLatencyMatrix(int numGPUs, bool p2p)
{
    int repeat=10000;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaMalloc(&buffers[d],1);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> latencyMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,g[i],g[j]);
                if (access)
                {
                    cudaDeviceEnablePeerAccess(g[j],0);
                    cudaCheckError();
                }
            }
            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],g[i],buffers[j],g[j],1);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*numGPUs+j]=time_ms*1e3/repeat;
            if(p2p && access)
            {
                cudaDeviceDisablePeerAccess(g[j]);
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", g[j]);
    }

    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d ",g[i]);

        for (int j=0; j<numGPUs; j++)
        {
            printf("%6.02f ", latencyMatrix[i*numGPUs+j]);
        }

        printf("\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(g[d]);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

int main(int argc, char **argv)
{

    //cudaGetDeviceCount(&numGPUs);

    printf("[%s]\n", sSampleName);

    //output devices
    for (int i=0; i<numGPUs; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,g[i]);
        printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, "
               "pciDomainID:%x\n", g[i], prop.name, prop.pciBusID,
               prop.pciDeviceID, prop.pciDomainID);
    }

    checkP2Paccess(numGPUs);

    //Check peer-to-peer connectivity
    printf("P2P Connectivity Matrix\n");
    printf("     D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d", g[j]);
    }
    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d\t", g[i]);
        for (int j=0; j<numGPUs; j++)
        {
            if (g[i]!=g[j])
            {
               int access;
               cudaDeviceCanAccessPeer(&access,g[i],g[j]);
               printf("%6d", (access) ? 1 : 0);
            }
            else
            {
                printf("%6d", 1);
            }
        }
        printf("\n");
    }

    printf("Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, false);
    printf("Unidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, true);
    printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, false);
    printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, true);


    printf("P2P=Disabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, false);
    printf("P2P=Enabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true);

    exit(EXIT_SUCCESS);
}
