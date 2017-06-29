#!/bin/bash

travis_retry wget https://developer.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64-deb
travis_retry sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
travis_retry sudo apt-get update -qq
travis_retry sudo apt-get install -y cuda
travis_retry sudo apt-get clean
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
