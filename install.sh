#!/bin/bash

pushd ./layers/convop_cython
bash compile.sh
popd

pushd ./layers/poolop_cython
bash compile.sh
popd

pushd ./layers/gemm_cython
bash compile.sh
popd
