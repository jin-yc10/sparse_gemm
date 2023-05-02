A simple gemm kernel for sparse convolution.
Mainly an implicit GEMM cuda kernel with naive tensor-core.
Used following tricks to improve overall runtime
1. tensor-core
2. float16 arithmetic
3. and half2 intrinsic ( hfma2, hmul2, hadd2 )
4. software pipelining
5. combined memory access ( ldg128, stg128 ). Note that for A100, we should consider using ldgsts intrinsic
