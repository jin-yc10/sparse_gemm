namespace tc128 {

constexpr uint32_t LDA  = 128 + 8;
constexpr uint32_t LDA2 = LDA / 2;
constexpr uint32_t LDB =  128 + 8;
constexpr int stepk = 16;

// load global to register set a and b
#define LDG(a, b) \
{\
  ldg64( from_a_0, a[0], a[1] );  \
  ldg64( from_a_1, a[2], a[3] );  \
  const __half2* ptr1 = reinterpret_cast<const __half2 *>(from_b_0); \
  ldg128( ptr1, b[0], b[1], b[2], b[3] ); \
} \

#define STS(to_a, to_b, a, b) \
{ \
  __half2 u[4] = { \
    __half2( a[0].x, a[2].x ), __half2( a[0].y, a[2].y ),   \
    __half2( a[1].x, a[3].x ), __half2( a[1].y, a[3].y )};  \
  __half2* _step_to_a = reinterpret_cast<__half2 *>(to_a); \
  _step_to_a[0       ] = u[0]; _step_to_a[1 * LDA2] = u[1]; \
  _step_to_a[2 * LDA2] = u[2]; _step_to_a[3 * LDA2] = u[3]; \
  __half2* _step_to_b = reinterpret_cast<__half2 *>(to_b); \
  _step_to_b[0] = b[0]; _step_to_b[1] = b[1]; _step_to_b[2] = b[2]; _step_to_b[3] = b[3]; \
} \

#define WMMA(As, Bs) \
{ \
  half* curr_As = As + warp_r; \
  wmma::load_matrix_sync(frag_a[0], &curr_As[0 ], LDA); \
  wmma::load_matrix_sync(frag_a[1], &curr_As[16], LDA); \
  wmma::load_matrix_sync(frag_a[2], &curr_As[32], LDA); \
  wmma::load_matrix_sync(frag_a[3], &curr_As[48], LDA); \
  half* curr_Bs = Bs + warp_c;                          \
  wmma::load_matrix_sync(frag_b[0], &curr_Bs[0 ], LDB); \
  wmma::load_matrix_sync(frag_b[1], &curr_Bs[16], LDB); \
  _Pragma("unroll")                                     \
  for (int i = 0; i < 2; i++) {                         \
    _Pragma("unroll")                                   \
    for (int j = 0; j < 4; j++) {                       \
      wmma::mma_sync(frag_c[i][j], frag_a[j], frag_b[i], frag_c[i][j]); \
    } \
  } \
} \

__global__ __launch_bounds__(256, 2)
void sparse_hgemm_fuse_128x128_kernel(
  int m, int n, int k,    // sizes
  int n_hot, int fuse_op, // 0 if none, 1 if bias_relu_bn
  const int* ii,    // input indices   [n_hot]
  const int* oi,    // output indices  [n_hot]
  const __half2* a, // feat_in         [m, k]
  const __half* b,  // weight          [k, n]
  const __half2* p, // params          [3, n], 3 if bias_relu_bn
  __half2 *c        // feat_out        [m, n]
) {
  // dims are fixed on specific arch
  using fragA_t   = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major>;
  using fragB_t   = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
  using fragAcc_t = wmma::fragment<wmma::accumulator, 16, 16, 16, __half>;

  const __half* ha = reinterpret_cast<const __half*>(a);
  __half* hc = reinterpret_cast<__half*>(c);
  __shared__ __align__(16 * 1024) char smem[48 * 1024];

  __half* A1s = reinterpret_cast<__half *>(smem);            
  __half* B1s = reinterpret_cast<__half *>(smem + 8 * 1024);
  __half* A2s = reinterpret_cast<__half *>(smem + 16 * 1024);            
  __half* B2s = reinterpret_cast<__half *>(smem + 24 * 1024);
  int32_t* Iis = reinterpret_cast<int32_t*>(smem + 32 * 1024);
  int32_t* Ios = reinterpret_cast<int32_t*>(smem + 33 * 1024);

  const __half2 zero_h2{__half(0.0), __half(0.0)};

  const uint32_t ti = threadIdx.x;
  const uint32_t bc = blockIdx.x * 128; // block column
  const uint32_t br = blockIdx.y * 128; // block row
  if( br > n_hot ) return;

  const uint32_t warp_id = ti / 32; // 0 ~ 7
  const uint32_t warp_th = ti % 32; // 0 ~ 31
  const uint32_t warp_c = 32 * (warp_id % 4);  // 0 ~ 3
  const uint32_t warp_r = 64 * (warp_id / 4);  // 0 ~ 1

  // Block: [128, 128]
  // Warp:  [ 32,  64], 4 col, 2 row
  fragA_t   frag_a[4];
  fragB_t   frag_b[2];
  fragAcc_t frag_c[2][4];

  #pragma unroll
  for (int i = 0; i < 2; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      wmma::fill_fragment(frag_c[i][j], __half(0.0));
    }
  }

  int32_t* to_Iis = Iis + ti;
  int32_t* to_Ios = Ios + ti;
  if( ti < 128 ) {
    to_Iis[0] = __ldg(ii + br + ti);
    to_Ios[0] = __ldg(oi + br + ti);
  }
  __syncthreads();

  // each thread load 2x4 halfs / 2x2 half2s
  // step = 16, each step handle 16 x 128 sub section
  // 
  int32_t from_a_i0 = Iis[(ti / 4) * 2    ];
  int32_t from_a_i1 = Iis[(ti / 4) * 2 + 1];
  __half* to_a1 = A1s + (ti / 4) * 2 + (ti % 4) * 4 * LDA;
  __half* to_a2 = A2s + (ti / 4) * 2 + (ti % 4) * 4 * LDA;
  const __half2* from_a_0 = a + from_a_i0 * (k / 2) + (ti % 4) * 2;  
  const __half2* from_a_1 = a + from_a_i1 * (k / 2) + (ti % 4) * 2;
  if( 0 > from_a_i0 or from_a_i0 >= m ) { from_a_0 = a; }
  if( 0 > from_a_i1 or from_a_i1 >= m ) { from_a_1 = a; }

  // each thread load 1x8 halves ( ldg128 )
  const __half* from_b_0 = b + (ti / 16) * n + (ti % 16) * 8;
  __half* to_b1 = B1s + (ti / 16) * LDB + (ti % 16) * 8;
  __half* to_b2 = B2s + (ti / 16) * LDB + (ti % 16) * 8;
  
  __half2 ldgC_buf[8][4] = {zero_h2};
  int32_t col = (bc + warp_c) + (warp_th % 4) * 8;

  #pragma unroll
  for( int fi=0; fi<4; fi++ ) {
    #pragma unroll
    for( int fc=0; fc<2; fc++ ) {  
      int i = fi * 2 + fc;
      int32_t cur_r = i * 8 + warp_th / 4;
      int32_t to_c = Ios[warp_r + cur_r];
      int32_t to_gc = 0;
      if( to_c >= 0 and to_c < m ) {
        to_gc = to_c * n + col;
      }
      ldg128(reinterpret_cast<__half2*>(hc + to_gc),
             ldgC_buf[i][0], ldgC_buf[i][1], ldgC_buf[i][2], ldgC_buf[i][3]);
    }
  }

  // k = 128, stepk = 16
  __half2 a1[4], a2[4];
  __half2 b1[4], b2[4];
#if 0
  _Pragma("unroll")
  for(int s=0; s<8; s++) {
    LDG(a1, b1)
    STS(to_a1, to_b1, a1, b1)
    __syncthreads();
    from_a_0 += stepk / 2; from_a_1 += stepk / 2;
    from_b_0 += stepk * n;
    WMMA(A1s, B1s);
    __syncthreads();
  }
#else
  // loop init
  LDG(a1, b1)
  STS(to_a1, to_b1, a1, b1)
  __syncthreads();

  // loop 0
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a2, b2)
  STS(to_a2, to_b2, a2, b2)

  WMMA(A1s, B1s);
  __syncthreads();

  // loop 1
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;  
  LDG(a1, b1)
  STS(to_a1, to_b1, a1, b1)

  WMMA(A2s, B2s);
  __syncthreads();
  
  // loop 2
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a2, b2)
  STS(to_a2, to_b2, a2, b2)

  WMMA(A1s, B1s);
  __syncthreads();

  // loop 3
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a1, b1)
  STS(to_a1, to_b1, a1, b1)

  WMMA(A2s, B2s);
  __syncthreads();  

  // loop 4
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a2, b2)
  STS(to_a2, to_b2, a2, b2)

  WMMA(A1s, B1s);
  __syncthreads();

  // loop 5
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a1, b1)
  STS(to_a1, to_b1, a1, b1)

  WMMA(A2s, B2s);
  __syncthreads();

  // loop 6
  from_a_0 += stepk / 2; from_a_1 += stepk / 2;
  from_b_0 += stepk * n;
  LDG(a2, b2)
  STS(to_a2, to_b2, a2, b2)

  WMMA(A1s, B1s);
  __syncthreads();

  // loop 7
  WMMA(A2s, B2s);
  __syncthreads();
#endif

  // write c back to global, each warp write it's own part
  // use 32kb shared memory
  __half* buf_sh = A1s + warp_id * 32 * 64;

  const __half* hp = reinterpret_cast<const __half*>(p);
  // each warp_th -> 8 halves, loop 8 times
  buf_sh = A1s + warp_id * 32 * 64;

  __half2     buf_bias[4] = {zero_h2};
  __half2 buf_bn_scale[4] = {zero_h2};
  __half2 buf_bn_shift[4] = {zero_h2};
  if(fuse_op == 1) {
    ldg128(reinterpret_cast<const __half2*>(hp + col),
          buf_bias[0], buf_bias[1], buf_bias[2], buf_bias[3]);
    ldg128(reinterpret_cast<const __half2*>(hp + col + n),
          buf_bn_scale[0], buf_bn_scale[1], buf_bn_scale[2], buf_bn_scale[3]);
    ldg128(reinterpret_cast<const __half2*>(hp + col + 2 * n),
          buf_bn_shift[0], buf_bn_shift[1], buf_bn_shift[2], buf_bn_shift[3]);
  } 

  #pragma unroll
  for( int fi=0; fi<4; fi++ ) {
    // load 16 rows
    buf_sh = A1s + warp_id * 32 * 64 + fi * 32 * 16;
    wmma::store_matrix_sync(
      buf_sh, frag_c[0][fi], 32, wmma::mem_row_major
    );
    wmma::store_matrix_sync(
      buf_sh + 16, frag_c[1][fi], 32, wmma::mem_row_major
    );    

    #pragma unroll
    for( int fc=0; fc<2; fc++ ) {  
      int i = fi * 2 + fc;
      // each step, each warp write back 8 x 32 halves
      int32_t cur_r = i * 8 + warp_th / 4;
      int32_t to_c = Ios[warp_r + cur_r];
      int32_t to_gc = 0;
      if( to_c >= 0 and to_c < m ) {
        to_gc = to_c * n + col;
      }
      __half2* buf_sh2 = reinterpret_cast<__half2*>(buf_sh + (fc * 8 + warp_th / 4) * 32 + (warp_th % 4) * 8);
      // __half2 tmp[4];
      // lds128(buf_sh2, tmp[0], tmp[1], tmp[2], tmp[3]);

      #pragma unroll
      for (int k=0; k<4; ++k) {
        ldgC_buf[i][k] = __hadd2(buf_sh2[k], ldgC_buf[i][k]);
        if(fuse_op == 1) {
          // ref: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cu
          ldgC_buf[i][k] = __hadd2( ldgC_buf[i][k], buf_bias[k] );  // c += conv.bias
          ldgC_buf[i][k] = __hmul2( ldgC_buf[i][k], __hgt2( ldgC_buf[i][k], zero_h2 ) ); // c = relu(c)
          ldgC_buf[i][k] = __hfma2( ldgC_buf[i][k], buf_bn_scale[k], buf_bn_shift[k] );  // c = c * bn_scale + bn_shift
        }
      }

      if( to_c >= 0 and to_c < m ) {
        stg128(reinterpret_cast<__half2*>(hc + to_gc),
               ldgC_buf[i][0], ldgC_buf[i][1], ldgC_buf[i][2], ldgC_buf[i][3]);
      }
    }
  }
  return;
}  

}; // namespace tc128