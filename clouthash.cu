#include <cstdint>
#include <memory>

#define MAX_GPUS 24
#define NONCES 3

static uint32_t *nonces_device[MAX_GPUS];
static __constant__ uint2 data_device[25];
static __constant__ uint32_t rc_x[24] = {0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001, 0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a, 0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080, 0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008};
static __constant__ uint32_t rc_y[24] = {0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000};
static __constant__ uint2 rotmask = {0b00100100100100100100100100100100, 0b10010010010010010010010010010000};
static __constant__ uint2 rotmask2 = {0b00001001001001001001001001001001, 0b00100100100100100100100100100100};
static __constant__ uint2 rot1finmask = {0b11010010010010010010010010010010, 0b01001001001001001001001001001011};

static __device__ __forceinline__
uint2 xor3(const uint2 a,const uint2 b,const uint2 c) {
  uint2 result;
  asm (
    "lop3.b32 %0, %2, %3, %4, 0x96; lop3.b32 %1, %5, %6, %7, 0x96;"
    : "=r" (result.x) "=r" (result.y)
    : "r" (a.x), "r" (b.x), "r" (c.x) "r" (a.y), "r" (b.y), "r" (c.y)
  );
  return result;
}

static __device__ __forceinline__
uint2 xor5(const uint2 a, const uint2 b, const uint2 c, const uint2 d, const uint2 e) {
	uint2 result;
  asm (
    "lop3.b32 %0, %2, %3, %4, 0x96; lop3.b32 %0, %0, %5, %6, 0x96; lop3.b32 %1, %7, %8, %9, 0x96; lop3.b32 %1, %1, %10, %11, 0x96;"
    : "=r" (result.x), "=r" (result.y)
    : "r" (a.x), "r" (b.x), "r" (c.x), "r" (d.x), "r" (e.x), "r" (a.y), "r" (b.y), "r" (c.y), "r" (d.y), "r" (e.y)
  );
  return result;
}

static __device__ __forceinline__
uint2 rotate1(const uint2 a)
{
	uint2 result;
	asm(
    "shf.l.wrap.b32 %0, %3, %2, 1; shf.l.wrap.b32 %1, %2, %3, 1;"
    : "=r" (result.x) "=r" (result.y)
    : "r" (a.x), "r" (a.y)
  );
	return result;
}

static __device__ __forceinline__
uint2 rotate(const uint2 a, const int offset)
{
	uint2 result;
	asm(
    "shf.l.wrap.b32 %0, %3, %2, %4; shf.l.wrap.b32 %1, %2, %3, %4;"
    : "=r" (result.x) "=r" (result.y)
    : "r" (offset < 32 ? a.x : a.y), "r" (offset < 32 ? a.y : a.x), "r" (offset)
  );
	return result;
}

static __device__ __forceinline__
uint2 chi(const uint2 a,const uint2 b,const uint2 c) {
  uint2 result;
  asm (
    "lop3.b32 %0, %2, %3, %4, 0xD2; lop3.b32 %1, %5, %6, %7, 0xD2;"
    : "=r" (result.x) "=r" (result.y)
    : "r" (a.x), "r" (b.x), "r" (c.x) "r" (a.y), "r" (b.y), "r" (c.y)
  );
  return result;
}

static __global__ __launch_bounds__(256)
void clouthash_device(uint32_t *nonces) {
  uint2 data[25], t1, t2, bc0, bc1, bc2, bc3, bc4, bc01, bc11, bc21, bc31, bc41;
  memcpy(data, data_device, 200);
  uint32_t nonce = 256 * blockIdx.x + threadIdx.x;
  data[12].x = nonce;
  #pragma unroll
  for (int i = 0; i < 23; i++) {
    bc0 = xor5(data[0], data[5], data[10], data[15], data[20]);
    bc1 = xor5(data[1], data[6], data[11], data[16], data[21]);
    bc2 = xor5(data[2], data[7], data[12], data[17], data[22]);
    bc3 = xor5(data[3], data[8], data[13], data[18], data[23]);
    bc4 = xor5(data[4], data[9], data[14], data[19], data[24]);
    bc01 = rotate1(bc0);
    bc11 = rotate1(bc1);
    bc21 = rotate1(bc2);
    bc31 = rotate1(bc3);
    bc41 = rotate1(bc4);
    t1 = xor3(data[1], bc0, bc21);
    data[0]  = xor3(data[0], bc4, bc11);
    data[1]  = rotate(xor3(data[6],  bc0, bc21), 44);
    data[6]  = rotate(xor3(data[9],  bc3, bc01), 20);
    data[9]  = rotate(xor3(data[22], bc1, bc31), 61);
    data[22] = rotate(xor3(data[14], bc3, bc01), 39);
    data[14] = rotate(xor3(data[20], bc4, bc11), 18);
    data[20] = rotate(xor3(data[2],  bc1, bc31), 62);
    data[2]  = rotate(xor3(data[12], bc1, bc31), 43);
    data[12] = rotate(xor3(data[13], bc2, bc41), 25);
    data[13] = rotate(xor3(data[19], bc3, bc01), 8);
    data[19] = rotate(xor3(data[23], bc2, bc41), 56);
    data[23] = rotate(xor3(data[15], bc4, bc11), 41);
    data[15] = rotate(xor3(data[4],  bc3, bc01), 27);
    data[4]  = rotate(xor3(data[24], bc3, bc01), 14);
    data[24] = rotate(xor3(data[21], bc0, bc21), 2);
    data[21] = rotate(xor3(data[8],  bc2, bc41), 55);
    data[8]  = rotate(xor3(data[16], bc0, bc21), 45);
    data[16] = rotate(xor3(data[5],  bc4, bc11), 36);
    data[5]  = rotate(xor3(data[3],  bc2, bc41), 28);
    data[3]  = rotate(xor3(data[18], bc2, bc41), 21);
    data[18] = rotate(xor3(data[17], bc1, bc31), 15);
    data[17] = rotate(xor3(data[11], bc0, bc21), 10);
    data[11] = rotate(xor3(data[7],  bc1, bc31), 6);
    data[7]  = rotate(xor3(data[10], bc4, bc11), 3);
    data[10] = rotate(t1, 1);
    #pragma unroll
    for(int j=0; j < 25; j += 5) {
      t1 = data[j];
      t2 = data[j + 1];
      data[j]     = chi(data[j], data[j + 1], data[j + 2]);
      data[j + 1] = chi(data[j + 1], data[j + 2], data[j + 3]);
      data[j + 2] = chi(data[j + 2], data[j + 3], data[j + 4]);
      data[j + 3] = chi(data[j + 3], data[j + 4], t1);
      data[j + 4] = chi(data[j + 4], t1, t2);
    }
    data[0].x = data[0].x ^ rc_x[i];
    data[0].y = data[0].y ^ rc_y[i];
    #pragma unroll
    for (int j = 0; j < 25; j++) {
      t1 = data[j];
      data[j].y = (rotmask2.x & (t1.y >> 2)) | (rot1finmask.x & t1.y) | (rotmask.x & ((t1.y << 2) | (t1.x >> 30)));
      data[j].x = (rotmask2.y & ((t1.y << 30) | (t1.x >> 2))) | (rot1finmask.y & t1.x) | (rotmask.y & (t1.x << 2));
    }
  }
  bc0 = xor5(data[0], data[5], data[10], data[15], data[20]);
  bc1 = xor5(data[1], data[6], data[11], data[16], data[21]);
  bc2 = xor5(data[2], data[7], data[12], data[17], data[22]);
  bc3 = xor5(data[3], data[8], data[13], data[18], data[23]);
  bc4 = xor5(data[4], data[9], data[14], data[19], data[24]);
  data[0] = xor3(data[0], bc4, rotate1(bc1));
  data[1] = rotate(xor3(data[6], bc0, rotate1(bc2)), 44);
  data[2] = rotate(xor3(data[12], bc1, rotate1(bc3)), 43);
  data[0] = chi(data[0], data[1], data[2]);
  t1.x = data[0].x ^ rc_x[23];
  t1.y = data[0].y ^ rc_y[23];
  data[0].x = (rotmask2.y & ((t1.y << 30) | (t1.x >> 2))) | (rot1finmask.y & t1.x) | (rotmask.y & (t1.x << 2));
  data[0].y = (rotmask2.x & (t1.y >> 2)) | (rot1finmask.x & t1.y) | (rotmask.x & ((t1.y << 2) | (t1.x >> 30)));
  if (data[0].x == 0xbba3b38c && (data[0].y & 0x000000f0) == 0x00000040) {
    uint32_t tmp = atomicExch(&nonces[0], nonce);
    if (tmp != 0xffffffff) tmp = atomicExch(&nonces[1], tmp);
    if (tmp != 0xffffffff) nonces[2] = tmp;
  }
}

__host__
void clouthash_init(uint8_t id) {
  cudaMalloc(&nonces_device[id], NONCES * sizeof(uint32_t));
}

__host__
void clouthash_update(uint64_t *data) {
  cudaMemcpyToSymbol(data_device, data, 25 * sizeof(uint2), 0, cudaMemcpyHostToDevice);
}

__host__
void clouthash_run(uint8_t id, uint32_t blocks, uint32_t *nonces) {
  cudaMemset(nonces_device[id], 0xff, NONCES * sizeof(uint32_t));
  dim3 block(256);
  dim3 grid(blocks);
  clouthash_device<<<grid, block>>>(nonces_device[id]);
  cudaMemcpy(nonces, nonces_device[id], NONCES * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
