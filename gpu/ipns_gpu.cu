// GPU-перебор вэнити-IPNS имён (ed25519, libp2p-key, base36).
// Ядро считает: seed -> SHA-512 -> clamp -> a*B (фиксированная база, 8-битные окна) -> имя -> проверка слов.
// Кандидаты перепроверяет на CPU ipns-vanity (dalek), поэтому ошибка ядра не даст неверного ключа.
//
// Сборка: nvcc -O3 -arch=sm_120 ipns_gpu.cu -o ipns_gpu.exe
// Запуск: ipns_gpu.exe <слова через запятую> <каталог ключей> <путь к ipns-vanity.exe> [duty%] [maxTemp]
//         ipns_gpu.exe --selftest <N>   (печатает "seed pub" для сверки с dalek)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t i64;

#include "k512.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// ---------- поле GF(2^255-19), 8 лимбов по 32 бита, значение в [0, 2^256) ----------
struct fe { u32 v[8]; };

__device__ __forceinline__ void fe_mul(fe &r, const fe &a, const fe &b) {
  u32 t[16];
#pragma unroll
  for (int i = 0; i < 16; i++) t[i] = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    u64 carry = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      u64 cur = (u64)a.v[i] * b.v[j] + t[i + j] + carry;
      t[i + j] = (u32)cur;
      carry = cur >> 32;
    }
    t[i + 8] = (u32)carry;
  }
  u64 carry = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    u64 cur = (u64)t[i] + (u64)t[i + 8] * 38 + carry;
    r.v[i] = (u32)cur;
    carry = cur >> 32;
  }
  u64 cur = (u64)r.v[0] + carry * 38;
  r.v[0] = (u32)cur;
  carry = cur >> 32;
#pragma unroll
  for (int i = 1; i < 8; i++) {
    cur = (u64)r.v[i] + carry;
    r.v[i] = (u32)cur;
    carry = cur >> 32;
  }
  r.v[0] += (u32)carry * 38;  // при переносе значение уже мало, переполнения нет
}

__device__ __forceinline__ void fe_sq(fe &r, const fe &a) { fe_mul(r, a, a); }

__device__ __forceinline__ void fe_add(fe &r, const fe &a, const fe &b) {
  u64 carry = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    u64 cur = (u64)a.v[i] + b.v[i] + carry;
    r.v[i] = (u32)cur;
    carry = cur >> 32;
  }
  u64 cur = (u64)r.v[0] + carry * 38;
  r.v[0] = (u32)cur;
  carry = cur >> 32;
#pragma unroll
  for (int i = 1; i < 8; i++) {
    cur = (u64)r.v[i] + carry;
    r.v[i] = (u32)cur;
    carry = cur >> 32;
  }
  r.v[0] += (u32)carry * 38;
}

__device__ __forceinline__ void fe_sub(fe &r, const fe &a, const fe &b) {
  i64 borrow = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    i64 cur = (i64)a.v[i] - (i64)b.v[i] - borrow;
    r.v[i] = (u32)cur;
    borrow = cur < 0 ? 1 : 0;
  }
  // 2^256 ≡ 38: если был заём, истинное значение = r - 38 (mod p)
  i64 bw = borrow * 38;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    i64 cur = (i64)r.v[i] - bw;
    r.v[i] = (u32)cur;
    bw = cur < 0 ? 1 : 0;
  }
  r.v[0] -= (u32)bw * 38;
}

// Полная редукция в [0, p).
__device__ void fe_freeze(fe &a) {
  const u32 P[8] = {0xFFFFFFEDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu};
  for (int rep = 0; rep < 2; rep++) {
    u32 t[8];
    i64 borrow = 0;
    for (int i = 0; i < 8; i++) {
      i64 cur = (i64)a.v[i] - (i64)P[i] - borrow;
      t[i] = (u32)cur;
      borrow = cur < 0 ? 1 : 0;
    }
    if (!borrow) for (int i = 0; i < 8; i++) a.v[i] = t[i];
  }
}

__device__ __forceinline__ void fe_sqn(fe &r, const fe &a, int n) {
  fe_sq(r, a);
  for (int i = 1; i < n; i++) fe_sq(r, r);
}

// z^(p-2), цепочка из ref10.
__device__ void fe_invert(fe &out, const fe &z) {
  fe z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t, t2;
  fe_sq(z2, z);
  fe_sqn(t, z2, 2);
  fe_mul(z9, z, t);
  fe_mul(z11, z2, z9);
  fe_sq(t, z11);
  fe_mul(z2_5_0, z9, t);
  fe_sqn(t, z2_5_0, 5);
  fe_mul(z2_10_0, t, z2_5_0);
  fe_sqn(t, z2_10_0, 10);
  fe_mul(z2_20_0, t, z2_10_0);
  fe_sqn(t, z2_20_0, 20);
  fe_mul(t2, t, z2_20_0);
  fe_sqn(t, t2, 10);
  fe_mul(z2_50_0, t, z2_10_0);
  fe_sqn(t, z2_50_0, 50);
  fe_mul(z2_100_0, t, z2_50_0);
  fe_sqn(t, z2_100_0, 100);
  fe_mul(t2, t, z2_100_0);
  fe_sqn(t, t2, 50);
  fe_mul(t2, t, z2_50_0);
  fe_sqn(t, t2, 5);
  fe_mul(out, t, z11);
}

// ---------- кривая ----------
struct ge { fe X, Y, Z, T; };
struct pre { fe ypx, ymx, xy2d; };

__constant__ u32 D2_L[8] = {0x26b2f159u, 0xebd69b94u, 0x8283b156u, 0x00e0149au,
                            0xeef3d130u, 0x198e80f2u, 0x56dffce7u, 0x2406d9dcu};
__constant__ u32 BX_L[8] = {0x8f25d51au, 0xc9562d60u, 0x9525a7b2u, 0x692cc760u,
                            0xfdd6dc5cu, 0xc0a4e231u, 0xcd6e53feu, 0x216936d3u};
__constant__ u32 BY_L[8] = {0x66666658u, 0x66666666u, 0x66666666u, 0x66666666u,
                            0x66666666u, 0x66666666u, 0x66666666u, 0x66666666u};

__device__ __forceinline__ fe fe_const(const u32 *c) {
  fe r;
#pragma unroll
  for (int i = 0; i < 8; i++) r.v[i] = c[i];
  return r;
}
__device__ __forceinline__ fe fe_small(u32 x) {
  fe r;
  r.v[0] = x;
#pragma unroll
  for (int i = 1; i < 8; i++) r.v[i] = 0;
  return r;
}

// Сложение двух расширенных точек (RFC 8032, 5.1.4).
__device__ void ge_add(ge &r, const ge &p, const ge &q) {
  fe d2 = fe_const(D2_L);
  fe A, B, C, D, E, F, G, H, t1, t2;
  fe_sub(t1, p.Y, p.X); fe_sub(t2, q.Y, q.X); fe_mul(A, t1, t2);
  fe_add(t1, p.Y, p.X); fe_add(t2, q.Y, q.X); fe_mul(B, t1, t2);
  fe_mul(t1, p.T, q.T); fe_mul(C, t1, d2);
  fe_mul(t1, p.Z, q.Z); fe_add(D, t1, t1);
  fe_sub(E, B, A); fe_sub(F, D, C); fe_add(G, D, C); fe_add(H, B, A);
  fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}

// Удвоение (RFC 8032, 5.1.4).
__device__ void ge_dbl(ge &r, const ge &p) {
  fe A, B, C, H, E, G, F, t;
  fe_sq(A, p.X); fe_sq(B, p.Y); fe_sq(t, p.Z); fe_add(C, t, t);
  fe_add(H, A, B);
  fe_add(t, p.X, p.Y); fe_sq(t, t); fe_sub(E, H, t);
  fe_sub(G, A, B); fe_add(F, C, G);
  fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}

// Смешанное сложение с точкой в аффинной предвычисленной форме.
__device__ __forceinline__ void ge_madd(ge &r, const ge &p, const pre &q) {
  fe A, B, C, D, E, F, G, H, t1;
  fe_sub(t1, p.Y, p.X); fe_mul(A, t1, q.ymx);
  fe_add(t1, p.Y, p.X); fe_mul(B, t1, q.ypx);
  fe_mul(C, p.T, q.xy2d);
  fe_add(D, p.Z, p.Z);
  fe_sub(E, B, A); fe_sub(F, D, C); fe_add(G, D, C); fe_add(H, B, A);
  fe_mul(r.X, E, F); fe_mul(r.Y, G, H); fe_mul(r.T, E, H); fe_mul(r.Z, F, G);
}

// Таблица: g_table[w*255 + j] = (j+1) * 256^w * B
__device__ pre g_table[32 * 255];

__global__ void build_table() {
  int w = threadIdx.x;
  if (w >= 32) return;
  fe bx = fe_const(BX_L), by = fe_const(BY_L);
  ge P;
  P.X = bx; P.Y = by; P.Z = fe_small(1); fe_mul(P.T, bx, by);
  for (int s = 0; s < 8 * w; s++) ge_dbl(P, P);
  ge cur = P;
  fe d2 = fe_const(D2_L);
  for (int j = 0; j < 255; j++) {
    fe zi, x, y, t;
    fe_invert(zi, cur.Z);
    fe_mul(x, cur.X, zi);
    fe_mul(y, cur.Y, zi);
    pre &o = g_table[w * 255 + j];
    fe_add(o.ypx, y, x);
    fe_sub(o.ymx, y, x);
    fe_mul(t, x, y);
    fe_mul(o.xy2d, t, d2);
    ge nxt;
    ge_add(nxt, cur, P);
    cur = nxt;
  }
}

// ---------- SHA-512 одного блока (сообщение 32 байта) ----------
__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

// seed (4 слова big-endian) -> первые 32 байта дайджеста (как 4 слова big-endian)
__device__ void sha512_32(const u64 seed[4], u64 out[4]) {
  u64 W[16];
  W[0] = seed[0]; W[1] = seed[1]; W[2] = seed[2]; W[3] = seed[3];
  W[4] = 0x8000000000000000ULL;
#pragma unroll
  for (int i = 5; i < 15; i++) W[i] = 0;
  W[15] = 256;
  u64 a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL, c = 0x3c6ef372fe94f82bULL,
      d = 0xa54ff53a5f1d36f1ULL, e = 0x510e527fade682d1ULL, f = 0x9b05688c2b3e6c1fULL,
      g = 0x1f83d9abfb41bd6bULL, h = 0x5be0cd19137e2179ULL;
#pragma unroll
  for (int i = 0; i < 80; i++) {
    u64 w;
    if (i < 16) {
      w = W[i];
    } else {
      u64 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
      u64 s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
      u64 s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
      w = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
      W[i & 15] = w;
    }
    u64 S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
    u64 ch = (e & f) ^ (~e & g);
    u64 t1 = h + S1 + ch + K512[i] + w;
    u64 S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
    u64 maj = (a & b) ^ (a & c) ^ (b & c);
    u64 t2 = S0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  out[0] = a + 0x6a09e667f3bcc908ULL;
  out[1] = b + 0xbb67ae8584caa73bULL;
  out[2] = c + 0x3c6ef372fe94f82bULL;
  out[3] = d + 0xa54ff53a5f1d36f1ULL;
}

__device__ __forceinline__ u64 be64(u64 x) {
  u32 hi = (u32)(x >> 32), lo = (u32)x;
  return ((u64)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// ---------- слова для поиска ----------
#define MAX_WORDS 16
__constant__ u8 c_word[MAX_WORDS][16];
__constant__ int c_wlen[MAX_WORDS];
__constant__ int c_nwords;
__constant__ int c_wsuf[MAX_WORDS];  // 1 = слово в конце имени

__device__ u32 g_found_count;
__device__ u64 g_found_idx[256];
__device__ u32 g_found_word[256];

// u8-массив pub (32 байта) из точки
__device__ void pubkey_from_seed(const u64 nonce[3], u64 idx, u8 pub[32]) {
  // seed = nonce[24] || idx(LE, 8 байт); слова — big-endian загрузка из байтов seed
  u64 s[4];
  u64 sw[4] = {nonce[0], nonce[1], nonce[2], idx};  // сами 64-битные слова в порядке LE-памяти
  // байты seed в памяти = LE-представление каждого слова; SHA-512 читает их как big-endian
#pragma unroll
  for (int i = 0; i < 4; i++) s[i] = be64(sw[i]);
  u64 hs[4];
  sha512_32(s, hs);
  u8 a[32];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 x = hs[i];
#pragma unroll
    for (int k = 0; k < 8; k++) a[i * 8 + k] = (u8)(x >> (56 - 8 * k));
  }
  a[0] &= 248; a[31] &= 127; a[31] |= 64;

  ge acc;
  acc.X = fe_small(0); acc.Y = fe_small(1); acc.Z = fe_small(1); acc.T = fe_small(0);
  for (int w = 0; w < 32; w++) {
    u32 b = a[w];
    if (b) {
      pre q = g_table[w * 255 + (b - 1)];
      ge n;
      ge_madd(n, acc, q);
      acc = n;
    }
  }
  fe zi, x, y;
  fe_invert(zi, acc.Z);
  fe_mul(x, acc.X, zi);
  fe_mul(y, acc.Y, zi);
  fe_freeze(x);
  fe_freeze(y);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    pub[i * 4 + 0] = (u8)(y.v[i]);
    pub[i * 4 + 1] = (u8)(y.v[i] >> 8);
    pub[i * 4 + 2] = (u8)(y.v[i] >> 16);
    pub[i * 4 + 3] = (u8)(y.v[i] >> 24);
  }
  pub[31] |= (u8)((x.v[0] & 1) << 7);
}

// Цифры base36 (старшая первая) числа HEADER||pub. Возвращает число цифр.
__device__ int name_digits(const u8 pub[32], u8 dig[64]) {
  u32 w[10];
  w[0] = 0x01720024u; w[1] = 0x08011220u;
#pragma unroll
  for (int i = 0; i < 8; i++)
    w[2 + i] = ((u32)pub[4 * i] << 24) | ((u32)pub[4 * i + 1] << 16) | ((u32)pub[4 * i + 2] << 8) | pub[4 * i + 3];
  u8 low[64];
  int n = 0;
  int start = 0;
  while (start < 10) {
    u32 rem = 0;
    for (int i = start; i < 10; i++) {
      u64 cur = ((u64)rem << 32) | w[i];
      w[i] = (u32)(cur / 36);
      rem = (u32)(cur % 36);
    }
    low[n++] = (u8)rem;
    while (start < 10 && w[start] == 0) start++;
  }
  for (int i = 0; i < n; i++) dig[i] = low[n - 1 - i];
  return n;
}

__global__ void search(u64 nonce0, u64 nonce1, u64 nonce2, u64 base, u32 iters, u8 *dbg_pub) {
  u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 nonce[3] = {nonce0, nonce1, nonce2};
  for (u32 it = 0; it < iters; it++) {
    u64 idx = base + gid * iters + it;
    u8 pub[32];
    pubkey_from_seed(nonce, idx, pub);
    if (dbg_pub) {
      for (int i = 0; i < 32; i++) dbg_pub[idx * 32 + i] = pub[i];
      continue;
    }
    u8 dig[64];
    int n = name_digits(pub, dig);
    // позиция имени c ↔ цифра j = c-1; слово — на позиции 12 (j=11) или 13 (j=12)
    for (int wi = 0; wi < c_nwords; wi++) {
      int L = c_wlen[wi];
      // суффикс: единственная позиция — конец числа; иначе позиции 12 или 13 имени
      int lo = c_wsuf[wi] ? n - L : 11, hi = c_wsuf[wi] ? n - L : 12;
      for (int off = lo; off <= hi; off++) {
        if (off + L > n) continue;
        bool ok = true;
        for (int k = 0; k < L; k++)
          if (dig[off + k] != c_word[wi][k]) { ok = false; break; }
        if (ok) {
          u32 slot = atomicAdd(&g_found_count, 1u);
          if (slot < 256) { g_found_idx[slot] = idx; g_found_word[slot] = wi; }
        }
      }
    }
  }
}

// ---------- хост ----------
static void random_bytes(u8 *buf, size_t n) {
#ifdef _WIN32
  if (BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) { fprintf(stderr, "нет ГСЧ\n"); exit(1); }
#else
  FILE *f = fopen("/dev/urandom", "rb"); if (!f || fread(buf, 1, n, f) != n) exit(1); fclose(f);
#endif
}

static int char_val(char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'z' ? c - 'a' + 10 : -1); }

static int gpu_temp() {
  FILE *p = _popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", "r");
  if (!p) return 0;
  int t = 0;
  if (fscanf(p, "%d", &t) != 1) t = 0;
  _pclose(p);
  return t;
}

static std::string hex(const u8 *b, size_t n) {
  static const char *H = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; i++) { s += H[b[i] >> 4]; s += H[b[i] & 15]; }
  return s;
}

int main(int argc, char **argv) {
  if (argc >= 3 && !strcmp(argv[1], "--selftest")) {
    u64 N = strtoull(argv[2], 0, 10);
    build_table<<<1, 32>>>();
    CK(cudaDeviceSynchronize());
    u64 blocks = (N + 63) / 64, padded = blocks * 64;  // ядро считает по ключу на поток, буфер — с запасом
    u8 *d; CK(cudaMalloc(&d, padded * 32));
    search<<<(unsigned)blocks, 64>>>(0, 0, 0, 0, 1, d);
    CK(cudaDeviceSynchronize());
    std::vector<u8> h(padded * 32);
    CK(cudaMemcpy(h.data(), d, padded * 32, cudaMemcpyDeviceToHost));
    for (u64 i = 0; i < N; i++) {
      u8 seed[32] = {0};
      memcpy(seed + 24, &i, 8);
      printf("%s %s\n", hex(seed, 32).c_str(), hex(&h[i * 32], 32).c_str());
    }
    return 0;
  }
  if (argc < 4) {
    fprintf(stderr, "ipns_gpu <слова,через,запятую> <каталог> <ipns-vanity.exe> [duty%%=100] [maxTemp=85]\n");
    return 2;
  }
  std::vector<std::string> words;
  {
    std::string s = argv[1], cur;
    for (char c : s + ",") { if (c == ',') { if (!cur.empty()) words.push_back(cur); cur.clear(); } else cur += c; }
  }
  if (words.empty() || words.size() > MAX_WORDS) { fprintf(stderr, "от 1 до %d слов\n", MAX_WORDS); return 2; }
  u8 cw[MAX_WORDS][16] = {{0}};
  int wl[MAX_WORDS] = {0};
  int wsuf[MAX_WORDS] = {0};
  for (size_t i = 0; i < words.size(); i++) {
    // «слово$» — искать в конце имени
    std::string body = words[i];
    if (!body.empty() && body.back() == '$') { wsuf[i] = 1; body.pop_back(); }
    if (body.empty() || body.size() > 16) { fprintf(stderr, "слово пустое или длиннее 16\n"); return 2; }
    wl[i] = (int)body.size();
    for (int k = 0; k < wl[i]; k++) {
      int v = char_val(body[k]);
      if (v < 0) { fprintf(stderr, "допустимы только a-z и 0-9\n"); return 2; }
      cw[i][k] = (u8)v;
    }
  }
  int nw = (int)words.size();
  CK(cudaMemcpyToSymbol(c_word, cw, sizeof(cw)));
  CK(cudaMemcpyToSymbol(c_wlen, wl, sizeof(wl)));
  CK(cudaMemcpyToSymbol(c_nwords, &nw, sizeof(nw)));
  CK(cudaMemcpyToSymbol(c_wsuf, wsuf, sizeof(wsuf)));
  std::string outdir = argv[2], tool = argv[3];
  int duty = argc > 4 ? atoi(argv[4]) : 100;
  int maxTemp = argc > 5 ? atoi(argv[5]) : 85;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s, SM=%d\n", prop.name, prop.multiProcessorCount);
  build_table<<<1, 32>>>();
  CK(cudaDeviceSynchronize());
  printf("таблица готова\n");

  u8 nonce[24]; random_bytes(nonce, 24);
  u64 n0, n1, n2; memcpy(&n0, nonce, 8); memcpy(&n1, nonce + 8, 8); memcpy(&n2, nonce + 16, 8);

  const int TPB = 128;
  int blocks = prop.multiProcessorCount * 8;
  u32 iters = 8;
  u64 base = 0, total = 0;
  auto t0 = std::chrono::steady_clock::now(), tlast = t0, ttemp = t0;
  u64 lastTotal = 0;
  std::vector<bool> done(nw, false);
  while (true) {
    auto ls = std::chrono::steady_clock::now();
    search<<<blocks, TPB>>>(n0, n1, n2, base, iters, nullptr);
    CK(cudaDeviceSynchronize());
    u64 batch = (u64)blocks * TPB * iters;
    base += batch; total += batch;
    auto le = std::chrono::steady_clock::now();

    u32 cnt = 0;
    CK(cudaMemcpyFromSymbol(&cnt, g_found_count, sizeof(cnt)));
    if (cnt) {
      u64 idxs[256]; u32 ws[256];
      u32 m = cnt < 256 ? cnt : 256;
      CK(cudaMemcpyFromSymbol(idxs, g_found_idx, sizeof(u64) * m));
      CK(cudaMemcpyFromSymbol(ws, g_found_word, sizeof(u32) * m));
      u32 zero = 0; CK(cudaMemcpyToSymbol(g_found_count, &zero, sizeof(zero)));
      for (u32 i = 0; i < m; i++) {
        u8 seed[32]; memcpy(seed, nonce, 24); memcpy(seed + 24, &idxs[i], 8);
        // CPU сам проверяет ключ и пишет файл; слова не отмечаем «найденными» до его подтверждения
        // Внешние кавычки нужны cmd.exe: system() запускает `cmd /c "<строка>"`.
        std::string cmd = "\"\"" + tool + "\" fromseed " + hex(seed, 32) + " \"" + outdir + "\" " + words[ws[i]] + "\"";
        printf("кандидат слова '%s', проверяю на CPU…\n", words[ws[i]].c_str());
        fflush(stdout);
        int rc = system(cmd.c_str());
        if (rc != 0) printf("⚠ CPU-проверка не подтвердила кандидата (rc=%d)\n", rc);
      }
    }
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - tlast).count() >= 10) {
      double dt = std::chrono::duration<double>(now - tlast).count();
      printf("%.1f млн ключей/с, всего %.2f млрд\n", (total - lastTotal) / dt / 1e6, total / 1e9);
      fflush(stdout);
      tlast = now; lastTotal = total;
    }
    if (std::chrono::duration<double>(now - ttemp).count() >= 10) {
      ttemp = now;
      int t = gpu_temp();
      while (t >= maxTemp) {
        printf("GPU %d°C ≥ %d°C, пауза 20 с\n", t, maxTemp);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(20));
        t = gpu_temp();
      }
    }
    if (duty < 100) {
      double busy = std::chrono::duration<double>(le - ls).count();
      std::this_thread::sleep_for(std::chrono::duration<double>(busy * (100 - duty) / duty));
    }
  }
}
