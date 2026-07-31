//the radix band engine device code, shared by the standalone kernels in
//radix.cu and the resident pair in persist2.cu so both compile the same
//gated bodies. schedules, swizzles and twiddle exponents match the
//reference in bench/radix_reference.py, see radix.cu for the story
#pragma once

constexpr int RX_LL = 32768;
constexpr int RX_TPB = 128;

struct RP {
    const float* zr; const float* zi;   //permuted packed spectrum [b, 32768]
    float* kr; float* ki;               //coefficient heap [b, n_k]
    const uint4* route;                 //16 byte taps, load order per band
    float* fr; float* fi;               //half spectrum planes [b, 32769]
    const float* gdv; const int* pos;   //dual window and scatter positions
    const float* rr; const float* ri;   //per length roots, synthesis sign
    const int* desc;                    //rows of m, ab, ob, ro
    long n_k;
};

constexpr int clog2(int r) { return r <= 1 ? 0 : 1 + clog2(r / 2); }

//two tap gather, v = A z1 + B conj z2 on the permuted packed spectrum.
//one 16 byte record per tap, b rebuilt from a and the window since
//re b = g - re a and im b = -im a in both the plain and conj cases
__device__ inline void route(const RP& p, long zb, int j,
                             float& vr, float& vi)
{
    const uint4 t = p.route[j];
    const int q1 = (int)(t.x & 0xffffu);
    const int q2 = (int)(t.x >> 16);
    const float a = __uint_as_float(t.y);
    const float c = __uint_as_float(t.z);
    const float g = __uint_as_float(t.w);
    const float e = g - a;
    const float f = -c;
    const float z1r = p.zr[zb + q1], z1i = p.zi[zb + q1];
    const float z2r = p.zr[zb + q2], z2i = p.zi[zb + q2];
    vr = a * z1r - c * z1i + e * z2r + f * z2i;
    vi = a * z1i + c * z1r + f * z2r - e * z2i;
}

//branch free radix r dit network across one subgroup of r lanes. lane
//ell arrives holding input digit brev(ell) and leaves holding natural
//output digit ell. butterfly roots come from the staged 32 point table
//in shared, every radix reads it at a stride since w_r(x) = w_32(32x/r)
template<int R, int SIGN>
__device__ inline void warp_dit(float& ar, float& ai, int ell,
                                unsigned mask, const float* s32r,
                                const float* s32i)
{
    #pragma unroll
    for (int s = 0; (1 << s) < R; s++) {
        const int h = 1 << s;
        const float pr = __shfl_xor_sync(mask, ar, h, R);
        const float pi = __shfl_xor_sync(mask, ai, h, R);
        const int isv = ell & h;
        const float vvr = isv ? ar : pr;
        const float vvi = isv ? ai : pi;
        const float uur = isv ? pr : ar;
        const float uui = isv ? pi : ai;
        const int e = (ell & (h - 1)) * (32 >> (s + 1));
        const float wr = s32r[e];
        const float wi = SIGN > 0 ? -s32i[e] : s32i[e];
        const float tr = vvr * wr - vvi * wi;
        const float ti = vvr * wi + vvi * wr;
        ar = isv ? uur - tr : uur + tr;
        ai = isv ? uui - ti : uui + ti;
    }
}

//stage the 32 point roots, the blob starts with them so the load is one
//coalesced warp read, values bit identical to the gated m roots
__device__ inline void stage32(float* s32r, float* s32i, const float* mr,
                               const float* mi)
{
    if (threadIdx.x < 32) {
        s32r[threadIdx.x] = mr[threadIdx.x];
        s32i[threadIdx.x] = mi[threadIdx.x];
    }
}

//inter level twiddle from the ordered tables, e is the warp read order
//position so consecutive lanes read consecutive floats
template<int SIGN>
__device__ inline void twiddle(float& ar, float& ai, const float* mr,
                               const float* mi, int e)
{
    const float wr = mr[e];
    const float wi = SIGN > 0 ? -mi[e] : mi[e];
    const float tr = ar * wr - ai * wi;
    ai = ar * wi + ai * wr;
    ar = tr;
}

//epilogue of one output point, mode 0 scales and stores into the heap
//at natural k, mode 1 reads the store ordered gain and position tables
//at j so the warp reads are contiguous, then scatters atomically
template<int M, int MODE>
__device__ inline void put(const RP& p, int b, int ab, int ob, int k,
                           int j, float vr, float vi)
{
    if (MODE == 0) {
        constexpr float s = 1.0f / (float)M;
        const long kb = (long)b * p.n_k;
        p.kr[kb + ob + k] = vr * s;
        p.ki[kb + ob + k] = vi * s;
    } else {
        const float g = p.gdv[ab + j];
        if (g != 0.0f) {
            const long fb = (long)b * (RX_LL + 1);
            const int pp = p.pos[ab + j];
            atomicAdd(p.fr + fb + pp, vr * g);
            atomicAdd(p.fi + fb + pp, vi * g);
        }
    }
}

//prologue of one input point, mode 0 routes through the load ordered
//tap tables at position pos, mode 1 reads the heap at natural n
template<int MODE>
__device__ inline void get(const RP& p, int b, int ab, int ob, int n,
                           int pos, float& vr, float& vi)
{
    if (MODE == 0) {
        route(p, (long)b * RX_LL, ab + pos, vr, vi);
    } else {
        const long kb = (long)b * p.n_k;
        vr = p.kr[kb + ob + n];
        vi = p.ki[kb + ob + n];
    }
}

//m 32, one warp
template<int SIGN, int MODE>
__device__ void band_one(const RP& p, int b, int ab, int ob,
                         const float* mr, const float* mi, float* sh)
{
    float* s32r = sh + 4096;
    float* s32i = s32r + 32;
    stage32(s32r, s32i, mr, mi);
    if (threadIdx.x < 32) {
        __syncwarp();
        const int lane = threadIdx.x;
        const int rb = (int)(__brev((unsigned)lane) >> 27);
        float vr, vi;
        get<MODE>(p, b, ab, ob, rb, lane, vr, vi);
        warp_dit<32, SIGN>(vr, vi, lane, 0xffffffffu, s32r, s32i);
        put<32, MODE>(p, b, ab, ob, lane, lane, vr, vi);
    }
}

//two level schedule, m = r0 r1, xor swizzled in place shared layout
template<int R0, int R1, int SHIFT, int SIGN, int MODE>
__device__ void band_two(const RP& p, int b, int ab, int ob,
                         const float* mr, const float* mi, float* sh)
{
    constexpr int M = R0 * R1;
    float* shr = sh;
    float* shi = sh + M;
    float* s32r = sh + 4096;
    float* s32i = s32r + 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    stage32(s32r, s32i, mr, mi);
    float* pbr = sh + 4160;
    float* pbi = pbr + 2048;
    if (MODE == 1) {
        //coalesced natural heap preload into the swizzled layout, in the
        //second buffer so level 0 reads it while writing the first, no
        //extra barrier and no register arrays
        const long kb = (long)b * p.n_k;
        for (int n = threadIdx.x; n < M; n += RX_TPB) {
            const int ph = n ^ (n >> SHIFT);
            pbr[ph] = p.kr[kb + ob + n];
            pbi[ph] = p.ki[kb + ob + n];
        }
    } else if (MODE == 2) {
        //single buffer preload, level 0 pulls operands over a barrier
        const long kb = (long)b * p.n_k;
        for (int n = threadIdx.x; n < M; n += RX_TPB) {
            const int ph = n ^ (n >> SHIFT);
            shr[ph] = p.kr[kb + ob + n];
            shi[ph] = p.ki[kb + ob + n];
        }
    }
    __syncthreads();
    //level 0, subgroup width r0, vector v is n1
    {
        const int sub = lane / R0;
        const int ell = lane & (R0 - 1);
        const unsigned mask = (unsigned)((1ull << R0) - 1ull) << (sub * R0);
        const int sg = warp * (32 / R0) + sub;
        const int rb = (int)(__brev((unsigned)ell) >> (32 - clog2(R0)));
        constexpr int NIT = M / RX_TPB > 0 ? M / RX_TPB : 1;
        float pvr[NIT], pvi[NIT];
        if (MODE == 2) {
            //the level 0 output overwrites the preloaded cells, all
            //reads must precede the first write
            int it = 0;
            for (int n1 = sg; n1 < R1; n1 += RX_TPB / R0, it++) {
                const int nn = n1 + R1 * rb;
                const int ph = nn ^ (nn >> SHIFT);
                pvr[it] = shr[ph];
                pvi[it] = shi[ph];
            }
            __syncthreads();
        }
        int itc = 0;
        for (int n1 = sg; n1 < R1; n1 += RX_TPB / R0, itc++) {
            float vr, vi;
            if (MODE == 0) {
                get<MODE>(p, b, ab, ob, n1 + R1 * rb,
                          n1 * R0 + ell, vr, vi);
            } else if (MODE == 2) {
                vr = pvr[itc];
                vi = pvi[itc];
            } else {
                const int nn = n1 + R1 * rb;
                const int ph = nn ^ (nn >> SHIFT);
                vr = pbr[ph];
                vi = pbi[ph];
            }
            warp_dit<R0, SIGN>(vr, vi, ell, mask, s32r, s32i);
            const int k0 = ell;
            twiddle<SIGN>(vr, vi, mr + 32, mi + 32, n1 * R0 + k0);
            const int lg = k0 * R1 + n1;
            const int ph = lg ^ (lg >> SHIFT);
            shr[ph] = vr;
            shi[ph] = vi;
        }
    }
    __syncthreads();
    //level 1, subgroup width r1, vector v is k0
    {
        const int sub = lane / R1;
        const int ell = lane & (R1 - 1);
        const unsigned mask = (unsigned)((1ull << R1) - 1ull) << (sub * R1);
        const int sg = warp * (32 / R1) + sub;
        const int rb = (int)(__brev((unsigned)ell) >> (32 - clog2(R1)));
        for (int k0 = sg; k0 < R0; k0 += RX_TPB / R1) {
            const int lg = k0 * R1 + rb;
            const int ph = lg ^ (lg >> SHIFT);
            float vr = shr[ph];
            float vi = shi[ph];
            warp_dit<R1, SIGN>(vr, vi, ell, mask, s32r, s32i);
            if (MODE == 0) {
                //write back in place, the subgroup owns exactly the set
                //it just read, natural drain follows the barrier
                const int lo = k0 * R1 + ell;
                const int po = lo ^ (lo >> SHIFT);
                shr[po] = vr;
                shi[po] = vi;
            } else {
                put<M, MODE>(p, b, ab, ob, k0 + R0 * ell,
                             k0 * R1 + ell, vr, vi);
            }
        }
    }
    if (MODE == 0) {
        //coalesced heap drain, each warp stores consecutive k
        __syncthreads();
        constexpr float sc = 1.0f / (float)M;
        const long kb = (long)b * p.n_k;
        for (int k = threadIdx.x; k < M; k += RX_TPB) {
            const int lg = (k % R0) * R1 + (k / R0);
            const int ph = lg ^ (lg >> SHIFT);
            p.kr[kb + ob + k] = shr[ph] * sc;
            p.ki[kb + ob + k] = shi[ph] * sc;
        }
    }
}

//conflict free bit linear shared map for the 2048 schedule, the high six
//bits store d and n2, the bank bits fold k0 against them
__device__ inline int phys2048(int k0, int n2, int d)
{
    const int b0 = ((k0 >> 0) ^ (d >> 0)) & 1;
    const int b1 = ((k0 >> 1) ^ (d >> 1)) & 1;
    const int b2 = ((k0 >> 2) ^ (d >> 2) ^ (n2 >> 2)) & 1;
    const int b3 = ((k0 >> 3) ^ (n2 >> 0)) & 1;
    const int b4 = ((k0 >> 4) ^ (n2 >> 1)) & 1;
    return 32 * (d + 8 * n2)
         + (b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4));
}

//three level schedule for m 2048 as 32 by 8 by 8, level one runs in
//place, each subgroup rewrites exactly its own eight cells
template<int SIGN, int MODE>
__device__ void band_three(const RP& p, int b, int ab, int ob,
                           const float* mr, const float* mi, float* sh)
{
    float* shr = sh;
    float* shi = sh + 2048;
    float* s32r = sh + 4096;
    float* s32i = s32r + 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    stage32(s32r, s32i, mr, mi);
    float* pbr = sh + 4160;
    float* pbi = pbr + 2048;
    if (MODE == 1) {
        //coalesced natural heap preload, n splits as n2 + 8 n1 + 64 n0,
        //second buffer so level 0 reads it while writing the first
        const long kb = (long)b * p.n_k;
        for (int n = threadIdx.x; n < 2048; n += RX_TPB) {
            const int ph = phys2048(n >> 6, n & 7, (n >> 3) & 7);
            pbr[ph] = p.kr[kb + ob + n];
            pbi[ph] = p.ki[kb + ob + n];
        }
    } else if (MODE == 2) {
        const long kb = (long)b * p.n_k;
        for (int n = threadIdx.x; n < 2048; n += RX_TPB) {
            const int ph = phys2048(n >> 6, n & 7, (n >> 3) & 7);
            shr[ph] = p.kr[kb + ob + n];
            shi[ph] = p.ki[kb + ob + n];
        }
    }
    __syncthreads();
    //level 0, radix 32, vector v is q = n2 + 8 n1
    {
        const int rb = (int)(__brev((unsigned)lane) >> 27);
        float pvr[16], pvi[16];
        if (MODE == 2) {
            int it = 0;
            for (int q = warp; q < 64; q += 4, it++) {
                const int ph = phys2048(rb, q & 7, q >> 3);
                pvr[it] = shr[ph];
                pvi[it] = shi[ph];
            }
            __syncthreads();
        }
        int itc = 0;
        for (int q = warp; q < 64; q += 4, itc++) {
            float vr, vi;
            if (MODE == 0) {
                get<MODE>(p, b, ab, ob, q + 64 * rb,
                          q * 32 + lane, vr, vi);
            } else if (MODE == 2) {
                vr = pvr[itc];
                vi = pvi[itc];
            } else {
                const int ph = phys2048(rb, q & 7, q >> 3);
                vr = pbr[ph];
                vi = pbi[ph];
            }
            warp_dit<32, SIGN>(vr, vi, lane, 0xffffffffu, s32r, s32i);
            const int k0 = lane;
            twiddle<SIGN>(vr, vi, mr + 32, mi + 32, q * 32 + k0);
            const int ph = phys2048(k0, q & 7, q >> 3);
            shr[ph] = vr;
            shi[ph] = vi;
        }
    }
    __syncthreads();
    //level 1, radix 8, vector v is 8 k0 + n2, in place
    {
        const int sub = lane >> 3;
        const int ell = lane & 7;
        const unsigned mask = 0xffu << (sub * 8);
        const int sg = warp * 4 + sub;
        const int rb = (int)(__brev((unsigned)ell) >> 29);
        for (int v = sg; v < 256; v += 16) {
            const int k0 = v >> 3;
            const int n2 = v & 7;
            float vr = shr[phys2048(k0, n2, rb)];
            float vi = shi[phys2048(k0, n2, rb)];
            warp_dit<8, SIGN>(vr, vi, ell, mask, s32r, s32i);
            twiddle<SIGN>(vr, vi, mr + 32 + 2048, mi + 32 + 2048,
                          v * 8 + ell);
            const int ph = phys2048(k0, n2, ell);
            shr[ph] = vr;
            shi[ph] = vi;
        }
    }
    __syncthreads();
    //level 2, radix 8, vector v is 8 k0 + k1
    {
        const int sub = lane >> 3;
        const int ell = lane & 7;
        const unsigned mask = 0xffu << (sub * 8);
        const int sg = warp * 4 + sub;
        const int rb = (int)(__brev((unsigned)ell) >> 29);
        for (int v = sg; v < 256; v += 16) {
            const int k0 = v >> 3;
            const int k1 = v & 7;
            float vr = shr[phys2048(k0, rb, k1)];
            float vi = shi[phys2048(k0, rb, k1)];
            warp_dit<8, SIGN>(vr, vi, ell, mask, s32r, s32i);
            if (MODE == 0) {
                shr[phys2048(k0, ell, k1)] = vr;
                shi[phys2048(k0, ell, k1)] = vi;
            } else {
                put<2048, MODE>(p, b, ab, ob,
                                k0 + 32 * k1 + 256 * ell,
                                (v << 3) + ell, vr, vi);
            }
        }
    }
    if (MODE == 0) {
        __syncthreads();
        constexpr float sc = 1.0f / 2048.0f;
        const long kb = (long)b * p.n_k;
        for (int k = threadIdx.x; k < 2048; k += RX_TPB) {
            const int ph = phys2048(k & 31, k >> 8, (k >> 5) & 7);
            p.kr[kb + ob + k] = shr[ph] * sc;
            p.ki[kb + ob + k] = shi[ph] * sc;
        }
    }
}


//dispatch one band and batch task, mode 0 analysis, mode 1 synthesis.
//shared needs 2 by 2048 data floats plus 64 root floats
template<int MODE>
__device__ inline void radix_band(const RP& p, int band, int b, float* sh)
{
    const int* d = p.desc + band * 4;
    const int m = d[0], ab = d[1], ob = d[2], ro = d[3];
    const float* mr = p.rr + ro;
    const float* mi = p.ri + ro;
    constexpr int SIGN = MODE == 0 ? 1 : -1;
    switch (m) {
    case 32:   band_one<SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 64:   band_two<8, 8, 3, SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 128:  band_two<16, 8, 4, SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 256:  band_two<16, 16, 4, SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 512:  band_two<32, 16, 4, SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 1024: band_two<32, 32, 5, SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    case 2048: band_three<SIGN, MODE>(p, b, ab, ob, mr, mi, sh); break;
    }
}
