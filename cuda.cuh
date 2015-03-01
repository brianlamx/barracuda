/*

   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: barracuda.cu  - CUDA alignment and samse kernels

   Copyright (C) 2012, University of Cambridge Metabolic Research Labs.
   Contributers: Petr Klus, Dag Lyberg, Simon Lam and Brian Lam

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 3
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   This program is based on a modified version of BWA
   File Creation date: 2012.6.8

*/

/* (0.7.0) beta: $Revision: 1.105 $
  27 Feb 2015 WBL scache_global_bwt failed to speed cuda2.cuh, instead
let cuda_inexact_match_caller use old code without mycache 
  26 Feb 2015 WBL cuda_inexact_match_caller uses up to 15(rather than 7) closelyplaced words so should work fine with mycache but doesnt
  for cuda_inexact_match_caller update bwt_cuda_match_exact.cuh to try passing d_mycache to bwt_cuda_match_exact
  25 Feb 2015 WBL r1.55-62 scache_global_bwt 6 Jan 2015 WBL Add cache_global_bwt scache_global_bwt
skip r1.63(cache_threads) r1.64-67(kl_split, kl_par) r1.68-75(occ_par)
skip r1.76-r1.90(mostly debugging occ_par)
  21 Feb 2015 WBLtry adding huge debug to each kernel launch from r1.89+(r1.92)
  mycache for global_bwt only from r1.14-r1.25 
  redo r1.26-r1.28, skip r1.29, redo r1.30-r1.31 (use assert), skip r1.32-r1.38
  for r1.39 non-working hack with DEBUG_LEVEL, skip r1.40-42, redo r1.43
  skip r1.44-46, redo 1.47,
  skip r1.48-51 (threads_per_sequence, many_blocks, simplify read_char, __shfl) maybe later
  skip r1.52 r1.53(direct_global_bwt) r1.54(cache_global_bwt)
  18 Dec 2014 WBL Add mycache4
  14 Dec 2014 WBL Add direct_sequence, direct_index
  10 Dec 2014 WBL r1.6--1.7 tried mycache_t, REVERT r1.5 try __ldg for sm3.5
   7 Dec 2014 WBL printf representative data accesses
remove option USE_SIMON_OCC4
   4 Dec 2014 WBL Extract aln device code to new separate file cuda.cuh
  25 Nov 2014 WBL Re-enable cuda_find_exact_matches changes. Note where sequence matches exactly once no longer report other potential matches
  21 Nov 2014 WBL disable cuda_find_exact_matches changes and add <<<>>> logging comments
                  Add header to text .sai file
  19 Nov 2014 WBL merge text and binary output, ie add stdout_aln_head stdout_barracuda_aln1
                  Explicitly clear unused parts of alignment records in binary .sai output file
  13 Nov 2014 WBL try re-enabling cuda_find_exact_matches
  13 Nov 2014 WBL ensure check status of all host cuda calls
  Ensure all kernels followed by cudaDeviceSynchronize so they can report asynchronous errors
*/

//CUDA DEVICE CODE STARTING FROM THIS LINE
/////////////////////////////////////////////////////////////////////////////

//#define IFP if(0 && blockIdx.x==2014 && threadIdx.x==17)

//configuration options for GP to tune
#define direct_sequence 1
#define direct_index 1
#define mycache4 1
//#define mycache2 1
#define cache_global_bwt 1
#define scache_global_bwt
//moved decision to set or not direct_global_bwt into barracuda.cu

__device__ unsigned char read_char(unsigned int pos, unsigned int * lastpos, unsigned int * data )
// read character back from sequence arrays
// which is packed as half bytes and stored as in a unsigned int array
{
	unsigned char c;
	unsigned int pos_shifted = pos >> 3;
	unsigned int tmp = *data;
	if (*lastpos!=pos_shifted)
	{
		*data = tmp = tex1Dfetch(sequences_array, pos_shifted);
		*lastpos=pos_shifted;
	}

	switch (pos&0x7)
	{
	case 7:
		c = tmp>>24;
		break;
	case 6:
		c = tmp>>28;
		break;
	case 5:
		c = tmp>>16;
		break;
	case 4:
		c = tmp>>20;
		break;
	case 3:
		c = tmp>>8;
		break;
	case 2:
		c = tmp>>12;
		break;
	case 1:
		c = tmp;
		break;
	case 0:
		c = tmp>>4;
		break;
	}

	return c&0xF;
}

__device__ uint32_t __occ_cuda_aux1(const uint32_t b, const ubyte_t c) {
  const int b0  = (b)&0xff;
  const int b8  = (b)>>8&0xff;
  const int b16 = (b)>>16&0xff;
  const int b24 = (b)>>24;

  const uint32_t cnt0  = bwt_cuda.cnt_table[b0]  >>(8*c)&0xff;
  const uint32_t cnt8  = bwt_cuda.cnt_table[b8]  >>(8*c)&0xff;
  const uint32_t cnt16 = bwt_cuda.cnt_table[b16] >>(8*c)&0xff;
  const uint32_t cnt24 = bwt_cuda.cnt_table[b24] >>(8*c)&0xff;
  const uint32_t cnt   = cnt0+cnt8+cnt16+cnt24;
  //IFP printf("__occ_cuda_aux1(%11d %08x) %02x %02x %02x %02x %2d %2d %2d %2d %3d\n",b,b,b0,b8,b16,b24, cnt0,cnt8,cnt16,cnt24, cnt);
  return cnt;
}

/*
__device__ uint32_t __occ_cuda_aux4(const uint32_t b) {
  const int b0  = (b)&0xff;
  const int b8  = (b)>>8&0xff;
  const int b16 = (b)>>16&0xff;
  const int b24 = (b)>>24;

  const uint32_t cnt0  = bwt_cuda.cnt_table[b0];
  const uint32_t cnt8  = bwt_cuda.cnt_table[b8];
  const uint32_t cnt16 = bwt_cuda.cnt_table[b16];
  const uint32_t cnt24 = bwt_cuda.cnt_table[b24];
  const uint32_t cnt   = cnt0+cnt8+cnt16+cnt24;
  IFP
  printf("__occ_cuda_aux4(%11d %08x) %02x %02x %02x %02x %8d %8d %8d %8d %9d\n",
	   b,b,b0,b8,b16,b24, cnt0,cnt8,cnt16,cnt24, cnt);
  return cnt;
}
*/
//#define __occ_cuda_aux4(b) (bwt_cuda.cnt_table[(b)&0xff]+bwt_cuda.cnt_table[(b)>>8&0xff]+bwt_cuda.cnt_table[(b)>>16&0xff]+bwt_cuda.cnt_table[(b)>>24])

//will end up in "local" memory, should be better than global?
//r1.7 memcpy awful Kernel speed: 4320.67 sequences/sec, 
//better without memcpy?
//could try shared?
//could get calling code load explicitly
//typedef struct { uint32_t bwt0[16]; int i0;} mycache_t;

//probably don not need all of this (yet not using multiple threads per sequence) but...

#ifdef threads_per_sequence
#define threads_mask (threads_per_sequence-1)
#endif /*threads_per_sequence*/

#ifdef cache_global_bwt
#ifdef mycache4
  #define ldg_t uint4
#else
#ifdef mycache2
  #define ldg_t uint2
#else
  #define ldg_t uint32_t
#endif
#endif
#endif /*cache_global_bwt*/

#define load_mycache8 {*my_cache0=load(x);*my_cache1=load(x+inc);*my_cache2=load(x+2*inc);*my_cache3=load(x+3*inc);*my_cache4=load(x+4*inc);*my_cache5=load(x+5*inc);*my_cache6=load(x+6*inc);*my_cache7=load(x+7*inc);}
#define load_mycache4 {*my_cache0=load(x);*my_cache1=load(x+inc);*my_cache2=load(x+2*inc);*my_cache3=load(x+3*inc);}
#define load_mycache2 {*my_cache0=load(x);*my_cache1=load(x+inc);}
#define load_mycache1 {*my_cache0=load(x);}

#ifdef scache_global_bwt
#ifdef mycache4
  #define read_mycache(x) read_mycache_uint4(l_mycache,x)
#else
#ifdef mycache2
  #define read_mycache(x) read_mycache_uint2(l_mycache,x)
#else
  #define read_mycache(x) read_mycache_uint(l_mycache,x)
#endif
#endif
#if (!defined(threads_per_sequence)) || threads_per_sequence == 1
#ifdef mycache4
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3
  #define load_mycache load_mycache4
#else
#ifdef mycache2
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3,&my_cache4,&my_cache5,&my_cache6,&my_cache7
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3,my_cache4,my_cache5,my_cache6,my_cache7
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3;ldg_t my_cache4;ldg_t my_cache5;ldg_t my_cache6;ldg_t my_cache7
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3,ldg_t* my_cache4,ldg_t* my_cache5,ldg_t* my_cache6,ldg_t* my_cache7
  #define load_mycache load_mycache8
#else
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3,&my_cache4,&my_cache5,&my_cache6,&my_cache7,&my_cache8,&my_cache9,&my_cache10,&my_cache11,&my_cache12,&my_cache13,&my_cache14,&my_cache15
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3,my_cache4,my_cache5,my_cache6,my_cache7,my_cache8,my_cache9,my_cache10,my_cache11,my_cache12,my_cache13,my_cache14,my_cache15
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3;ldg_t my_cache4;ldg_t my_cache5;ldg_t my_cache6;ldg_t my_cache7;ldg_t my_cache8;ldg_t my_cache9;ldg_t my_cache10;ldg_t my_cache11;ldg_t my_cache12;ldg_t my_cache13;ldg_t my_cache14;ldg_t my_cache15
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3,ldg_t* my_cache4,ldg_t* my_cache5,ldg_t* my_cache6,ldg_t* my_cache7,ldg_t* my_cache8,ldg_t* my_cache9,ldg_t* my_cache10,ldg_t* my_cache11,ldg_t* my_cache12,ldg_t* my_cache13,ldg_t* my_cache14,ldg_t* my_cache15
  #define load_mycache {*my_cache0=load(0);*my_cache1=load(1);*my_cache2=load(2);*my_cache3=load(3);*my_cache4=load(4);*my_cache5=load(5);*my_cache6=load(6);*my_cache7=load(7);*my_cache8=load(8);*my_cache9=load(9);*my_cache10=load(10);*my_cache11=load(11);*my_cache12=load(12);*my_cache13=load(13);*my_cache14=load(14);*my_cache15=load(15);}
#endif
#endif
#endif /* not threads_per_sequence or 1*/
#if threads_per_sequence == 2
#ifdef mycache4
  #define l_mycache0 &my_cache0,&my_cache1
  #define l_mycache my_cache0,my_cache1
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1
  #define load_mycache load_mycache2
#else
#ifdef mycache2
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3
  #define load_mycache load_mycache4
#else
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3,&my_cache4,&my_cache5,&my_cache6,&my_cache7
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3,my_cache4,my_cache5,my_cache6,my_cache7
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3;ldg_t my_cache4;ldg_t my_cache5;ldg_t my_cache6;ldg_t my_cache7
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3,ldg_t* my_cache4,ldg_t* my_cache5,ldg_t* my_cache6,ldg_t* my_cache7
  #define load_mycache load_mycache8
#endif
#endif
#endif /*threads_per_sequence 2 */
#if threads_per_sequence == 4
#ifdef mycache4
  #define l_mycache0 &my_cache0
  #define l_mycache my_cache0
  #define D_mycache ldg_t my_cache0
  #define d_mycache ldg_t* my_cache0
  #define load_mycache load_mycache1
#else
#ifdef mycache2
  #define l_mycache0 &my_cache0,&my_cache1
  #define l_mycache my_cache0,my_cache1
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1
  #define load_mycache load_mycache2
#else
  #define l_mycache0 &my_cache0,&my_cache1,&my_cache2,&my_cache3
  #define l_mycache my_cache0,my_cache1,my_cache2,my_cache3
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1;ldg_t my_cache2;ldg_t my_cache3
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1,ldg_t* my_cache2,ldg_t* my_cache3
  #define load_mycache load_mycache4
#endif
#endif
#endif /*threads_per_sequence == 4 */
#if threads_per_sequence == 8
#if defined(mycache4) || defined(mycache2)
  #define l_mycache0 &my_cache0
  #define l_mycache my_cache0
  #define D_mycache ldg_t my_cache0
  #define d_mycache ldg_t* my_cache0
  #define load_mycache load_mycache1
#else
  #define l_mycache0 &my_cache0,&my_cache1
  #define l_mycache my_cache0,my_cache1
  #define D_mycache ldg_t my_cache0;ldg_t my_cache1
  #define d_mycache ldg_t* my_cache0,ldg_t* my_cache1
  #define load_mycache load_mycache2
#endif
#endif /*threads_per_sequence == 8 */
#if threads_per_sequence >= 16
  #define l_mycache0 &my_cache0
  #define l_mycache my_cache0
  #define D_mycache ldg_t my_cache0
  #define d_mycache ldg_t* my_cache0
  #define load_mycache load_mycache1
#endif /*threads_per_sequence >= 16 */
#else /*not scache_global_bwt*/
  #define l_mycache0 mycache
  #define l_mycache mycache
#ifndef threads_per_sequence
  #define size_mycache 16
#else
  #define size_mycache (16/threads_per_sequence)
#ifdef mycache4
  #if size_mycache < 4
    #undef size_mycache
    #define size_mycache 4
  #endif
#else
#ifdef mycache2
  #if size_mycache < 2
    #undef size_mycache
    #define size_mycache 2
  #endif
#else
  #if size_mycache < 1
    #undef size_mycache
    #define size_mycache 1
  #endif
#endif
#endif
#endif /*threads_per_sequence*/
  #define d_mycache uint32_t __restrict__ mycache[size_mycache]
  #define read_mycache(x) mycache[x]
#endif /*scache_global_bwt*/

//special for GP 3101804804/4 ncbi_hs_ref/h_sapiens_37.5_asm
#define FIXED_MAX_global_bwt 775451201

__device__ uint32_t __global_bwt(uint32_t *global_bwt, const int i, const int j, int* last, d_mycache) {
  const int cache_loads = 16*sizeof(uint32_t)/sizeof(ldg_t);
  const int ii = i+sizeof(bwtint_t)/sizeof(uint32_t)*j;
  const int base = ii & (~0xf);
  if((*last) != base) {
#if defined(FIXED_MAX_global_bwt) || DEBUG_LEVEL > 0
    assert(!(base<0 || base >= FIXED_MAX_global_bwt)); //provide debug bounds checks
#endif /*debug*/

    const ldg_t* p = (ldg_t*)&(global_bwt[base]);
#ifdef direct_global_bwt
    #define load(x) p[x]
#else
    #define load(x) __ldg(&p[x])
#endif /*direct_global_bwt*/

    //threads_per_sequence not in use...
    int x = 0;
    const int inc = 1;
#ifdef scache_global_bwt
    if(x<cache_loads) load_mycache;
#else /*scache_global_bwt*/
    for(;x<cache_loads;x+=inc) ((ldg_t*)mycache)[x/inc] = load(x);
#endif /*scache_global_bwt*/
    #undef load
    *last = base;
  }
  //threads_per_sequence not in use...
  //IFP printf("__global_bwt(%d,%d) global_bwt[0x%08x] %u %u\n",i,j,ii,global_bwt[ii],mycache[ii & 0xf]);
  return read_mycache(ii & 0xf);
  //return __ldg(&global_bwt[ii]);
  //uint32_t * p = global_bwt + i;
  //return ((bwtint_t *)(p))[j];
}

/*
__device__ ulong4 bwt_cuda_occ4(uint32_t *global_bwt, bwtint_t k)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	// total number of character c in the up to the interval of k
	ulong4 n = {0,0,0,0};
	uint32_t i = 0;

IFP printf("bwt_cuda_occ4(global_bwt,%lu)\n",k);

	if (k == bwt_cuda.seq_len)
	{
		//printf("route 1 - lookup at seqence length\n");
		n.x = bwt_cuda.L2[1]-bwt_cuda.L2[0];
		n.y = bwt_cuda.L2[2]-bwt_cuda.L2[1];
		n.z = bwt_cuda.L2[3]-bwt_cuda.L2[2];
		n.w = bwt_cuda.L2[4]-bwt_cuda.L2[3];
		return n;
	}
	//if (k == (bwtint_t)(-1)) return n;

	if (k >= bwt_cuda.primary){
//		printf("k >= primary, %i\n",int(bwt_cuda.primary));
		--k; // because $ is not in bwt
	}
//	printf("route 3\n");
	//based on #define bwt_occ_intv(b, k) ((b)->bwt + (k)/OCC_INTERVAL*12) where OCC_INTERVAL = 0x80, i.e. 128
	i = k >>7<<4;
//	printf("occ_i = %u\n",i);

//#define USE_SIMON_OCC4 0

	//shifting the array to the right position
	//uint32_t * p = global_bwt + i;
	//IFP	printf("global_bwt[%d %08x] p: %u\n",i,i,p);

	//casting bwtint_t to uint32_t??
	**
	n.x  = ((bwtint_t *)(p))[0];
	n.y  = ((bwtint_t *)(p))[1];
	n.z  = ((bwtint_t *)(p))[2];
	n.w  = ((bwtint_t *)(p))[3];
	**
	n.x  = __global_bwt(global_bwt,i,0);
	n.y  = __global_bwt(global_bwt,i,1);
	n.z  = __global_bwt(global_bwt,i,2);
	n.w  = __global_bwt(global_bwt,i,3);

//	printf("n using occ(i)) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",n.x,n.y,n.z,n.w);

	//p += 8 ; //not sizeof(bwtint_t) coz it equals to 7 not 8;
	int I = i+8;

	bwtint_t j, l, x ;

	j = k >> 4 << 4;

	for (l = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; l < j; l += 16)//, ++p)
	{
		const uint32_t data =  __global_bwt(global_bwt,I,0); //*p;
		I++;
		x += __occ_cuda_aux4(data); //*p);
	}

	const uint32_t data = __global_bwt(global_bwt,I,0); //*p;
	x += __occ_cuda_aux4( data & ~((1U<<((~k&15)<<1)) - 1)) - (~k&15);

	//Return final counts (n)
	n.x += x&0xff;
	n.y += x>>8&0xff;
	n.z += x>>16&0xff;
	n.w += x>>24;



//	printf("calculated n.0 = %u\n",n.x);
//	printf("calculated n.1 = %u\n",n.y);
//	printf("calculated n.2 = %u\n",n.z);
//	printf("calculated n.3 = %u\n",n.w);
	return n;
}
*/
//bwt_cuda_occ4 is confusing as hell, try just doing one base
__device__ ulong bwt_cuda_occ1(uint32_t *global_bwt, bwtint_t k, const ubyte_t c, int* last, d_mycache)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	// total number of character c in the up to the interval of k
	ulong n = 0;
	uint32_t i = 0;

	//IFP printf("bwt_cuda_occ1(global_bwt,%lu,%d)\n",k,c);

	if (k == bwt_cuda.seq_len)
	{
		//printf("route 1 - lookup at seqence length\n");
		return bwt_cuda.L2[c+1]-bwt_cuda.L2[c];
	}
	//if (k == (bwtint_t)(-1)) return n;

	if (k >= bwt_cuda.primary){
//		printf("k >= primary, %i\n",int(bwt_cuda.primary));
		--k; // because $ is not in bwt
	}
//	printf("route 3\n");
	//based on #define bwt_occ_intv(b, k) ((b)->bwt + (k)/OCC_INTERVAL*12) where OCC_INTERVAL = 0x80, i.e. 128
	i = k >>7<<4;
//	printf("occ_i = %u\n",i);

//#define USE_SIMON_OCC4 0

	//shifting the array to the right position
	//uint32_t * p = global_bwt + i;
	//IFP	printf("global_bwt[%d %08x] p: %u\n",i,i,p);

	//casting bwtint_t to uint32_t??
	/*
	n.x  = ((bwtint_t *)(p))[0];
	n.y  = ((bwtint_t *)(p))[1];
	n.z  = ((bwtint_t *)(p))[2];
	n.w  = ((bwtint_t *)(p))[3];
	*/
	n  = __global_bwt(global_bwt,i,c,last,l_mycache);

//	printf("n using occ(i)) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",n.x,n.y,n.z,n.w);

	//p += 8 ; //not sizeof(bwtint_t) coz it equals to 7 not 8;
	int I = i+8;

	bwtint_t j, l, x ;

	j = k >> 4 << 4;

	for (l = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; l < j; l += 16)//, ++p)
	{
		//IFP printf("x %ld %lx, k %lu %lx, l %lu %lx ",x,x,k,k,l,l);
		const uint32_t data =  __global_bwt(global_bwt,I,0,last,l_mycache); //*p;
		I++;
		x += __occ_cuda_aux1(data,c); //*p);
	}
	n += x;

	const uint32_t data = __global_bwt(global_bwt,I,0,last,l_mycache); //*p;
	//take lower 4 bits of ~k (0..9), double it and use it to mask of lower (0..18) bits of data
	const uint32_t nk15 = (~k)&15;
	const uint32_t mask = ~((1U<<(nk15<<1)) - 1);
	//x = __occ_cuda_aux4( data & mask) - nk15;
	x = __occ_cuda_aux1( data & mask,c);
	if(c==0) x -= nk15;
	//IFP printf("c %d, k %lu %lx, nk15 %x, mask %x, x=%lu %lx\n",c,k,k,nk15,mask,x,x);

	//Return final count (n)
	//n += x>>(c*8)&0xff;
	n += x;



//	printf("calculated n.0 = %u\n",n.x);
//	printf("calculated n.1 = %u\n",n.y);
//	printf("calculated n.2 = %u\n",n.z);
//	printf("calculated n.3 = %u\n",n.w);
	return n;
}

__device__ bwtint_t bwt_cuda_occ(uint32_t *global_bwt, bwtint_t k, ubyte_t c, char is_l, int* last,d_mycache)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	//if((k+1)==0) return 0;
	if(c>3) return 0;
	return bwt_cuda_occ1(global_bwt, k, c, last,l_mycache);
}
/*WBL 27 Feb 2015 Let cuda_inexact_match_caller use old code without cache */
__device__ bwtint_t bwt_cuda_occ(uint32_t *global_bwt, bwtint_t k, ubyte_t c, char is_l)
{
//	printf("bwtcudaocc: k:%u\n",k);
	ulong4 ok = bwt_cuda_occ4(global_bwt, k);
	switch ( c )
	{
	case 0:
		return ok.x;
	case 1:
		return ok.y;
	case 2:
		return ok.z;
	case 3:
		return ok.w;
	}
	return 0;
}

#ifndef direct_sequence
__device__ int bwt_cuda_match_exact( uint32_t * global_bwt, unsigned int length, const unsigned char * str, bwtint_t *k0, bwtint_t *l0)
#else
__device__ int bwt_cuda_match_exact( uint32_t * global_bwt, unsigned int length, const unsigned int sequence_offset, bwtint_t *k0, bwtint_t *l0)
#endif
//exact match algorithm
{
	//printf("in exact match function\n");
	int i;
	int last = -1;		
#ifdef scache_global_bwt
		D_mycache;
#else
	//may need non-default __align__ to allow efficent access from __ldg()
#ifdef mycache4
	__align__(16) uint32_t mycache[size_mycache];
#else
#ifdef mycache2
	__align__(8) uint32_t mycache[size_mycache];
#else
	uint32_t mycache[size_mycache];
#endif
#endif
#endif /*scache_global_bwt*/
	bwtint_t k, l;
	k = *k0; l = *l0;
#ifdef direct_sequence
	unsigned int last_read = ~0;
	unsigned int last_read_data = 0;
#endif
	for (i = length - 1; i >= 0; --i)
	{
#ifndef direct_sequence
		unsigned char c = str[i];
		//IFP printf("bwt_cuda_match_exact(..) %d %d\n",i,c);
		if (c > 3){
		  //printf("thread %d,%d k:%lu l:%lu no exact match found, c>3\n",blockIdx.x,threadIdx.x,k,l);
			return 0; // there is an N here. no match
		}
#else
		unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
		if (c > 3){
		  *k0 = 0;
		  *l0 = bwt_cuda.seq_len;
			return 0; // there is an N here. no match
		}
		c = 3 - c;
#endif

		if (!k) k = bwt_cuda.L2[c] + 1;
		else  k = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, k - 1, c, 0, &last,l_mycache0) + 1;

		l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 0, &last,l_mycache0);
		/*if (!k) k = bwt_cuda.L2[c] + 1;
		const bwtint_t save = bwt_cuda_occ(global_bwt, k - 1, c, 0);
		if(k - 1 == l) {
		  k = bwt_cuda.L2[c] + save + 1;
		  l = k - 1;
		} else {
		  k = bwt_cuda.L2[c] + save + 1;
		  l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 0);		  
		}
		*/

		*k0 = k;
		*l0 = l;

		//printf("i:%u, bwt->L2:%lu, c:%u, k:%lu, l:%lu",i,bwt_cuda.L2[c],c,k,l);
		//if (k<=l) printf(" k<=l\n");
				//else printf("\n");
		// no match
		if (k > l)
		{
		  //printf("thread %d,%d k:%lu l:%lu no exact match found, k>l\n",blockIdx.x,threadIdx.x,k,l);
			return 0;
		}

	}
	*k0 = k;
	*l0 = l;

	//printf("thread %d,%d k:%lu l:%lu exact match found: %u\n",blockIdx.x,threadIdx.x,k,l,(l-k+1));

	return (int)(l - k + 1);
}
//#ifdef cache_global_bwt
//now need to keep macro ldg_t as may be used by bwt_cuda_match_exact.cuh cuda2.cuh
//#endif

//////////////////////////////////////////////////////////////////////////////////////////
//DFS MATCH
//////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_find_exact_matches(/*const*/ uint32_t * global_bwt, const int no_of_sequences, 
					const int length, //for direct_index
					bwtkl_t* kl_device)
//init_info_t* global_init, char* global_has_exact)
{
	//WBL re-enabling cuda_find_exact_matches
	//use k and l to signal if exact match or not and return them for use in global alns

	//***EXACT MATCH CHECK***
	//everything that has been commented out in this function should be re-activated if being used
	//comments are only to stop compilation warnings
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	if ( blockId < no_of_sequences ) {
		unsigned char local_complemented_sequence[MAX_SEQUENCE_LENGTH]; //was SEQUENCE_HOLDER_LENGTH];
		//init_info_t local_init;
		//local_init = global_init[blockId];

#ifndef direct_index
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);//local_init.sequence_id);

		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
#else
		const unsigned int sequence_offset = blockId*length;
		const unsigned short sequence_length = length;
#endif
		unsigned int last_read = ~0;
		unsigned int last_read_data = 0;

		if(sequence_length>MAX_SEQUENCE_LENGTH) {
		  //printf("thread %d,%d %d exceeds MAX_SEQUENCE_LENGTH\n",
		  //	 blockIdx.x,threadIdx.x,sequence_length);
		  return;
		}

		if(sequence_length!=length) {
		  //printf("thread %d,%d sequence_length %d not %d\n",
		  //	 blockIdx.x,threadIdx.x,sequence_length,length);
		  return;
		}

		if(sequence_offset!=blockId*length) {
		  //printf("thread %d,%d sequence_offset %d not %d\n",
		  //	 blockIdx.x,threadIdx.x,sequence_offset,blockId*length);
		  return;
		}

#ifndef direct_sequence
		for (int i = 0; i < sequence_length; i++){
			unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
			if(c>3){
			  bwtkl_t tmp_kl = {0,bwt_cuda.seq_len};
			  kl_device[blockId] = tmp_kl;
				return;
			}
			local_complemented_sequence[i] = 3 - c;
		}
#endif /*direct_sequence*/

		bwtint_t k = 0, l = bwt_cuda.seq_len;
		//global_init[blockId].has_exact = global_has_exact[local_init.sequence_id] = bwt_cuda_match_exact(global_bwt, sequence_length, local_complemented_sequence, &k, &l)>0 ? 1 : 0;
#ifndef direct_sequence
		bwt_cuda_match_exact(global_bwt, sequence_length, local_complemented_sequence, &k, &l);
#else
		bwt_cuda_match_exact(global_bwt, sequence_length, sequence_offset, &k, &l);
#endif
		bwtkl_t tmp_kl = {k,l};
		kl_device[blockId] = tmp_kl;
	}
	return;
}

//END CUDA DEVICE CODE

