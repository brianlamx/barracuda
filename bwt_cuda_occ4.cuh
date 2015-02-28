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

/* (0.7.0) beta: $Revision: 1.2 $
  27 Feb 2015 WBL mycache4 but not direct_global_bwt __ldg to access global_bwt
  12 Feb 2015 WBL taken from cuda.cuh r1.1 for cuda2.cuh
   4 Dec 2014 WBL Extract aln device code to new separate file cuda.cuh
*/

#define __occ_cuda_aux4(b) (bwt_cuda.cnt_table[(b)&0xff]+bwt_cuda.cnt_table[(b)>>8&0xff]+bwt_cuda.cnt_table[(b)>>16&0xff]+bwt_cuda.cnt_table[(b)>>24])

__device__ inline uint32_t 
helper(const int i4, uint4* mycache, const uint32_t * global_bwt4) {
  const int i = (i4 & 0x3);
  if(i==0) {
#ifdef direct_global_bwt
    *mycache =      *(uint4*)global_bwt4;
#else
    *mycache = __ldg((uint4*)global_bwt4);
#endif /*direct_global_bwt*/
  }
  return ((uint32_t *)mycache)[i];
}
__device__ inline uint32_t 
occ_cuda_aux4(const int i4, uint4* mycache, const uint32_t * global_bwt4) {
  return __occ_cuda_aux4(helper(i4,mycache,global_bwt4));}
__device__ inline uint32_t 
occ_cuda_aux4(const int i4, uint4* mycache, const uint32_t * global_bwt4, const uint32_t mask) {
  return __occ_cuda_aux4(helper(i4,mycache,global_bwt4) & mask);}


__device__ ulong4 bwt_cuda_occ4(uint32_t *global_bwt, bwtint_t k)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	// total number of character c in the up to the interval of k
	ulong4 n = {0,0,0,0};
	uint32_t i = 0;

//	printf("bwtcudaocc4: k:%u\n",k);

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

#define USE_SIMON_OCC4 0

#if USE_SIMON_OCC4 == 0
	//shifting the array to the right position
	uint32_t * p = global_bwt + i;
//	printf("p: %p\n", p);


//effectively assumes mycache4, ie ldg_t is uint4
	//assert(sizeof(ulong4)==2*sizeof(uint4));
#ifdef direct_global_bwt
	((uint4*)(&n))[0] = ((uint4*)&p[0])[0];   //n.x,n.y
	((uint4*)(&n))[1] = ((uint4*)&p[2])[4];   //n.z,n.w
#else
	((uint4*)(&n))[0] = __ldg((uint4*)p);     //n.x,n.y
	((uint4*)(&n))[1] = __ldg((uint4*)&p[4]); //n.z,n.w
#endif /*direct_global_bwt*/
	/*casting bwtint_t to uint32_t??
	n.x  = ((bwtint_t *)(p))[0];
	n.y  = ((bwtint_t *)(p))[1];
	n.z  = ((bwtint_t *)(p))[2];
	n.w  = ((bwtint_t *)(p))[3];
	*/

//	printf("n using occ(i)) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",n.x,n.y,n.z,n.w);

	p += 8 ; //not sizeof(bwtint_t) coz it equals to 7 not 8;

	bwtint_t j, l, x ;

	j = k >> 4 << 4;

	int ii=0; uint4 b;
	for (l = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; l < j; l += 16, ++p,++ii)
	{
		x += occ_cuda_aux4(ii,&b,p);
	}

	x += occ_cuda_aux4(ii,&b,p, ~((1U<<((~k&15)<<1)) - 1)) - (~k&15);

	//Return final counts (n)
	n.x += x&0xff;
	n.y += x>>8&0xff;
	n.z += x>>16&0xff;
	n.w += x>>24;


#else
	//TODO: Simon's BWTOCC4 is not working yet!
	bwtint_t m = 0;
	ulong4 tmp;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// tmp variables
	unsigned int tmp1,tmp2;//, tmp3;

	//shifting the array to the right position
	uint32_t * p = global_bwt + i;
	printf("p: %u\n", p);
	uint32_t * p1 = p + 1;
	printf("p1: %u\n", p1);
	//casting bwtint_t to uint32_t??
	tmp.x  = ((bwtint_t *)(p1))[0];
	tmp.y  = ((bwtint_t *)(p1))[1];
	tmp.z  = ((bwtint_t *)(p1))[2];
	tmp.w  = ((bwtint_t *)(p1))[3];

	printf("tmp using occ(p1)) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",tmp.x,tmp.y,tmp.z,tmp.w);

	if (k&0x40)
	{
		uint32_t *p2 = p + 2;
		printf("k&0x40 true: p1: %u\n", p2);
		m = __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		m += __occ_cuda_aux4(tmp.z);
		m += __occ_cuda_aux4(tmp.w);
		printf("m: %lu\n", m);
		tmp.x  = ((bwtint_t *)(p2))[0];
		tmp.y  = ((bwtint_t *)(p2))[1];
		tmp.z  = ((bwtint_t *)(p2))[2];
		tmp.w  = ((bwtint_t *)(p2))[3];
		printf("k&0x40 is true: occ(p2) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",tmp.x,tmp.y,tmp.z,tmp.w);
	}
	if (k&0x20)
	{
		m += __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		printf("k&020 is true: m: %lu\n", m);
		tmp1=tmp.z;
		tmp2=tmp.w;
	} else {
		tmp1=tmp.x;
		tmp2=tmp.y;
	}
	if (k&0x10)
	{
		m += __occ_cuda_aux4(tmp1);
		printf("k&010 is true: m: %lu\n", m);
		tmp1=tmp2;
	}

	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	m += __occ_cuda_aux4(tmp1>>(((~k)&15)<<1));

	printf("none of the ks is true: m: %lu\n", m);
	n.x = m&0xff; n.y = m>>8&0xff; n.z = m>>16&0xff; n.w = m>>24; //look into this
	printf ("m numbers: %lu, %lu, %lu, %lu\n", n.x, n.y, n.z, n.w);

	// retrieve the total count from index the number of character C in the up k/128bits interval
	tmp.x  = ((bwtint_t *)(p))[0];
	tmp.y  = ((bwtint_t *)(p))[1];
	tmp.z  = ((bwtint_t *)(p))[2];
	tmp.w  = ((bwtint_t *)(p))[3];
	printf("final occ(p) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",tmp.x,tmp.y,tmp.z,tmp.w);
	n.x += tmp.x; n.x -= ~k&15; n.y += tmp.y; n.z += tmp.z; n.w += tmp.w;

#endif

//	printf("calculated n.0 = %u\n",n.x);
//	printf("calculated n.1 = %u\n",n.y);
//	printf("calculated n.2 = %u\n",n.z);
//	printf("calculated n.3 = %u\n",n.w);
	return n;
}

