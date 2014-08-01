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

#define PACKAGE_VERSION "0.7.0 beta"
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include "bwtaln.h"
#include "bwtgap.h"
#include "utils.h"
#include "barracuda.h"
#include "barracuda.cuh"

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

// Performance switches
#define BWT_2_OCC_ENABLE 0 // enable looking up of k and l in the same time for counting character occurrence (slower, so disable by default)
#define BWT_TABLE_LOOKUP_ENABLE 1 // use lookup table when instead of counting character occurrence, (faster so enable by default)

//The followings are settings for memory allocations and memory requirements
#define MIN_MEM_REQUIREMENT 768 // minimal global memory requirement in (MiB).  Currently at 768MB
#define CUDA_TESLA 1350 // enlarged workspace buffer. Currently at 1350MB will be halved if not enough mem available

#define SEQUENCE_TABLE_SIZE_EXPONENTIAL 23// DO NOT CHANGE! buffer size in (2^)units for sequences and alignment storages (batch size)
// Maximum exponential is up to 30 [~ 1  GBytes] for non-debug, non alignment
// Maximum exponential is up to 26 [~ 128MBytes] for debug
// Maximum exponential is up to 23 for alignment with 4GB RAM(default : 23)

//The followings are for DEBUG only
#define CUDA_SAMSE 0 //Enable CUDA SAMSE code, debug only (leave ON)

// how much debugging information shall the kernel output? kernel output only works for fermi and above
#define DEBUG_LEVEL 0
#define USE_PETR_SPLIT_KERNEL 0
// how long should a subsequence be for one kernel launch


//Global variables for inexact match <<do not change>>
#define STATE_M 0
#define STATE_I 1
#define STATE_D 2


#define write_to_half_byte_array(array,index,data) \
	(array)[(index)>>1]=(unsigned char)(((index)&0x1)?(((array)[(index)>>1]&0xF0)|((data)&0x0F)):(((data)<<4)|((array)[(index)>>1]&0x0F)))

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//CUDA global variables
__device__ __constant__ bwt_t bwt_cuda;
__device__ __constant__ bwt_t rbwt_cuda;
__device__ __constant__ uint32_t* bwt_occ_array2;
__device__ __constant__ gap_opt_t options_cuda;

//Texture Maps
// uint4 is used because the maximum width for CUDA texture bind of 1D memory is 2^27,
// and uint4 the structure 4xinteger is x,y,z,w coordinates and is 16 bytes long,
// therefore effectively there are 2^27x16bytes memory can be access = 2GBytes memory.
texture<uint4, 1, cudaReadModeElementType> bwt_occ_array;
texture<uint4, 1, cudaReadModeElementType> rbwt_occ_array;
texture<unsigned int, 1, cudaReadModeElementType> sequences_array;
texture<uint2, 1, cudaReadModeElementType> sequences_index_array;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void report_cuda_error_GPU(const char *message)
{
	cudaError_t cuda_err = cudaGetLastError();

	if(cudaSuccess != cuda_err)
	{
		fprintf(stderr,"%s\n",message);
		fprintf(stderr,"%s\n", cudaGetErrorString(cuda_err));
		exit(1);
	}
}

void report_cuda_error_GPU(cudaError_t cuda_error, const char *message)
{
	if(cudaSuccess != cuda_error)
	{
		fprintf(stderr,"%s\n",message);
		fprintf(stderr,"%s\n", cudaGetErrorString(cuda_error));
		exit(1);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


unsigned long copy_bwts_to_cuda_memory(const char * prefix, uint32_t ** bwt, uint32_t mem_available, bwtint_t* seq_len)
// bwt occurrence array to global and bind to texture, bwt structure to constant memory
// this function only load part of the bwt for alignment only.  SA is not loaded.
// mem available in MiB (not bytes)
{
	bwt_t * bwt_src;
	unsigned long  size_read = 0;

	#if DEBUG_LEVEL > 0
			fprintf(stderr,"[aln_debug] mem left: %d\n", mem_available);
	#endif

	//Original BWT
	//Load bwt occurrence array from from disk
	char *str = (char*)calloc(strlen(prefix) + 10, 1);
	strcpy(str, prefix); strcat(str, ".bwt");
	bwt_src = bwt_restore_bwt(str);
	free(str);

	#if DEBUG_LEVEL > 0
			fprintf(stderr,"[aln_debug] bwt loaded to CPU \n");
	#endif
	size_read = bwt_src->bwt_size*sizeof(uint32_t);
	mem_available = mem_available - uint32_t (size_read>>20); // mem available in MiB (not bytes)
	*seq_len = bwt_src->seq_len;

	if(mem_available > 0)
	{
		//Allocate memory for bwt
		cudaMalloc((void**)bwt, bwt_src->bwt_size*sizeof(uint32_t));
		report_cuda_error_GPU("[aln_core] Error allocating memory for \"bwt_occurrence array\".\n");
		//copy bwt occurrence array from host to device and dump the bwt to save CPU memory
		cudaMemcpy (*bwt, bwt_src->bwt, bwt_src->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
		report_cuda_error_GPU("[aln_core] Error copying  \"bwt occurrence array\" to GPU.\n");

		#if DEBUG_LEVEL > 0
			fprintf(stderr,"[aln_debug] bwt loaded to GPU \n");
			fprintf(stderr,"[aln_debug] bwtsize in MiB %u\n",(bwt_src->bwt_size*sizeof(uint32_t)) >>20);
		#endif

		//copy bwt structure data to constant memory bwt_cuda structure
		cudaMemcpyToSymbol ( bwt_cuda, bwt_src, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
		report_cuda_error_GPU("[aln_core] Error binding  \"bwt_src\" to bwt_cuda constant.\n");

		//free bwt_src from memory
		bwt_destroy(bwt_src);


	}
	else
	{
		fprintf(stderr,"[aln_core] Not enough device memory to perform alignment.\n");
		//free bwt_src from memory
		bwt_destroy(bwt_src);
		return 0;
	}

	#if DEBUG_LEVEL > 0
			fprintf(stderr,"[aln_debug] bwt loaded, mem left: %d MiB\n", mem_available);
	#endif

	return size_read;
}

void free_bwts_from_cuda_memory( unsigned int * bwt)
{
	if ( bwt != 0 )
	{
		//cudaUnbindTexture(bwt_occ_array);
		cudaFree(bwt);
	}
}

void swap(bwt_aln1_t *x, bwt_aln1_t *y)
{
   bwt_aln1_t temp;
   temp = *x;
   *x = *y;
   *y = temp;
}

int choose_pivot(int i,int j)
{
   return((i+j) /2);
}

void aln_quicksort(bwt_aln1_t *aln, int m, int n)
//This function sorts the alignment array from barracuda to make it compatible with SAMSE/SAMPE cores
{
	int key,i,j,k;

	if (m < n)
	{
	      k = choose_pivot(m, n);
	      swap(&aln[m],&aln[k]);
	      key = aln[m].score;
	      i = m+1;
	      j = n;
	      while(i <= j)
	      {
	         while((i <= n) && (aln[i].score <= key))
	                i++;
	         while((j >= m) && (aln[j].score > key))
	                j--;
	         if(i < j)
	                swap(&aln[i],&aln[j]);
	      }
	      // swap two elements
	      swap(&aln[m],&aln[j]);
	      // recursively sort the lesser lists
	      aln_quicksort(aln, m, j-1);
	      aln_quicksort(aln, j+1, n);
	 }
}

///new sorting and sequence input code


//TODO starts here!!!!!!!!!!!

void barracuda_sort_queries(bwa_seq_t *seqs, unsigned int *order)
{
	return;
}

inline void barracuda_seq_reverse(int len, char *seq)
{
	int i;
	for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			seq[len-1-i] = seq[i]; seq[i] = tmp;
	}
}

inline void  barracuda_write_to_half_byte_array(barracuda_query_array_t *seqs, unsigned char *half_byte_array, uint2 *main_sequences_index, int nseqs)
{
	int i, j = 0;
	int start_pos = 0;

	for (i = 0; i < nseqs; i++){
		barracuda_query_array_t * p = seqs + i;
		//moved from barracuda_read_seq.  Reason: sorting needed to be done from the back.
		barracuda_seq_reverse(p->len, p->seq); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		main_sequences_index[i].x = start_pos;
		main_sequences_index[i].y = p->len;
		for (j = 0; j < p->len; j++)
		{
			//fprintf(stderr,"now writing at position %i, character %i\n", start_pos+j, p->seq[j] );
			write_to_half_byte_array(half_byte_array,start_pos+j,p->seq[j]);
		}
		//printf("index: %i\n", start_pos);
		start_pos += p->len;
	}
	return;
}

int copy_sequences_to_cuda_memory(
		bwa_seqio_t *bs,
		uint2 *global_sequences_index,
		uint2 *main_sequences_index,
		unsigned char *global_sequences,
		unsigned char *main_sequences,
		unsigned int *read_size,
		unsigned short *max_length,
		int buffer,
		unsigned long long *clump_array,
		unsigned char clump_len)
{
	//sum of length of sequences up the the moment
	unsigned int accumulated_length = 0;
	//sequence's read length

	unsigned int number_of_sequences = 0;

#define NEW_QUERY_READER 1

#if NEW_QUERY_READER == 0
	unsigned short read_length = 0;
	while (bwa_read_seq_one_half_byte(bs,main_sequences,accumulated_length,&read_length,clump_array,clump_len,number_of_sequences)>0)
	{
		main_sequences_index[number_of_sequences].x = accumulated_length;
		main_sequences_index[number_of_sequences].y = read_length;
		if (read_length > *max_length) *max_length = read_length;

		accumulated_length += read_length;
		number_of_sequences++;

		if ( accumulated_length + MAX_SEQUENCE_LENGTH > (1ul<<(buffer+1)) ) break;
	}
#else
	int n_seqs = 0;
	barracuda_query_array_t *seqs = barracuda_read_seqs(bs,  buffer, &n_seqs, 0, 0, &accumulated_length);
	//TODO: insert  sort here!!!!
	//TODO: Arran: put the clumping code here.
	barracuda_write_to_half_byte_array(seqs, main_sequences, main_sequences_index, n_seqs);
	number_of_sequences = (unsigned int) n_seqs;

#endif

	//copy main_sequences_width from host to device
	cudaUnbindTexture(sequences_index_array);
    cudaMemcpy(global_sequences_index, main_sequences_index, (number_of_sequences)*sizeof(uint2), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_index_array, global_sequences_index, (number_of_sequences)*sizeof(uint2));

    //copy main_sequences from host to device, sequences array length should be accumulated_length/2
    cudaUnbindTexture(sequences_array);
    cudaMemcpy(global_sequences, main_sequences, (1ul<<(buffer))*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_array, global_sequences, (1ul<<(buffer))*sizeof(unsigned char));

    if ( read_size ) *read_size = accumulated_length;
    free (seqs);
    return number_of_sequences;
}

//CUDA DEVICE CODE STARTING FROM THIS LINE
/////////////////////////////////////////////////////////////////////////////


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

__device__ inline unsigned int numbits(unsigned int i, unsigned char c)
// with y of 32 bits which is a string sequence encoded with 2 bits per alphabet,
// count the number of occurrence of c ( one pattern of 2 bits alphabet ) in y
{
	i = ((c&2)?i:~i)>>1&((c&1)?i:~i)&0x55555555;
	i = (i&0x33333333)+(i>>2&0x33333333);
	return((i+(i>>4)&0x0F0F0F0F)*0x01010101)>>24;
}

#define __occ_cuda_aux4(b) (bwt_cuda.cnt_table[(b)&0xff]+bwt_cuda.cnt_table[(b)>>8&0xff]+bwt_cuda.cnt_table[(b)>>16&0xff]+bwt_cuda.cnt_table[(b)>>24])


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
//	printf("p: %u\n", p);

	//casting bwtint_t to uint32_t??
	n.x  = ((bwtint_t *)(p))[0];
	n.y  = ((bwtint_t *)(p))[1];
	n.z  = ((bwtint_t *)(p))[2];
	n.w  = ((bwtint_t *)(p))[3];

//	printf("n using occ(i)) tmp.x: %lu, tmp.y: %lu, tmp.z: %lu, tmp.w: %lu\n",n.x,n.y,n.z,n.w);

	p += 8 ; //not sizeof(bwtint_t) coz it equals to 7 not 8;

	bwtint_t j, l, x ;

	j = k >> 4 << 4;

	for (l = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; l < j; l += 16, ++p)
	{
		x += __occ_cuda_aux4(*p);
	}

	x += __occ_cuda_aux4( *p & ~((1U<<((~k&15)<<1)) - 1)) - (~k&15);

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

__device__ bwtint_t bwt_cuda_occ(uint32_t *global_bwt, bwtint_t k, ubyte_t c, char is_l)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
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

__device__ void bwt_cuda_device_calculate_width (uint32_t * global_bwt, unsigned char* sequence, unsigned int * widths, unsigned char * bids, unsigned short offset, unsigned short length)
//Calculate bids and widths for worst case bound, returns widths[senquence length] and bids[sequence length]
{
#if DEBUG_LEVEL > 8
	printf ("in cal width\n");
#endif
	unsigned short bid;
	//suffix array interval k(lower bound) and l(upper bound)
	bwtint_t k, l;
	unsigned int i;

	// do calculation and update w and bid
	bid = 0;
	k = 0;
	l = bwt_cuda.seq_len;
	//printf("seq_len: %llu\n", bwt_cuda.seq_len);
//	printf("from GPU\n");

//	printf("k&l calculations\n");
	for (i = offset; i < length; ++i) {
		unsigned char c = sequence[i];
		//printf("\n\nbase %u,%u\n",i,c);
		//printf("k: %llu l: %llu\n",k,l);
		if (c < 4) {
				//printf("Calculating k\n");
			//unsigned long long startK = k;
			//unsigned long long tmpK = ((k==0)?0:bwt_cuda_occ(global_bwt, k - 1, c)) + 1;
				k = bwt_cuda.L2[c] + ((k==0)?0:bwt_cuda_occ(global_bwt, k - 1, c, 0)) + 1;
				//printf("Calculating l\n");
				//unsigned long long startL = l;
				//unsigned long long tmpL = bwt_cuda_occ(global_bwt, l, c);
				l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 1);
				//printf("%i	occ: %llu\n", c, bwt_cuda.L2[c]);
				//if(offset==0) printf("i:%d,c:%d,bwt:%u,\n", i, c, bwt_cuda.L2[c]);
//				printf("k:%u,",k);
//				printf("l:%u\n",l);
		}
		if (k > l || c > 3) {
			k = 0;
			l = bwt_cuda.seq_len;
			++bid;
		}
		widths[i] = l - k + 1;
		bids[i] = bid;
	}
	widths[length] = k + 1;
	bids[length] = bid;
	//printf("\n\n here comes width and bids\n");
	//for(i = 0; i < length; ++i){
//		printf("%u,%d,%u,%u\n",i,sequence[i],widths[i],bids[i]);
//	}
	return;
}

__device__ int bwa_cuda_cal_maxdiff(int l, double err, double thres)
{
	double elambda = exp(-l * err);
	double sum, y = 1.0;
	int k, x = 1;
	for (k = 1, sum = elambda; k < 1000; ++k) {
		y *= l * err;
		x *= k;
		sum += elambda * y / x;
		if (1.0 - sum < thres) return k;
	}
	return 2;
}

__device__ void gap_stack_shadow_cuda(int x, bwtint_t max, int last_diff_pos, unsigned int * width, unsigned char * bid)
{
	int i, j;
	for (i = j = 0; i < last_diff_pos; ++i)
	{
		if (width[i] > x)
		{
			width[i] -= x;
		}
		else if (width[i] == x)
		{
			bid[i] = 1;
			width[i] = max - (++j);
		} // else should not happen
	}
}

__device__ unsigned int int_log2_cuda(uint32_t v)
//integer log
{
	unsigned int c = 0;
	if (v & 0xffff0000u) { v >>= 16; c |= 16; }
	if (v & 0xff00) { v >>= 8; c |= 8; }
	if (v & 0xf0) { v >>= 4; c |= 4; }
	if (v & 0xc) { v >>= 2; c |= 2; }
	if (v & 0x2) c |= 1;
	return c;
}

__device__ int bwt_cuda_match_exact( uint32_t * global_bwt, unsigned int length, const unsigned char * str, bwtint_t *k0, bwtint_t *l0)
//exact match algorithm
{
	//printf("in exact match function\n");
	int i;
	bwtint_t k, l;
	k = *k0; l = *l0;
	for (i = length - 1; i >= 0; --i)
	{
		unsigned char c = str[i];
		if (c > 3){
			//printf("no exact match found, c>3\n");
			return 0; // there is an N here. no match
		}

		if (!k) k = bwt_cuda.L2[c] + 1;
		else  k = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, k - 1, c, 0) + 1;

		l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 0);

		*k0 = k;
		*l0 = l;

		//printf("i:%u, bwt->L2:%lu, c:%u, k:%lu, l:%lu",i,bwt_cuda.L2[c],c,k,l);
		//if (k<=l) printf(" k<=l\n");
				//else printf("\n");
		// no match
		if (k > l)
		{
			//printf("no exact match found, k>l\n");
			return 0;
		}

	}
	*k0 = k;
	*l0 = l;

	//printf("exact match found: %u\n",(l-k+1));

	return (int)(l - k + 1);
}


//////////////////////////////////////////////////////////////////////////////////////////
//DFS MATCH
//////////////////////////////////////////////////////////////////////////////////////////

__device__ void cuda_dfs_initialize(ulong2 *stack, uchar4 *stack_mm, uint2 *position_data, char4 *pushes/*, int * scores*/)
//initialize memory store for dfs_match
{
	int i;
	ulong2 temp1 = {0,0};
	uchar4 temp2 = {0,0,0,0};
	char4 temp3 = {0,0,0,0};;
	uint2 temp4 = {0,0};

	// zero out the whole stack
	for (i = 0; i < MAX_ALN_LENGTH; i++)
	{
		stack[i] = temp1;
		stack_mm[i] = temp2;
		pushes[i] = temp3;
		position_data[i] = temp4;
	}
	return;
}

__device__ void cuda_dfs_push(ulong2 *stack, uchar4 *stack_mm, uint2* position_data, char4 *pushes, int i, bwtint_t k, bwtint_t l, int n_mm, int n_gapo, int n_gape, int state, int is_diff, int current_stage)
//create a new entry in memory store
{
	//printf(",%i/%i", i, current_stage);
	stack[current_stage].x = k;
	stack[current_stage].y = l;
	//stack[current_stage].z = i;
	//if (is_diff)stack[current_stage].w = i;

	position_data[current_stage].x = i;
	if(is_diff) position_data[current_stage].y = i;

	//printf("length pushed: %u\n", i);
	stack_mm[current_stage].x = n_mm;
	stack_mm[current_stage].y = n_gapo;
	stack_mm[current_stage].z = n_gape;
	stack_mm[current_stage].w = state;

	char4 temp = {0,0,0,0};
	pushes[current_stage] = temp;
	return;
}
#if USE_PETR_SPLIT_KERNEL > 0

__device__ int cuda_split_dfs_match(const int len, const unsigned char *str, const int sequence_type, unsigned int *widths, unsigned char *bids, const gap_opt_t *opt, alignment_meta_t *aln, int best_score, const int max_aln)
//This function tries to find the alignment of the sequence and returns SA coordinates, no. of mismatches, gap openings and extensions
//It uses a depth-first search approach rather than breath-first as the memory available in CUDA is far less than in CPU mode
//The search rooted from the last char [len] of the sequence to the first with the whole bwt as a ref from start
//and recursively narrow down the k(upper) & l(lower) SA boundaries until it reaches the first char [i = 0], if k<=l then a match is found.
{

	//Initialisations
	int start_pos = aln->start_pos;
	// only obey the sequence_type for the first run
	int best_diff = (start_pos)? aln->init.best_diff :opt->max_diff + 1;
	int max_diff = opt->max_diff;
	//int best_cnt = (start_pos)? aln->init.best_cnt:0;
	int best_cnt = 0;
	const bwt_t * bwt = (sequence_type == 0)? &rbwt_cuda: &bwt_cuda; // rbwt for sequence 0 and bwt for sequence 1;
	const int bwt_type = 1 - sequence_type;
	int current_stage = 0;
	uint4 entries_info[MAX_SEQUENCE_LENGTH];
	uchar4 entries_scores[MAX_SEQUENCE_LENGTH];
	char4 done_push_types[MAX_SEQUENCE_LENGTH];
	int n_aln = (start_pos)? aln->no_of_alignments : 0;
	int loop_count = 0;
	const int max_count = options_cuda.max_entries;


	/* debug to print out seq only for a first 5, trick to unserialise
	if (!start_pos && aln->sequence_id < 5 && sequence_type == 0) {
		// trick to serialise execution
		for (int g = 0; g < 5; g++) {
			if (g == aln->sequence_id) {
				printf("seq id: %d",aln->sequence_id);
				for (int x = 0; x<len; x++) {
					printf(".%d",str[x]);
				}
				printf("\n");
			}
		}
	}
	*/

	//Initialise memory stores first in, last out
	cuda_dfs_initialize(entries_info, entries_scores, done_push_types/*, scores*/); // basically zeroes out the stack

	//push first entry, the first char of the query sequence into memory stores for evaluation
	cuda_dfs_push(entries_info, entries_scores, done_push_types, len, aln->init.lim_k, aln->init.lim_l, aln->init.cur_n_mm, aln->init.cur_n_gapo, aln->init.cur_n_gape, aln->init.cur_state, 0, current_stage); //push initial entry to start


	#if DEBUG_LEVEL > 6
	printf("initial k:%u, l: %u \n", aln->init.lim_k, aln->init.lim_l);
	#endif

	#if DEBUG_LEVEL > 6
	for (int x = 0; x<len; x++) {
		printf(".%d",str[x]);
	}

	// print out the widths and bids
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%i,",bids[x]);
	}
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%d;",widths[x]);
	}


	printf("\n");

	printf("max_diff: %d\n", max_diff);

	#endif



	while(current_stage >= 0)
	{

		int i,j, m;
		int hit_found, allow_diff, allow_M;
		bwtint_t k, l;
		char e_n_mm, e_n_gapo, e_n_gape, e_state;
		unsigned int occ;
		loop_count ++;

		int worst_tolerated_score = (options_cuda.mode & BWA_MODE_NONSTOP)? 1000: best_score + options_cuda.s_mm;



		//define break from loop conditions

		if (n_aln == max_aln) {
#if DEBUG_LEVEL > 7
			printf("breaking on n_aln == max_aln\n");
#endif
			break;
		}
		// TODO tweak this, otherwise we miss some branches
		if (best_cnt > options_cuda.max_top2 + (start_pos==0)*2) {
		//if (best_cnt > options_cuda.max_top2) {
#if DEBUG_LEVEL > 7
			printf("breaking on best_cnt>...\n");
#endif
			break;
		}
		if (loop_count > max_count) {
#if DEBUG_LEVEL > 7
			printf("loop_count > max_count\n");
#endif
			break;

		}


		//put extracted entry into local variables
		k = entries_info[current_stage].x; // SA interval
		l = entries_info[current_stage].y; // SA interval
		i = entries_info[current_stage].z; // length
		e_n_mm = entries_scores[current_stage].x; // no of mismatches
		e_n_gapo = entries_scores[current_stage].y; // no of gap openings
		e_n_gape = entries_scores[current_stage].z; // no of gap extensions
		e_state = entries_scores[current_stage].w; // state (M/I/D)


//		// TODO seed length adjustment - get this working after the split length - is it even important?
//		// debug information
//		if (aln->sequence_id == 1 && i > len-2) {
//			printf("\n\ninlocal maxdiff: %d\n", opt->max_diff);
//			printf("inlocal seed diff: %d\n\n", opt->max_seed_diff);
//			printf("inlocal seed length: %d\n", opt->seed_len);
//			printf("inlocal read length: %d\n", len);
//			printf("inlocal start_pos: %d\n", start_pos);
//			printf("inlocal seed_pos: %d, %d\n\n\n", start_pos + (len-i), i);
//		}
		//max_diff = (start_pos + (len-i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;

		// new version not applying seeding after the split
		max_diff = (!start_pos && (len-i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;


		int allow_gap;

		if(opt->fast)
		{
			allow_gap = (!start_pos && (len-i) <  opt->seed_len)? 0 : opt->max_gapo;
		}else
		{
			allow_gap = opt->max_gapo;
		}


		//calculate score
		int score = e_n_mm * options_cuda.s_mm + e_n_gapo * options_cuda.s_gapo + e_n_gape * options_cuda.s_gape;



		//calculate the allowance for differences
		m = max_diff - e_n_mm - e_n_gapo;


#if DEBUG_LEVEL > 7
		printf("k:%u, l: %u, i: %i, score: %d, cur.stage: %d, mm: %d, go: %d, ge: %d, m: %d\n", k, l,i, score, current_stage, e_n_mm, e_n_gapo, e_n_gape, m);
#endif

		if (options_cuda.mode & BWA_MODE_GAPE) m -= e_n_gape;


		if(score > worst_tolerated_score) break;

		// check if the entry is outside boundary or is over the max diff allowed)
		if (m < 0 || (i > 0 && m < bids[i-1]))
		{
#if DEBUG_LEVEL > 6

			printf("breaking: %d, m:%d\n", bids[i-1],m);
#endif
			current_stage --;
			continue;
		}

		// check whether a hit (full sequence when it reaches the last char, i.e. i = 0) is found, if it is, record the alignment information
		hit_found = 0;
		if (!i)
		{
			hit_found = 1;
		}else if (!m) // alternatively if no difference is allowed, just do exact match)
		{
			if ((e_state == STATE_M ||(options_cuda.mode&BWA_MODE_GAPE) || e_n_gape == opt->max_gape))
			{
				if (bwt_cuda_match_exact(bwt_type, i, str, &k, &l))
				{
					hit_found = 1;
				}else
				{
					current_stage --;
					continue; // if there is no hit, then go backwards to parent stage
				}
			}
		}


		if (hit_found)
		{
			// action for found hits
			//int do_add = 1;

			if (score < best_score)
			{
				best_score = score;
				best_diff = e_n_mm + e_n_gapo + (options_cuda.mode & BWA_MODE_GAPE) * e_n_gape;
				best_cnt = 0; //reset best cnt if new score is better
				if (!(options_cuda.mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;

			if (e_n_gapo)
			{ // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array unless the new score is better.
				for (j = 0; j < n_aln; ++j)
					if (aln->alignment_info[j].k == k && aln->alignment_info[j].l == l) break;
				if (j < n_aln)
				{
					if (score < aln->alignment_info[j].score)
						{
							aln->alignment_info[j].score = score;
							aln->alignment_info[j].n_mm = e_n_mm;
							aln->alignment_info[j].n_gapo = e_n_gapo;
							aln->alignment_info[j].n_gape = e_n_gape;
							aln->alignment_info[j].score = score;
					//		aln->alignment_info[j].best_cnt = best_cnt;
						//	aln->alignment_info[j].best_diff = best_diff;

						}
					//do_add = 0;
					hit_found = 0;
#if DEBUG_LEVEL > 8
printf("alignment already present, amending score\n");
#endif
				}
			}

			if (hit_found)
			{ // append result the alignment record array
				gap_stack_shadow_cuda(l - k + 1, len, bwt->seq_len, e_state,
						widths, bids);
					// record down number of mismatch, gap open, gap extension and a??

					aln->alignment_info[n_aln].n_mm = entries_scores[current_stage].x;
					aln->alignment_info[n_aln].n_gapo = entries_scores[current_stage].y;
					aln->alignment_info[n_aln].n_gape = entries_scores[current_stage].z;
					aln->alignment_info[n_aln].a = sequence_type;
					// the suffix array interval
					aln->alignment_info[n_aln].k = k;
					aln->alignment_info[n_aln].l = l;
					aln->alignment_info[n_aln].score = score;
				//	aln->alignment_info[n_aln].best_cnt = best_cnt;
				//	aln->alignment_info[n_aln].best_diff = best_diff;
#if DEBUG_LEVEL > 8
					printf("alignment added: k:%u, l: %u, i: %i, score: %d, cur.stage: %d, m:%d\n", k, l, i, score, current_stage, m);
#endif
					++n_aln;

			}
			current_stage --;
			continue;
		}




		// proceed and evaluate the next base on sequence
		--i;

		// retrieve Occurrence values and determine all the eligible daughter nodes, done only once at the first instance and skip when it is revisiting the stage
		unsigned int ks[MAX_SEQUENCE_LENGTH][4], ls[MAX_SEQUENCE_LENGTH][4];
		char eligible_cs[MAX_SEQUENCE_LENGTH][5], no_of_eligible_cs=0;

		if(!done_push_types[current_stage].x)
		{
			uint4 cuda_cnt_k ;//(!sequence_type)? rbwt_cuda_occ4(k-1): bwt_cuda_occ4(k-1);
			uint4 cuda_cnt_l ;//(!sequence_type)? rbwt_cuda_occ4(l): bwt_cuda_occ4(l);
			ks[current_stage][0] = bwt->L2[0] + cuda_cnt_k.x + 1;
			ls[current_stage][0] = bwt->L2[0] + cuda_cnt_l.x;
			ks[current_stage][1] = bwt->L2[1] + cuda_cnt_k.y + 1;
			ls[current_stage][1] = bwt->L2[1] + cuda_cnt_l.y;
			ks[current_stage][2] = bwt->L2[2] + cuda_cnt_k.z + 1;
			ls[current_stage][2] = bwt->L2[2] + cuda_cnt_l.z;
			ks[current_stage][3] = bwt->L2[3] + cuda_cnt_k.w + 1;
			ls[current_stage][3] = bwt->L2[3] + cuda_cnt_l.w;

			if (ks[current_stage][0] <= ls[current_stage][0])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 0;
			}
			if (ks[current_stage][1] <= ls[current_stage][1])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 1;
			}
			if (ks[current_stage][2] <= ls[current_stage][2])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 2;
			}
			if (ks[current_stage][3] <= ls[current_stage][3])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 3;
			}
			eligible_cs[current_stage][4] = no_of_eligible_cs;
		}else
		{
			no_of_eligible_cs = eligible_cs[current_stage][4];
		}

		// test whether difference is allowed
		allow_diff = 1;
		allow_M = 1;

		if (i)
		{
			if (bids[i-1] > m -1)
			{
				allow_diff = 0;
			}else if (bids[i-1] == m-1 && bids[i] == m-1 && widths[i-1] == widths[i])
			{
				allow_M = 0;
			}
		}

		//donepushtypes stores information for each stage whether a prospective daughter node has been evaluated or not
		//donepushtypes[current_stage].x  exact match, =0 not done, =1 done
		//donepushtypes[current_stage].y  mismatches, 0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].z  deletions, =0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].w  insertions match, =0 not done, =1 done
		//.z and .w are shared among gap openings and extensions as they are mutually exclusive


		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// exact match
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//try exact match first
		if (!done_push_types[current_stage].x)
		{
			#if DEBUG_LEVEL > 8
			printf("trying exact\n");
			#endif

			//shifted already
			int c = str[i];
			//if (start_pos) c = 3;
			done_push_types[current_stage].x = 1;
			if (c < 4)
			{
				#if DEBUG_LEVEL > 8
				printf("c:%i, i:%i\n",c,i);
				 printf("k:%u\n",ks[current_stage][c]);
				 printf("l:%u\n",ls[current_stage][c]);
				#endif

				if (ks[current_stage][c] <= ls[current_stage][c])
				{
					#if DEBUG_LEVEL > 8
					printf("ex match found\n");
					#endif

					cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape, STATE_M, 0, current_stage+1);
					current_stage++;
					continue;
				}
			}
		}else if (score == worst_tolerated_score)
		{
			allow_diff = 0;
		}

		//if (i<20) break;
		if (allow_diff)
		{
			#if DEBUG_LEVEL > 8
			printf("trying inexact...\n");
			#endif
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// mismatch
			////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (done_push_types[current_stage].y < no_of_eligible_cs) //check if done before
			{
				int c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
				done_push_types[current_stage].y++;
				if (allow_M) // daughter node - mismatch
				{
					if (score + options_cuda.s_mm <= worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (c != str[i])
						{
							// TODO is the debug message ok?
							#if DEBUG_LEVEL > 8
							 printf("mismatch confirmed\n");
							#endif
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}else if (done_push_types[current_stage].y < no_of_eligible_cs)
						{
							c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
							done_push_types[current_stage].y++;
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}
					}
				}
			}

				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Indels (Insertions/Deletions)
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				if (!e_state) // daughter node - opening a gap insertion or deletion
				{
					if (score + options_cuda.s_gapo <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (e_n_gapo < allow_gap)
						{
							if (!done_push_types[current_stage].w)
							{	//insertions
								done_push_types[current_stage].w = 1;
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
										current_stage++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i, k, l, e_n_mm, e_n_gapo + 1, e_n_gape, STATE_I, 1, current_stage);
										continue;
								}
							}
							else if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
							{	//deletions
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									done_push_types[current_stage].z++;
									cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo + 1, e_n_gape, STATE_D, 1, current_stage+1);
									current_stage++; //advance stage number by 1
									continue;
								}
								else
								{
									done_push_types[current_stage].z++;
								}
							}
						}
					}
				}else if (e_state == STATE_I) //daughter node - extend an insertion entry
				{
					if(!done_push_types[current_stage].w)  //check if done before
					{
						done_push_types[current_stage].w = 1;
						if (e_n_gape < opt->max_gape)  //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									current_stage++; //advance stage number by 1
									cuda_dfs_push(entries_info, entries_scores,  done_push_types, i, k, l, e_n_mm, e_n_gapo, e_n_gape + 1, STATE_I, 1, current_stage);
									continue; //skip the rest and proceed to next stage
								}
							}
						}
					}
				}else if (e_state == STATE_D) //daughter node - extend a deletion entry
				{
					occ = l - k + 1;
					if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
					{
						if (e_n_gape < opt->max_gape) //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								if (e_n_gape + e_n_gapo < max_diff || occ < options_cuda.max_del_occ)
								{
									unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;

									if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
									{
										int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
										done_push_types[current_stage].z++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape + 1, STATE_D, 1, current_stage+1);
										current_stage++; //advance stage number
										continue;
									}
								}
							}
						}
						else
						{
							done_push_types[current_stage].z++;
						}
					}
				} //end else if (e_state == STATE_D)*/

		}//end if (!allow_diff)
		current_stage--;

	} //end do while loop



	aln->no_of_alignments = n_aln;

	return best_score;
}
#endif

__device__ int cuda_dfs_match(uint32_t * global_bwt, int len, const unsigned char *str,  unsigned int *widths,  unsigned char *bids, const gap_opt_t *opt, alignment_meta_t *aln, barracuda_aln1_t* alns, init_info_t* init, int best_score, int full_len, char seeding)
//This function tries to find the alignment of the sequence and returns SA coordinates, no. of mismatches, gap openings and extensions
//It uses a depth-first search approach rather than breath-first as the memory available in CUDA is far less than in CPU mode
//The search rooted from the last char [len] of the sequence to the first with the whole bwt as a ref from start
//and recursively narrow down the k(upper) & l(lower) SA boundaries until it reaches the first char [i = 0], if k<=l then a match is found.
{

#if DEBUG_LEVEL > 6
	printf ("in dfs match\n");
#endif
	//Initialisations
	int start_pos = init->start_pos;
	int remain = full_len - len - start_pos;
	aln->pos = init->start_pos + len;
	int best_diff = start_pos==0 ? opt->max_diff + 1 : init->best_diff;
	int best_cnt = init->best_cnt;
	const bwt_t * bwt = &bwt_cuda; // rbwt for sequence 0 and bwt for sequence 1;
	int current_stage = 0;
	ulong2 entries_info[MAX_ALN_LENGTH];
	uint2 position_data[MAX_ALN_LENGTH];
	uchar4 entries_scores[MAX_ALN_LENGTH];
	char4 done_push_types[MAX_ALN_LENGTH];
	int n_aln = 0;
	int loop_count = 0;
	const int max_count = options_cuda.max_entries;

	/*if(init->has_exact){
		best_score = 0;
	}
	else */if(start_pos) {
		best_score = init->score;
	}

	int max_diff_base, allow_gap, max_no_partial_hits;
	if(seeding){
		max_diff_base = opt->max_seed_diff;
		max_no_partial_hits = MAX_NO_SEEDING_PARTIALS;
	}
	else {
		max_diff_base = opt->max_diff;
		max_no_partial_hits = MAX_NO_REGULAR_PARTIALS;
	}

	if(options_cuda.fast){
		allow_gap = seeding ? 0 : options_cuda.max_gapo;
	}
	else {
		allow_gap = opt->max_gapo;
	}

	//There are other places that these can be replaced (i.e. not calculated on every while loop) but they (empirically) slow things down. No idea why! Arran
	uchar2 mode = {options_cuda.mode & BWA_MODE_NONSTOP, options_cuda.mode & BWA_MODE_GAPE};
#define USE_MODE_NONSTOP mode.x
#define USE_MODE_GAPE mode.y


	//printf("worst_score:%u, query sequence length: %i\n", best_score,len);

	//Initialise memory stores first in, last out

	//Zero-ing the values is unnecessary because the push overwrites everything
	cuda_dfs_initialize(entries_info, entries_scores, position_data, done_push_types/*, scores*/); //initialize initial entry, current stage set at 0 and done push type = 0

	//push first entry, the first char of the query sequence into memory stores for evaluation
	cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, len, init->lim_k, start_pos>0 ? init->lim_l : bwt->seq_len, init->cur_n_mm, init->cur_n_gapo, init->cur_n_gape, 0, 0, current_stage); //push initial entry to start

#if ARRAN_DEBUG_LEVEL > 1
	char bases[4] = {'A', 'C', 'G', 'T'};
	int start_score = init->cur_n_mm * options_cuda.s_mm + init->cur_n_gapo * options_cuda.s_gapo + init->cur_n_gape * options_cuda.s_gape;
	printf("\n---STARTING LOOP start_pos: %i	k: %lu	l: %lu	start_score: %i	best_score: %i", start_pos, init->lim_k, entries_info[0].y, start_score, best_score);
#endif

	ulong4 cuda_cnt_k = {0,0,0,0};
	while(current_stage >= 0)
	{
		int i,j, m;
		int hit_found, allow_diff, allow_M;
		bwtint_t k, l;
		char e_n_mm, e_n_gapo, e_n_gape, e_state;
		loop_count ++;
		//define break from loop conditions
		if ((remain==0 && n_aln == options_cuda.max_aln) || n_aln==max_no_partial_hits) {
#if DEBUG_LEVEL > 7 || ARRAN_DEBUG_LEVEL > 1
			printf("\n\n*****breaking on n_aln == max_aln\n");
#endif
			break;
		}
		if (remain==0 && best_cnt > options_cuda.max_top2) {
#if DEBUG_LEVEL > 7 || ARRAN_DEBUG_LEVEL > 1
			printf("\n\n*****breaking on best_cnt>... %i>%i\n", best_cnt, options_cuda.max_top2);
#endif
			break;
		}
		if (loop_count > max_count) {
#if DEBUG_LEVEL > 7 || ARRAN_DEBUG_LEVEL > 1
			printf("\n\n*****loop_count > max_count\n");
#endif
			break;

		}

		//put extracted entry into local variables
		k = entries_info[current_stage].x; // SA interval
		l = entries_info[current_stage].y; // SA interval
		//i = entries_info[current_stage].z; // length
		i = position_data[current_stage].x;
		//printf(",%i/%i", i, current_stage);

		e_n_mm = entries_scores[current_stage].x; // no of mismatches
		e_n_gapo = entries_scores[current_stage].y; // no of gap openings
		e_n_gape = entries_scores[current_stage].z; // no of gap extensions
		e_state = entries_scores[current_stage].w; // state (M/I/D)

		//calculate score
		//Storing this in position_data[current_stage].z has been tested by including score+relevant_penalty in
		//calls to cuda_dfs_push but it is slower than calculating each time
		int score = e_n_mm * options_cuda.s_mm + e_n_gapo * options_cuda.s_gapo + e_n_gape * options_cuda.s_gape;
		int worst_tolerated_score = (USE_MODE_NONSTOP)? 1000: best_score + options_cuda.s_mm;
		if(score > worst_tolerated_score) {
#if DEBUG_LEVEL > 7 || ARRAN_DEBUG_LEVEL > 1
			printf("\n\n*****breaking on score(%i) > worst_tolerated_score(%i)\n", score, worst_tolerated_score);
#endif
			break;
		}

		int max_diff = max_diff_base;

		//calculate the allowance for differences
		m = max_diff - e_n_mm - e_n_gapo;

#if DEBUG_LEVEL > 8
		printf("dfs stage %u:",current_stage);
		printf("k:%lu, l: %lu, i: %u, diff remaining :%u, best_score: %u, n_mm:%u\n", k, l,i,m, best_score, e_n_mm);
#endif

		if (USE_MODE_GAPE) m -= e_n_gape;

		// check if the entry is outside boundary or is over the max diff allowed)
		if (m < 0 || (i > 0 && m < bids[i-1+remain]))
		{
#if DEBUG_LEVEL > 8 || ARRAN_DEBUG_LEVEL > 1
			printf("breaking: %d, m:%d\n", bids[i-1+remain],m);
#endif
			current_stage --;
			continue;
		}

		// check whether a hit (full sequence when it reaches the last char, i.e. i = 0) is found, if it is, record the alignment information
		hit_found = 0;
		if (!i)
		{
			hit_found = 1;
		}else if (!m) // alternatively if no difference is allowed, just do exact match)
		{
			if (e_state == STATE_M ||(options_cuda.mode&BWA_MODE_GAPE) || e_n_gape == opt->max_gape)
			{
				if (bwt_cuda_match_exact(global_bwt, i, str, &k, &l))
				{
#if ARRAN_DEBUG_LEVEL > 1
	printf("\n[aln_core][exact_match] n_aln: %i pos: %i k: %lu l: %lu mm: %i gapo: %i gape: %i state: %i score: %i seq: %i", n_aln, start_pos + len - i, k, l, e_n_mm, e_n_gapo, e_n_gape, e_state, score, init->sequence_id);
#endif
					hit_found = 1;
				}else
				{
#if ARRAN_DEBUG_LEVEL > 1
	printf("\n[aln_core][no_exact] n_aln: %i pos: %i k: %lu l: %lu mm: %i gapo: %i gape: %i state: %i score: %i seq: %i", n_aln, start_pos + len - i, k, l, e_n_mm, e_n_gapo, e_n_gape, e_state, score, init->sequence_id);
#endif
					current_stage --;
					continue; // if there is no hit, then go backwards to parent stage
				}
			}
		}

		if (hit_found)
		{
			// action for found hits
			//int do_add = 1;
			if (score < best_score)
			{
				best_score = score;
				best_diff = e_n_mm + e_n_gapo + (options_cuda.mode & BWA_MODE_GAPE) * e_n_gape;
				best_cnt = 0; //reset best cnt if new score is better
				if (!(options_cuda.mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;

			if (e_n_gapo)
			{ // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array unless the new score is better.
				for (j = 0; j < n_aln; ++j)
					if (alns[j].k == k && alns[j].l == l) break;
				if (j < n_aln)
				{
					if (score < alns[j].score)
						{
							alns[j].score = score;
							alns[j].n_mm = e_n_mm;
							alns[j].n_gapo = e_n_gapo;
							alns[j].n_gape = e_n_gape;
							alns[j].best_cnt = best_cnt;
						}
					//do_add = 0;
					hit_found = 0;
				}
			}

			if (hit_found)
			{ // append result the alignment record array
				gap_stack_shadow_cuda(l - k + 1, bwt->seq_len, position_data[current_stage].y, widths, bids);
					// record down number of mismatch, gap open, gap extension and a??

					alns[n_aln].n_mm = entries_scores[current_stage].x;
					alns[n_aln].n_gapo = entries_scores[current_stage].y;
					alns[n_aln].n_gape = entries_scores[current_stage].z;
					// the suffix array interval
					alns[n_aln].k = k;
					alns[n_aln].l = l;
					alns[n_aln].score = score;
					alns[n_aln].best_cnt = best_cnt;
#if ARRAN_DEBUG_LEVEL > 1
					printf("\n[aln_core][hit_found] n_aln: %i pos: %i k: %lu l: %lu mm: %i gapo: %i gape: %i state: %i score: %i seq: %i", n_aln, start_pos + len - i, k, l, e_n_mm, e_n_gapo, e_n_gape, e_state, score, init->sequence_id);
#endif

					++n_aln;

			}
			current_stage --;
			continue;
		}
		// proceed and evaluate the next base on sequence
		--i;

		// retrieve Occurrence values and determine all the eligible daughter nodes, done only once at the first instance and skip when it is revisiting the stage
		bwtint_t ks[MAX_ALN_LENGTH][4], ls[MAX_ALN_LENGTH][4];
		char eligible_cs[MAX_ALN_LENGTH][5], no_of_eligible_cs=0;

		if(!done_push_types[current_stage].x)
		{
			if(k) {
				cuda_cnt_k = bwt_cuda_occ4(global_bwt, k-1);
			}
			ulong4 cuda_cnt_l = bwt_cuda_occ4(global_bwt, l);
			ks[current_stage][0] = bwt->L2[0] + cuda_cnt_k.x + 1;
			ls[current_stage][0] = bwt->L2[0] + cuda_cnt_l.x;
			ks[current_stage][1] = bwt->L2[1] + cuda_cnt_k.y + 1;
			ls[current_stage][1] = bwt->L2[1] + cuda_cnt_l.y;
			ks[current_stage][2] = bwt->L2[2] + cuda_cnt_k.z + 1;
			ls[current_stage][2] = bwt->L2[2] + cuda_cnt_l.z;
			ks[current_stage][3] = bwt->L2[3] + cuda_cnt_k.w + 1;
			ls[current_stage][3] = bwt->L2[3] + cuda_cnt_l.w;

			if (ks[current_stage][0] <= ls[current_stage][0])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 0;
			}
			if (ks[current_stage][1] <= ls[current_stage][1])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 1;
			}
			if (ks[current_stage][2] <= ls[current_stage][2])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 2;
			}
			if (ks[current_stage][3] <= ls[current_stage][3])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 3;
			}
			eligible_cs[current_stage][4] = no_of_eligible_cs;

#if ARRAN_DEBUG_LEVEL > 1
			if(start_pos + len - i ARRAN_DEBUG_POSITION){
				printf("\nScore / Worst: %i/%i", score, worst_tolerated_score);
				printf("\n%i %c i: %i k: %lu l: %lu OK: %i\n", start_pos + len - i, bases[str[i]], i, k, l, no_of_eligible_cs);
				for(int q=0; q<len; q++){
					printf("%c", bases[str[q]]);
				}
				for(int q=0; q<4; q++){
					printf("\n%c	l: %lu	k: %lu %s", bases[q], ks[current_stage][q], ls[current_stage][q], ks[current_stage][q]<=ls[current_stage][q] ? "OK" : "");
				}
			}
#endif
		}else
		{
			no_of_eligible_cs = eligible_cs[current_stage][4];
		}

		// test whether difference is allowed
		allow_diff = 1;
		allow_M = 1;

		if (i)
		{
			if (bids[i-1+remain] > m -1)
			{
				allow_diff = 0;
				//printf("\n...%i %i	%i", real_pos, bids[i-1+remain], m);
			}else if (bids[i-1+remain] == m-1 && bids[i+remain] == m-1 && widths[i-1+remain] == widths[i+remain])
			{
				allow_M = 0;
			}
		}

		//donepushtypes stores information for each stage whether a prospective daughter node has been evaluated or not
		//donepushtypes[current_stage].x  exact match, =0 not done, =1 done
		//donepushtypes[current_stage].y  mismatches, 0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].z  deletions, =0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].w  insertions match, =0 not done, =1 done
		//.z and .w are shared among gap openings and extensions as they are mutually exclusive


		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// exact match
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//try exact match first
		if (!done_push_types[current_stage].x)
		{
#if DEBUG_LEVEL >8
			printf("trying exact\n");
#endif
			int c = str[i];

			done_push_types[current_stage].x = 1;
			if (c < 4)
			{
#if DEBUG_LEVEL > 8
				printf("c:%i, i:%i\n",c,i);
				printf("k:%u\n",ks[current_stage][c]);
				printf("l:%u\n",ls[current_stage][c]);
#endif

				if (ks[current_stage][c] <= ls[current_stage][c])
				{
					#if DEBUG_LEVEL > 8
					printf("ex match found\n");
					#endif

#if ARRAN_DEBUG_LEVEL > 1 && ARRAN_PRINT_DFS > 0 && ARRAN_PRINT_EXACT_MATCHES > 0
	printf("\n--- E %i %c stg: %i", start_pos + len - i, bases[c], current_stage);
#endif
					cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape, STATE_M, 0, current_stage+1);
					current_stage++;
					continue;
				}
			}
		}else if (score == worst_tolerated_score)
		{
			allow_diff = 0;
		}

		if (allow_diff)
		{
		#if DEBUG_LEVEL > 8
			printf("trying inexact...\n");
		#endif
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// mismatch
			////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (done_push_types[current_stage].y < no_of_eligible_cs) //check if done before
			{
				int c = eligible_cs[current_stage][(done_push_types[current_stage].y)];

				done_push_types[current_stage].y++;
				if(c==str[i]){
					c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
					done_push_types[current_stage].y++;
				}
				if (allow_M) // daughter node - mismatch
				{
					if (score + options_cuda.s_mm <= worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (done_push_types[current_stage].y <= no_of_eligible_cs)
						{
#if ARRAN_DEBUG_LEVEL > 1 && ARRAN_PRINT_DFS > 0 && ARRAN_PRINT_MISMATCHES > 0
	printf("\n### M %i %c stg: %i", start_pos + len - i, bases[c], current_stage);
#endif
							cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}
					}
				}
			}



				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Indels (Insertions/Deletions)
				////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define USE_OLD_GAP_CODE 0
#if USE_OLD_GAP_CODE==0

			unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;

			signed char * done_type;
			uint2 push_data;
			uchar4 gap_meta, gap_meta2;

#define GAP_TYPE gap_meta.x
#define GAP_PENALTY gap_meta.y
#define GAP_COUNT gap_meta.z
#define GAP_LIMIT gap_meta.w
#define GAP_INC gap_meta2.x
#define TYPE_LIMIT gap_meta2.y
#define DELTA_I gap_meta2.z
#define DEL_MAX gap_meta2.w
#define PUSH_K push_data.x
#define PUSH_L push_data.y

			if(e_state){ //extending a gap
				GAP_TYPE = e_state;
				GAP_PENALTY = options_cuda.s_gape;
				GAP_COUNT = e_n_gape;
				GAP_LIMIT = opt->max_gape;
				GAP_INC = 0;
			}
			else { //opening a gap
				GAP_TYPE = !done_push_types[current_stage].w ? STATE_I : STATE_D;
				GAP_PENALTY = options_cuda.s_gapo;
				GAP_COUNT = e_n_gapo;
				GAP_LIMIT = allow_gap;
				GAP_INC = 1;
			}

			if(GAP_TYPE & STATE_I){
				done_type = &(done_push_types[current_stage].w);
				TYPE_LIMIT = 1;
				PUSH_K = k;
				PUSH_L = l;
				DELTA_I = 0;
				DEL_MAX = 1;
			}
			else {
				done_type = &(done_push_types[current_stage].z);
				TYPE_LIMIT = no_of_eligible_cs;
				PUSH_K =	ks[current_stage][(eligible_cs[current_stage][(done_push_types[current_stage].z)])];
				PUSH_L =	ls[current_stage][(eligible_cs[current_stage][(done_push_types[current_stage].z)])];
				DELTA_I = 1;
				// we only checked max_del_occ when extending a deletion and !e_state means we are opening a gap
				DEL_MAX = !e_state || e_n_gape + e_n_gapo < max_diff || l - k < options_cuda.max_del_occ;
			}

			(*done_type)++;
			// the order of these if-statements has been optimised through trial & error / bubble sort
			if(i + remain >= options_cuda.indel_end_skip + tmp && start_pos + len - i >= options_cuda.indel_end_skip + tmp){
				if(score+GAP_PENALTY <= worst_tolerated_score){
					if(GAP_COUNT < GAP_LIMIT){
						if(*done_type <= TYPE_LIMIT){
							if(DEL_MAX){
#if ARRAN_DEBUG_LEVEL > 1 && ARRAN_PRINT_DFS > 0 && ARRAN_PRINT_GAPS > 0
	printf("\n@@@ G%s %s %i	stg: %i", e_state ? "E" : "O", GAP_TYPE & STATE_I ? "I" : "D", start_pos + len - i, current_stage);
#endif
								current_stage++;
								cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i + DELTA_I, PUSH_K, PUSH_L, e_n_mm, e_n_gapo + GAP_INC, e_n_gape + 1 - GAP_INC, GAP_TYPE, 1, current_stage);
								continue;
							}
						}
					}
				}
			}



#else

			unsigned int occ;
			//int allow_gap = 0;
			if(allow_gap){
				if (!e_state) // daughter node - opening a gap insertion or deletion
				{
					if (score + options_cuda.s_gapo <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (e_n_gapo < opt->max_gapo)
						{
							if (!done_push_types[current_stage].w)
							{	//insertions
								done_push_types[current_stage].w = 1;
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
										current_stage++;
										cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i, k, l, e_n_mm, e_n_gapo + 1, e_n_gape, STATE_I, 1, current_stage);
										continue;
								}
							}
							else if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
							{	//deletions
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if(i + remain >= options_cuda.indel_end_skip + tmp && real_pos >= options_cuda.indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									done_push_types[current_stage].z++;
									cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo + 1, e_n_gape, STATE_D, 1, current_stage+1);
									current_stage++; //advance stage number by 1
									continue;
								}
								else
								{
									done_push_types[current_stage].z++;
								}
							}
						}
					}
				}else if (e_state == STATE_I) //daughter node - extend an insertion entry
				{
					if(!done_push_types[current_stage].w)  //check if done before
					{
						done_push_types[current_stage].w = 1;
						if (e_n_gape < opt->max_gape)  //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									current_stage++; //advance stage number by 1
									cuda_dfs_push(entries_info, entries_scores,  position_data, done_push_types, i, k, l, e_n_mm, e_n_gapo, e_n_gape + 1, STATE_I, 1, current_stage);
									continue; //skip the rest and proceed to next stage
								}
							}
						}
					}
				}else if (e_state == STATE_D) //daughter node - extend a deletion entry
				{
					occ = l - k + 1;
					if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
					{
						if (e_n_gape < opt->max_gape) //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								if (e_n_gape + e_n_gapo < max_diff || occ < options_cuda.max_del_occ)
								{
									unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;

									if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
									{
										int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
										done_push_types[current_stage].z++;
										cuda_dfs_push(entries_info, entries_scores, position_data, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape + 1, STATE_D, 1, current_stage+1);
										current_stage++; //advance stage number
										continue;
									}
								}
							}
						}
						else
						{
							done_push_types[current_stage].z++;
						}
					}
				} //end else if (e_state == STATE_D)
			} //end allow_gap

//end USE_OLD_GAP_CODE==1
#endif

		}//end if (!allow_diff)

		current_stage--;

	} //end do while loop

	aln->no_of_alignments = n_aln;
	aln->best_cnt = best_cnt;

#if ARRAN_DEBUG_LEVEL > 1
	printf("\n[aln_core][end_of_split_pass] n_aln: %i", n_aln);
#endif
//	printf("no of alignments %u\n",n_aln);

	return best_score;
}

__global__ void cuda_inexact_match_caller(uint32_t * global_bwt, int no_of_sequences, alignment_meta_t* global_alignment_meta, barracuda_aln1_t* global_alns, init_info_t* global_init, widths_bids_t *global_w_b, int best_score, char split_engage, char clump)
//CUDA kernal for inexact match on both strands
//calls bwt_cuda_device_calculate_width to determine the boundaries of the search space
//and then calls dfs_match to search for alignment using dfs approach
{
	// Block ID for CUDA threads, as there is only 1 thread per block possible for now
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences ) {

		unsigned char local_complemented_sequence[SEQUENCE_HOLDER_LENGTH];
		alignment_meta_t local_alignment_meta;
		barracuda_aln1_t local_alns[MAX_NO_PARTIAL_HITS];
		widths_bids_t local_w_b;
		init_info_t local_init;

		memset(&local_alignment_meta, 0, sizeof(alignment_meta_t));
		memset(&local_alns, 0, MAX_NO_PARTIAL_HITS*sizeof(barracuda_aln1_t));
		local_init = global_init[blockId];

		local_alignment_meta.sequence_id = local_init.sequence_id;
		local_w_b = global_w_b[local_init.sequence_id];

		//initialize local options for each query sequence
		gap_opt_t local_options = options_cuda;

		//get sequences from texture memory
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, local_init.sequence_id);

		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
		unsigned int last_read = ~0;
		unsigned int last_read_data = 0;
		unsigned int pass_length;
		char seeding = local_init.start_pos<local_options.seed_len;

		if(split_engage){
			if(seeding){
				pass_length = min(MAX_SEEDING_PASS_LENGTH, local_options.seed_len - local_init.start_pos);
			}
			else {
				pass_length = min(MAX_REGULAR_PASS_LENGTH, sequence_length - local_init.start_pos);
			}
		}
		else {
			pass_length = sequence_length;
		}

		if(clump){
			pass_length = min(pass_length, SUFFIX_CLUMP_WIDTH);
		}

		//the data is packed in a way that requires us to cycle through and change last_read and last_read_data
		int cycle_i = sequence_length - local_init.start_pos - pass_length;
		for(int i=0; i<cycle_i; i++){
			read_char(sequence_offset + i, &last_read, &last_read_data);
		}

		for (int i = 0; i < pass_length; i++){
			unsigned char c = read_char(sequence_offset + cycle_i + i, &last_read, &last_read_data );
			//local_sequence[i] = c;
			local_complemented_sequence[i] = c > 3 ? 4 : 3 - c;
		}

		local_alignment_meta.finished = (local_init.start_pos+pass_length == sequence_length);

		//initialize local options
		if (options_cuda.fnr > 0.0) local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_length, BWA_AVG_ERR, options_cuda.fnr);
		//make sure when max diff < seed diff will override seed diff settings
		if (local_options.max_seed_diff > local_options.max_diff) local_options.max_seed_diff = local_options.max_diff;
		if (local_options.max_diff < options_cuda.max_gapo) local_options.max_gapo = local_options.max_diff;

		//Align with two the 2-way bwt reference
		local_alignment_meta.best_score = local_init.score = cuda_dfs_match(global_bwt, pass_length, local_complemented_sequence, local_w_b.widths, local_w_b.bids, &local_options, &local_alignment_meta, local_alns, &local_init, best_score, sequence_length, seeding);

		// copy alignment info to global memory
		global_alignment_meta[blockId] = local_alignment_meta;
		int max_no_partial_hits = seeding ? MAX_NO_SEEDING_PARTIALS : MAX_NO_REGULAR_PARTIALS;
		memcpy(global_alns + blockId*max_no_partial_hits, local_alns, max_no_partial_hits*sizeof(barracuda_aln1_t));

	}
	return;
}

__global__ void cuda_prepare_widths(uint32_t * global_bwt, int no_of_sequences, widths_bids_t * global_w_b, char * global_N_too_high)
{
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	if ( blockId < no_of_sequences ) {

		widths_bids_t local_w_b;
		unsigned char local_sequence[MAX_SEQUENCE_LENGTH];

		gap_opt_t local_options = options_cuda;

		const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);

		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
		unsigned int last_read = ~0;
		unsigned int last_read_data;
		int N = 0;

		if (options_cuda.fnr > 0.0) local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_length, BWA_AVG_ERR, options_cuda.fnr);

		for(int i=0; i<sequence_length; ++i)
		{
			unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
			local_sequence[i] = c;
			if(c>3) N++;
			if(N>local_options.max_diff) break;
		}

		if(N<=local_options.max_diff){
			global_N_too_high[blockId] = 0;
			bwt_cuda_device_calculate_width(global_bwt, local_sequence, local_w_b.widths, local_w_b.bids, 0, sequence_length - local_options.seed_len - 1);
			bwt_cuda_device_calculate_width(global_bwt, local_sequence, local_w_b.widths, local_w_b.bids, sequence_length - local_options.seed_len - 1, sequence_length);
		}
		else {
			global_N_too_high[blockId] = 1;
		}

		//for(int i = 0; i < sequence_length; ++i){
			//printf("%u	%d	%u	%u\n",i,local_sequence[i],local_w_b.bids[i],local_w_b.widths[i]);
		//}

		global_w_b[blockId] = local_w_b;
	}
	return;
}

__global__ void cuda_find_exact_matches(uint32_t * global_bwt, int no_of_sequences, init_info_t* global_init, char* global_has_exact)
{
	//***EXACT MATCH CHECK***
	//everything that has been commented out in this function should be re-activated if being used
	//comments are only to stop compilation warnings
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	if ( blockId < no_of_sequences ) {
		//unsigned char local_complemented_sequence[SEQUENCE_HOLDER_LENGTH];
		init_info_t local_init;
		local_init = global_init[blockId];

		const uint2 sequence_info = tex1Dfetch(sequences_index_array, local_init.sequence_id);

		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
		unsigned int last_read = ~0;
		unsigned int last_read_data = 0;

		for (int i = 0; i < sequence_length; i++){
			unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
			if(c>3){
				return;
			}
			//local_complemented_sequence[i] = 3 - c;
		}

		//bwtint_t k = 0, l = bwt_cuda.seq_len;
		//global_init[blockId].has_exact = global_has_exact[local_init.sequence_id] = bwt_cuda_match_exact(global_bwt, sequence_length, local_complemented_sequence, &k, &l)>0 ? 1 : 0;
	}
	return;
}

#if USE_PETR_SPLIT_KERNEL > 0
//TODO: remove reverse alignment
__global__ void cuda_split_inexact_match_caller(int no_of_sequences, unsigned short max_sequence_length, alignment_meta_t* global_alignment_store, unsigned char cuda_opt)
//CUDA kernal for inexact match on a specified strand
// modified for split kernels
//calls bwt_cuda_device_calculate_width to determine the boundaries of the search space
//and then calls dfs_match to search for alignment using dfs approach
{

	// Block ID for CUDA threads, as there is only 1 thread per block possible for now
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;
	//Local store for sequence widths bids and alignments
	unsigned int local_sequence_widths[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence_bids[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence[MAX_SEQUENCE_LENGTH];
	unsigned char local_rc_sequence[MAX_SEQUENCE_LENGTH];



	alignment_meta_t local_alignment_store;

	//fetch the alignment store from memory
	local_alignment_store = global_alignment_store[blockId];

	int max_aln = options_cuda.max_aln;
	//initialize local options for each query sequence
	gap_opt_t local_options = options_cuda;

	const int pass_length = (options_cuda.seed_len > PASS_LENGTH)? options_cuda.seed_len : PASS_LENGTH;
	const int split_engage = pass_length + 6;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences )
	{
#if DEBUG_LEVEL > 5
		printf("start..\n");
#endif

		//get sequences from texture memory
		//const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);

		// sequences no longer align with the block ids
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, local_alignment_store.sequence_id);


		const unsigned int sequence_offset = sequence_info.x;
		unsigned int last_read = ~0;
		unsigned int last_read_data;

		//calculate new length - are we dealing with the last bit?
		int start_pos = local_alignment_store.start_pos;

		unsigned short process_length;

		// decide if this is the last part to process
		if (!start_pos && sequence_info.y >= split_engage) {
			// first round and splitting is going to happen, forcing if necessary
			process_length = min(sequence_info.y, pass_length);
		} else {
			// subsequent rounds or splitting is not happening
			if (sequence_info.y - start_pos < pass_length * 2) {
				// mark this pass as last
				local_alignment_store.finished = 1;
				if (sequence_info.y - start_pos > pass_length) {
					// "natural" splitting finish
					process_length = min(sequence_info.y, sequence_info.y%pass_length + pass_length);
				} else {
					// last pass of "forced" splitting
					process_length = sequence_info.y - start_pos;
				}

			} else {
				process_length = min(sequence_info.y, pass_length);
			}
		}


#if DEBUG_LEVEL > 7
		printf("process length: %d, start_pos: %d, sequence_length: %d\n", process_length, start_pos, sequence_info.y);
#endif
		//const unsigned short sequence_length = (!start_pos) ? process_length : sequence_info.y;

		// TODO can be slightly sped up for one directional matching
		for (int i = 0; i < process_length; ++i)
		{
			//offsetting works fine, again, from the back of the seq.
			// copies from the end to the beginning
			unsigned char c = read_char(sequence_offset + i + (sequence_info.y- process_length - start_pos), &last_read, &last_read_data );

			local_sequence[i] = c;

			if (local_options.mode & BWA_MODE_COMPREAD)
			{
				local_rc_sequence[i] = (c > 3)? c : (3 - c);
			}else
			{
				local_rc_sequence[i] = c;
			}
		}


#define SEEDING 0

		if (options_cuda.fnr > 0.0) {
			//tighten the search for the first bit of sequence
#if SEEDING == 1
			if (!start_pos) {
				local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_length, BWA_AVG_ERR, options_cuda.fnr);

			} else {
#endif
				local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_info.y, BWA_AVG_ERR, options_cuda.fnr);
#if SEEDING == 1
			}
#endif
		}

//		// TODO remove debug out
//		if (blockId == 1) {
//			printf("\n\nlocal maxdiff: %d\n", local_options.max_diff);
//			printf("local seed diff: %d\n\n", local_options.max_seed_diff);
//		}

		if (local_options.max_diff < options_cuda.max_gapo) local_options.max_gapo = local_options.max_diff;

		//the worst score is lowered from +1 (bwa) to +0 to tighten the search space esp. for long reads

		int worst_score = aln_score2(local_options.max_diff, local_options.max_gapo, local_options.max_gape, local_options);


#if DEBUG_LEVEL > 6
		printf("worst score: %d\n", worst_score);
#endif

		//test if there is too many Ns, if true, skip everything and return 0 number of alignments.
		int N = 0;
		for (int i = 0 ; i < process_length; ++i)
		{
			if (local_sequence[i] > 3) ++N;
			if (N > local_options.max_diff)
			{
#if DEBUG_LEVEL > 7
				printf("Not good quality seq, quitting kernel.\n");
#endif
				global_alignment_store[blockId].no_of_alignments = 0;
				return;
			}
		}

		int sequence_type = 0;
		sequence_type = (cuda_opt == 2) ? 1 : local_alignment_store.init.sequence_type;
		// Calculate w
		syncthreads();

#if DEBUG_LEVEL > 7
		printf("calc width..\n");
#endif

		// properly resume for reverse alignment
		if (sequence_type == 1) {
#if DEBUG_LEVEL > 6
			printf("reverse alignment...");
#endif
			//Align to forward reference sequence
			//bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);
			cuda_split_dfs_match(process_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		} else {
#if DEBUG_LEVEL > 6
			printf("normal alignment...");
#endif
			//bwt_cuda_device_calculate_width(local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);
			cuda_split_dfs_match(process_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		}


		// copy alignment info to global memory
		global_alignment_store[blockId] = local_alignment_store;

		// now align the second strand, only during the first run, subsequent runs do not execute this part
		if (!start_pos && !cuda_opt) {
			int no_aln = local_alignment_store.no_of_alignments;

			sequence_type = 1;
			// Calculate w
			syncthreads();
			//bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);

			//Align to reverse reference sequence
			syncthreads();
			cuda_split_dfs_match(process_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
#if DEBUG_LEVEL > 6
			printf("local_alignment_store.no_of_alignments: %d\n", local_alignment_store.no_of_alignments);
#endif

			// copy alignment info to global memory
			#if OUTPUT_ALIGNMENTS == 1
			short rc_no_aln = 0;
			while (rc_no_aln <= (max_aln + max_aln - no_aln) && rc_no_aln < local_alignment_store.no_of_alignments)
			{
				global_alignment_store[blockId].alignment_info[no_aln + rc_no_aln] = local_alignment_store.alignment_info[rc_no_aln];
				rc_no_aln++;
			}
			global_alignment_store[blockId].no_of_alignments = local_alignment_store.no_of_alignments + no_aln;
			#endif // OUTPUT_ALIGNMENTS == 1
		}
#if DEBUG_LEVEL > 6
		printf("kernel finished\n");
#endif
	}

	/*if (blockId < 5) {
		for (int x = 0; x<len; x++) {
			printf(".%d",str[x]);
		}
	}*/

	return;
}
#endif

// return the difference in second between two timeval structures
double diff_in_seconds(struct timeval *finishtime, struct timeval * starttime)
{
	double sec;
	sec=(finishtime->tv_sec-starttime->tv_sec);
	sec+=(finishtime->tv_usec-starttime->tv_usec)/1000000.0;
	return sec;
}

gap_opt_t *gap_init_bwaopt(gap_opt_t * opt)
{
	gap_opt_t *o;
	o = (gap_opt_t*)calloc(1, sizeof(gap_opt_t));
	o->s_mm = opt->s_mm;
	o->s_gapo = opt->s_gapo;
	o->s_gape = opt->s_gape;
	o->max_diff = opt->max_diff;
	o->max_gapo = opt->max_gapo;
	o->max_gape = opt->max_gape;
	o->indel_end_skip = opt->indel_end_skip;
	o->max_del_occ = opt->max_del_occ;
	o->max_entries = opt->max_entries;
	o->mode = opt->mode;
	o->seed_len = opt->seed_len;
	o->max_seed_diff = opt->max_seed_diff;
	o->fnr = opt->fnr;
	o->n_threads = 0;
	o->max_top2 = opt->max_top2;
	o->trim_qual = 0;
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void add_to_process_queue(init_bin_list * bin, barracuda_aln1_t * aln, alignment_meta_t * partial, unsigned int sequence_id){
#if ARRAN_DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_core][split_bin] pos: %i k: %llu l: %llu mm: %i gapo: %i gape: %i score: %i seq: %i", partial->pos, aln->k, aln->l, aln->n_mm, aln->n_gapo, aln->n_gape, aln->score, sequence_id);
#endif
	init_list * new_aln = (init_list*) malloc(sizeof(init_list));
	memset(new_aln, 0, sizeof(init_list));
	if(bin->aln_list){
		new_aln->next = bin->aln_list;
	}
	bin->aln_list = new_aln;
	bin->processed = 0;

	init_info_t	*to_queue = &(new_aln->init);

	to_queue->score = aln->score;
	to_queue->sequence_id = sequence_id;
	to_queue->best_cnt = partial->best_cnt;

	//each kernel updates its own start_pos for the next round based upon the pass length it determines
	//this allows us to run partial hits from different points in sequences or for sequences of different lengths
	to_queue->start_pos = partial->pos;

	to_queue->lim_k = aln->k;
	to_queue->lim_l = aln->l;
	to_queue->cur_n_mm = aln->n_mm;
	to_queue->cur_n_gapo = aln->n_gapo;
	to_queue->cur_n_gape = aln->n_gape;
}

void core_kernel_loop(int sel_device, int buffer, gap_opt_t *opt, bwa_seqio_t *ks, double total_time_used, uint32_t *global_bwt)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Core loop (this loads sequences to host memory, transfers to cuda device and aligns via cuda in CUDA blocks)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
{
	//Variable Initializations
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Timings
		double total_calculation_time_used = 0;
		struct timeval start, end;
		double time_used;

	//Sequence Reads related Variables
	// The length of the longest read;
		unsigned short max_sequence_length=0;
		// maximum read size from sequences in bytes
		unsigned int read_size = 0;
		// Number of sequences read per batchequences reside in global memory of GPU
		unsigned int no_of_sequences = 0;
		// total number of sequences read
		unsigned long long total_no_of_sequences = 0;
		// total number of reads in base pairs
		unsigned long long total_no_of_base_pair = 0;
		unsigned char * global_sequences = 0;
		// sequences reside in main memory of CPU
		unsigned char * main_sequences = 0;
		unsigned long long * main_suffixes = 0;
		// sequences index reside in global memory of GPU
		uint2 * global_sequences_index = 0;
		// sequences index reside in main memory of CPU
		uint2 * main_sequences_index = 0;

		// initializing pointer for device options from user
		gap_opt_t *options;

		// initial best score is the worst tolerated score without any alignment hit.
		const int best_score = aln_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);

		// global alignment stores for device
		//Variable for alignment result stores
		alignment_meta_t * global_alignment_meta_device;
		barracuda_aln1_t * global_alns_device;
		init_info_t * global_init_device;
		widths_bids_t * global_w_b_device;
		char * global_seq_flag_device, *global_seq_flag_host;
		// global alignment stores for host
		alignment_meta_t * global_alignment_meta_host, * global_alignment_meta_host_final;
		barracuda_aln1_t * global_alns_host, * global_alns_host_final;
		init_info_t * global_init_host;
#if USE_PETR_SPLIT_KERNEL > 0
		alignment_meta_t * global_alignment_meta_host_final;
#endif



	//CPU and GPU memory allocations
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	gettimeofday (&start, NULL);
		//allocate global_sequences memory in device
		cudaMalloc((void**)&global_sequences, (1ul<<(buffer))*sizeof(unsigned char));
		main_sequences = (unsigned char *)malloc((1ul<<(buffer))*sizeof(unsigned char));
		//suffixes for clumping
		main_suffixes = (unsigned long long *)malloc((1ul<<(buffer-3))*sizeof(unsigned long long));
		//allocate global_sequences_index memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		cudaMalloc((void**)&global_sequences_index, (1ul<<(buffer-3))*sizeof(uint2));
		main_sequences_index = (uint2*)malloc((1ul<<(buffer-3))*sizeof(uint2));
		//allocate and copy options (opt) to device constant memory
		cudaMalloc((void**)&options, sizeof(gap_opt_t));
		cudaMemcpy ( options, opt, sizeof(gap_opt_t), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol ( options_cuda, opt, sizeof(gap_opt_t), 0, cudaMemcpyHostToDevice);
		//allocate alignment stores for host and device
		cudaMalloc((void**)&global_alignment_meta_device, (1ul<<(buffer-3))*sizeof(alignment_meta_t));
		cudaMalloc((void**)&global_alns_device, MAX_NO_PARTIAL_HITS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		cudaMalloc((void**)&global_init_device, (1ul<<(buffer-3))*sizeof(init_info_t));
		cudaMalloc((void**)&global_w_b_device, (1ul<<(buffer-3))*sizeof(widths_bids_t));
		cudaMalloc((void**)&global_seq_flag_device, (1ul<<(buffer-3))*sizeof(char));
		//allocate alignment store memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		global_alignment_meta_host = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
		global_alns_host = (barracuda_aln1_t*)malloc(MAX_NO_PARTIAL_HITS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		global_alignment_meta_host_final = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
		global_alns_host_final = (barracuda_aln1_t*)malloc(MAX_NO_OF_ALIGNMENTS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		global_init_host = (init_info_t*)malloc((1ul<<(buffer-3))*sizeof(init_info_t));
		global_seq_flag_host = (char*)malloc((1ul<<(buffer-3))*sizeof(char));
#if USE_PETR_SPLIT_KERNEL > 0
		global_alignment_meta_host_final = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
#endif

	gettimeofday (&end, NULL);
	time_used = diff_in_seconds(&end,&start);
	total_time_used += time_used;

#if DEBUG_LEVEL > 0
	fprintf(stderr,"[aln_debug] Finished allocating CUDA device memory\n");
#endif


	//Core loop starts here
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	gettimeofday (&start, NULL);
	int loopcount = 0;
	// unsigned int cur_sequence_id = 0; //unique sequence identifier
	// determine block size according to the compute capability
	int blocksize, avg_length;
	char split_engage;
	cudaDeviceProp selected_properties;
	cudaGetDeviceProperties(&selected_properties, sel_device);
	if ((int) selected_properties.major > 1) {
		blocksize = 64;
	} else {
		blocksize = 320;
	}

	while ( ( no_of_sequences = copy_sequences_to_cuda_memory(ks, global_sequences_index, main_sequences_index, global_sequences, main_sequences, &read_size, &max_sequence_length, buffer, main_suffixes, SUFFIX_CLUMP_WIDTH) ) > 0 )
	{
		#define GRID_UNIT 32
		int gridsize = GRID_UNIT * (1 + int (((no_of_sequences/blocksize) + ((no_of_sequences%blocksize)!=0))/GRID_UNIT));
		dim3 dimGrid(gridsize);
		dim3 dimBlock(blocksize);

		avg_length = (read_size/no_of_sequences);
		split_engage = avg_length > SPLIT_ENGAGE;

		if(opt->seed_len > avg_length)
		{
			fprintf(stderr,"[aln_core] Warning! Specified seed length [%d] exceeds average read length, setting seed length to %d bp.\n", opt->seed_len, int ((read_size/no_of_sequences)>>1));
			opt->seed_len = read_size/no_of_sequences >> 1; //if specify seed length not valid, set to half the sequence length
		}

		if (!loopcount) fprintf(stderr, "[aln_core] Now aligning sequence reads to reference assembly, please wait..\n");

		if (!loopcount)	{
#if DEBUG_LEVEL > 0
			fprintf(stderr, "[aln_debug] Average read size: %dbp\n", read_size/no_of_sequences);
			fprintf(stderr, "[aln_debug] Using Reduced kernel\n");
			fprintf(stderr, "[aln_debug] Using SIMT with grid size: %u, block size: %d. ", gridsize,blocksize) ;
			fprintf(stderr,"\n[aln_debug] Loop count: %i\n", loopcount + 1);
#endif
			fprintf(stderr,"[aln_core] Processing %d sequence reads at a time.\n[aln_core] ", (gridsize*blocksize)) ;
		}

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);

		int run_no_sequences = no_of_sequences; //for compatibility to PETR_SPLIT_KERNEL only

		total_time_used+=time_used;

		// initialise the alignment stores
		memset(global_alignment_meta_host_final, 0, no_of_sequences*sizeof(alignment_meta_t));
		memset(global_alns_host_final, 0, no_of_sequences*MAX_NO_OF_ALIGNMENTS*sizeof(barracuda_aln1_t));
		for(int i=0; i<no_of_sequences; i++){
			global_alignment_meta_host_final[i].best_score = best_score;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Core match function per sequence readings
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);

#if DEBUG_LEVEL > 0
		fprintf(stderr,"\n[aln_debug] reduced kernel starts \n", time_used);
#endif
#if DEBUG_LEVEL > 3
		//printf("cuda opt:%d\n", cuda_opt);
#endif

		cudaError_t cuda_err;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Widths & Bids
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//in this case, seq_flag is used to note sequences that have too many N characters
		cuda_prepare_widths<<<dimGrid,dimBlock>>>(global_bwt, no_of_sequences, global_w_b_device, global_seq_flag_device);

		cudaDeviceSynchronize();
		cuda_err = cudaGetLastError();
		if(int(cuda_err))
		{
			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during width/bid preparation! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
			return;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Cull for too many Ns
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		memset(global_init_host, 0, no_of_sequences*sizeof(init_info_t));
		cudaMemcpy(global_seq_flag_host, global_seq_flag_device, no_of_sequences*sizeof(char), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		unsigned int no_to_process = 0;
		for(int i=0; i<no_of_sequences; i++){
			if(global_seq_flag_host[i]){
				memset(global_alignment_meta_host_final + i, 0, sizeof(alignment_meta_t));
				global_alignment_meta_host_final[i].sequence_id = i;
			}
			else {
				global_init_host[no_to_process].sequence_id = i;
				no_to_process++;
			}
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Clump based on matching suffixes
		// Assumes that sequences were sorted
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		unsigned char first=1, clumping;
		unsigned int prev_suffix, old_no_to_process=no_to_process; //, unique=0;
		suffix_bin_list *suff_lst=0, *currSuff, *suff_ptrs[no_of_sequences];
		suffix_seq_list *currSeq;

		if(SUFFIX_CLUMP_WIDTH){
			for(int i=0; i<no_to_process; i++){
				if(main_suffixes[i]!=prev_suffix || first){
#if ARRAN_DEBUG_LEVEL > 0
	fprintf(stderr, "\n[aln_core][suffix_bin] Creating bin for suffix %llu", main_suffixes[i]);
#endif
					currSuff = (suffix_bin_list*) malloc(sizeof(suffix_bin_list));
					memset(currSuff, 0, sizeof(suffix_bin_list));
					if(suff_lst){
						currSuff->next = suff_lst;
					}
					suff_lst = currSuff;
					first = 0;
					prev_suffix = main_suffixes[i];
					//unique++;
				}

				suff_ptrs[global_init_host[i].sequence_id] = suff_lst;
				currSeq = (suffix_seq_list*) malloc(sizeof(suffix_seq_list));
				memset(currSeq, 0, sizeof(suffix_seq_list));
				currSeq->sequence_id = global_init_host[i].sequence_id;
				if(suff_lst->seq){
					currSeq->next = suff_lst->seq;
				}
#if ARRAN_DEBUG_LEVEL > 0
	fprintf(stderr, "\n[aln_core][suffix_bin] Adding sequence %i to bin for suffix %llu", currSeq->sequence_id, main_suffixes[i]);
#endif
				suff_lst->seq = currSeq;
			}

			no_to_process = 0;
			currSuff = suff_lst;
			do {
				global_init_host[no_to_process].sequence_id = currSuff->seq->sequence_id;
				no_to_process++;
				currSuff = currSuff->next;
			}
			while(currSuff);

			clumping = old_no_to_process!=no_to_process;

#if ARRAN_DEBUG_LEVEL > -1
	fprintf(stderr, "\n[aln_core][suffix_clump] Width: %i - Reduced to %i of %i (%0.2f%%)", SUFFIX_CLUMP_WIDTH, no_to_process, old_no_to_process, 100*(1-float(no_to_process)/float(no_of_sequences)));
#endif

		}

		////////////////////////////////////////////////////////////////////////////
		// Checking for exact matches using bwt_cuda_match_exact() allows best_score to be set to 0 from the outset
		// which theoretically culls the DFS tree further. However, it is empirically slower on SRR063699_1.
		// Code has been left here to give a guide to the approach used.
		// NOTE: all blocks of associated code have been marked with a comment:
		//***EXACT MATCH CHECK***
		////////////////////////////////////////////////////////////////////////////

		//in this case, seq_flag is used to note sequences that have an exact match - allows setting of best_score=0 for every run
		//cuda_find_exact_matches<<<dimGrid,dimBlock>>>(global_bwt, no_to_process, global_init_device, global_seq_flag_device);
		//cudaDeviceSynchronize();
		//cuda_err = cudaGetLastError();
		//if(int(cuda_err))
		//{
//			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during exact match pre-check! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
//			return;
//		}

		fprintf(stderr, "'");
		cudaMemcpy(global_init_device, global_init_host, no_to_process*sizeof(init_info_t), cudaMemcpyHostToDevice);
		//cuda_find_exact_matches writes straight to global_init_device so we can launch the first kernel and then deal with global_seq_flag_device
		cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_to_process, global_alignment_meta_device, global_alns_device, global_init_device, global_w_b_device, best_score, split_engage, SUFFIX_CLUMP_WIDTH>0);
		fprintf(stderr, "'");

		//***EXACT MATCH CHECK***
		//store knowledge of an exact match to be copied into init struct during partial hit queueing
		//cudaMemcpy(global_seq_flag_host, global_seq_flag_device, no_of_sequences*sizeof(char), cudaMemcpyDeviceToHost);
		//for(int i=0; i<no_of_sequences; i++){
//				if(global_seq_flag_host[i]){
//					global_alignment_meta_host_final[i].has_exact = 1;
//				}
		//}


#if DEBUG_LEVEL > 0
		fprintf(stderr,"\n[aln_debug] kernel started, waiting for data... \n", time_used);
#endif
		// Did we get an error running the code? Abort if yes.
		cuda_err = cudaGetLastError();
		if(int(cuda_err))
		{
			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during first kernel run! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
			return;
		}

#if DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_debug] Last CUDA error message: %s\n[aln_core]", cudaGetErrorString(cuda_err));
#endif

		//Check time
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_calculation_time_used += time_used;
		total_time_used += time_used;
		fprintf(stderr, ".");
		// query device for error

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// retrieve alignment information from CUDA device to host
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);

		unsigned int process_capacity = (unsigned int) (1ul<<(buffer-3));

		alignment_meta_t	* partial, * final;
		barracuda_aln1_t	* aln, * final_aln;
		int max_score = aln_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
		init_bin_list		* bins[max_score]; //will be a sparse array but it's tiny so memory wasting is minimal
		for(int s=0; s<max_score; s++){
			bins[s] = 0;
		}

		bins[0] = (init_bin_list*) malloc(sizeof(init_bin_list));
		memset(bins[0], 0, sizeof(init_bin_list));

		int split_loop_count = 0;
		do {
			cudaDeviceSynchronize();
			if(!split_loop_count){
				fprintf(stderr, "'");
			}
			cudaMemcpy(global_alignment_meta_host, global_alignment_meta_device, no_to_process*sizeof(alignment_meta_t), cudaMemcpyDeviceToHost);
			int max_no_partial_hits = (!split_loop_count ? MAX_NO_SEEDING_PARTIALS : MAX_NO_REGULAR_PARTIALS);
			cudaMemcpy(global_alns_host, global_alns_device, max_no_partial_hits*no_to_process*sizeof(barracuda_aln1_t), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			if(!split_engage) break;

#if ARRAN_DEBUG_LEVEL > 0
	fprintf(stderr, "\n[aln_core][split_loop]");
#endif

			fprintf(stderr, ":");
			for(int i=0; i<no_to_process; i++){
				partial = global_alignment_meta_host + i;
				final = global_alignment_meta_host_final + partial->sequence_id;
#if ARRAN_DEBUG_LEVEL > 0
	if(partial->no_of_alignments==0){
		init_info_t partial_init = global_init_host[i];
		fprintf(stderr, "\n[aln_core][split_null] pos: %i k: %llu l: %llu mm: %i gapo: %i gape: %i state: %i seq: %i", partial_init.start_pos, partial_init.lim_k, partial_init.lim_l, partial_init.cur_n_mm, partial_init.cur_n_gapo, partial_init.cur_n_gape, partial_init.cur_state, partial->sequence_id);
	}
#endif

				unsigned long long	partial_offset = i*max_no_partial_hits,
									final_offset = partial->sequence_id*MAX_NO_OF_ALIGNMENTS;
				for(int j=0; j<partial->no_of_alignments; j++){
					aln = global_alns_host + partial_offset + j;
					if(partial->finished){
						if(final->no_of_alignments==MAX_NO_OF_ALIGNMENTS){
							break;
						}
						final_aln = global_alns_host_final + final_offset + final->no_of_alignments;

						final_aln->k = aln->k;
						final_aln->l = aln->l;
						final_aln->n_mm = aln->n_mm;
						final_aln->n_gapo = aln->n_gapo;
						final_aln->n_gape = aln->n_gape;
						final_aln->score = aln->score;

#if ARRAN_DEBUG_LEVEL > 0
	fprintf(stderr, "\n[aln_core][split_complete] pos: %i k: %llu l: %llu mm: %i gapo: %i gape: %i score: %i", partial->pos, aln->k, aln->l, aln->n_mm, aln->n_gapo, aln->n_gape, aln->score);
#endif

						if(aln->score < final->best_score){
							final->best_score = aln->score;
							final->best_cnt = 0;
						}
						if(aln->score==final->best_score){
							final->best_cnt += aln->best_cnt;
						}

						final->no_of_alignments++;
					}
					else { // partial not finished

						//splice the linked list and add our new node in position to keep the ordering by score
						//keep a copy of the pointer for quick access
						if(!bins[aln->score]){
							init_bin_list	* prev, //used to find the bin with the greatest score < aln->score (i.e. previous one in the sorted list)
											* new_bin = (init_bin_list*) malloc(sizeof(init_bin_list));

							bins[aln->score] = new_bin;
							memset(new_bin, 0, sizeof(init_bin_list));
							new_bin->score = aln->score;

							for(prev = bins[0]; prev->next && prev->next->score < aln->score; prev = prev->next){}

							if(prev->next){
								new_bin->next = prev->next;
							}
#if ARRAN_DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_core][score_list] making bin: %i after %i", aln->score, prev->score);
#endif
							prev->next = new_bin;
						}

						if(SUFFIX_CLUMP_WIDTH && clumping && !split_loop_count){
							currSeq = suff_ptrs[partial->sequence_id]->seq;
							do {
								//NB note currSeq->sequence_id - not partial->sequence_id as with standard runs
								add_to_process_queue(bins[aln->score], aln, partial, currSeq->sequence_id);
								currSeq = currSeq->next;
							}
							while(currSeq);
						}
						else {
							add_to_process_queue(bins[aln->score], aln, partial, partial->sequence_id);
						}

					}
				}

			}

			fprintf(stderr, ":");
			init_bin_list * bin = bins[0];
			bool more_bins = true;
			int bins_processed=0, bins_to_process=split_loop_count<2 ? 1 : 2;
			for(no_to_process=0; more_bins && bins_processed<=bins_to_process && no_to_process<process_capacity; no_to_process++){
				while(!(bin->aln_list)){
					bins_processed += bin->processed;
					bin->processed = 0; //for the next loop
					if(!(bin->next) || bins_processed==bins_to_process){
						more_bins = false;
						break;
					}
					bin = bin->next;
				}
				if(!more_bins){
					break;
				}

				init_list * to_queue = bin->aln_list;

				final = global_alignment_meta_host_final + to_queue->init.sequence_id;
				//***EXACT MATCH CHECK***
				//to_queue->init.has_exact = final->has_exact;
				if(final->no_of_alignments){
					if(
							final->no_of_alignments==opt->max_aln
							|| to_queue->init.score > final->best_score + opt->s_mm //worst_tolerated_score will never be high enough
							|| (to_queue->init.score==final->best_score && final->best_cnt >= opt->max_top2) //best_cnt culling before it is even queued
					){ //woot! cull the tree!
#if ARRAN_DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_core][dfs_cull] init_score: %i	final_score: %i", to_queue->init.score, final->best_score);
#endif
						bin->aln_list = to_queue->next;
						free(to_queue);
						no_to_process--;
						continue;
					}
					to_queue->init.score = final->best_score;
				}
				else {
					to_queue->init.score = best_score; //give as much leeway as possible until alignments have been found and then cull the DFS tree
					to_queue->init.best_cnt = 0;
				}
				bin->processed = 1;

#if ARRAN_DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_core][split_queue] pos: %i k: %llu l: %llu mm: %i gapo: %i gape: %i score: %i bin: %i seq: %i", to_queue->init.start_pos, to_queue->init.lim_k, to_queue->init.lim_l, to_queue->init.cur_n_mm, to_queue->init.cur_n_gapo, to_queue->init.cur_n_gapo, bin->score, bins_processed, to_queue->init.sequence_id);
#endif

				memcpy(global_init_host + no_to_process, &(to_queue->init), sizeof(init_info_t));
				bin->aln_list = to_queue->next;
				free(to_queue);
			}

			if(no_to_process>0){
				fprintf(stderr, "|");
#if ARRAN_DEBUG_LEVEL > 0
		fprintf(stderr, "\n[aln_core][split_process] no_to_process: %i", no_to_process);
#endif
				cudaMemcpy(global_init_device, global_init_host, no_to_process*sizeof(init_info_t), cudaMemcpyHostToDevice);

				int gridsize = GRID_UNIT * (1 + int (((no_to_process/blocksize) + ((no_to_process%blocksize)!=0))/GRID_UNIT));
				dim3 dimGrid(gridsize);
				cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_to_process, global_alignment_meta_device, global_alns_device, global_init_device, global_w_b_device, best_score, split_engage, 0);
				cuda_err = cudaGetLastError();
				if(int(cuda_err))
				{
					fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during split kernel run! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
					return;
				}
			}
			split_loop_count++;
		}
		while(no_to_process>0);

		//pop everything and free it
		init_bin_list_t * top = bins[0], * to_free;
		do {
			to_free = top;
			top = to_free->next;
			free(to_free);
		} while(top);

		if(SUFFIX_CLUMP_WIDTH){
			do {
				do {
					currSeq = suff_lst->seq;
					suff_lst->seq = suff_lst->seq->next;
					free(currSeq);
				}
				while(suff_lst->seq);
				currSuff = suff_lst;
				suff_lst = suff_lst->next;
				free(currSuff);
			}
			while(suff_lst);
		}

		if(split_engage){
			memcpy(global_alignment_meta_host, global_alignment_meta_host_final, no_of_sequences*sizeof(alignment_meta_t));
			memcpy(global_alns_host, global_alns_host_final, MAX_NO_OF_ALIGNMENTS*no_of_sequences*sizeof(barracuda_aln1_t));
		}


#if DEBUG_LEVEL > 0
		fprintf(stderr,"\n[aln_debug] Kernel finished, transfering data to host... \n", time_used);
#else
		fprintf(stderr,".");
#endif


#if USE_PETR_SPLIT_KERNEL > 0

		const int pass_length = (opt->seed_len > PASS_LENGTH)? opt->seed_len: PASS_LENGTH;
		const int split_engage = pass_length + 6;


		// which kernel are we running?
		char split_kernel = (read_size/no_of_sequences >= split_engage);

		split_kernel = 0;
#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] pass length %d, split engage %d.\n", pass_length, split_engage);
#endif

		if (!loopcount) fprintf(stderr, "[aln_core] Now aligning sequence reads to reference assembly, please wait..\n");

		if (!loopcount)	{
#if DEBUG_LEVEL > 0

			fprintf(stderr, "[aln_debug] Average read size: %dbp\n", read_size/no_of_sequences);

			if (split_kernel)
				fprintf(stderr, "[aln_debug] Using split kernel\n");
			else
				fprintf(stderr, "[aln_debug] Using normal kernel\n");
				fprintf(stderr,"[aln_core] Using SIMT with grid size: %u, block size: %d.\n[aln_core] ", gridsize,blocksize) ;
#endif


			fprintf(stderr,"[aln_core] Processing %d sequence reads at a time.\n[aln_core] ", (gridsize*blocksize)) ;
		}

		// zero out the final alignment store
		memset(global_alignment_meta_host_final, 0, (1ul<<(buffer-3))*sizeof(alignment_meta_t));

		// create host memory store which persists between kernel calls, on the stack
		main_alignment_store_host_t  main_store;
		memset(main_store.score_align, 0, MAX_SCORE*sizeof(align_store_lst *));


		run_no_sequences = no_of_sequences;
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);

		//fprintf(stderr,"time used: %f\n", time_used);

		total_time_used+=time_used;

		// initialise the alignment stores
		memset(global_alignment_meta_host, 0, (1ul<<(buffer-3))*sizeof(alignment_meta_t));

		for (int i = 0; i < no_of_sequences; i++)
		{
			alignment_meta_t* tmp = global_alignment_meta_host + i;

			// store the basic info to filter alignments into initialisation file
			tmp->init.lim_k = 0;
			tmp->init.lim_l = forward_seq_len;
			tmp->init.sequence_type = 0;
			tmp->start_pos = 0; //first part
			tmp->sequence_id = i; //cur_sequence_id; cur_sequence_id++;
			//if (!split_kernel) tmp->finished = 1;//mark as finished for normal kernel
		}

		// copy the initialised alignment store to the device
		cudaMemcpy (global_alignment_meta_device,global_alignment_meta_host, no_of_sequences*sizeof(alignment_meta_t), cudaMemcpyHostToDevice);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Core match function per sequence readings
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);

#if DEBUG_LEVEL > 3
		printf("cuda opt:%d\n", cuda_opt);
#endif

		fprintf(stderr,"[aln_debug] kernels run \n", time_used);
		//debug only

		if (split_kernel) {
			cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);
		} else {
			cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);
		}
		fprintf(stderr,"[aln_debug] kernels return \n", time_used);

		// Did we get an error running the code? Abort if yes.
		cudaError_t cuda_err = cudaGetLastError();
		if(int(cuda_err))
		{
			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
			return;
		}

		//Check time
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_calculation_time_used += time_used;
		total_time_used += time_used;
		fprintf(stderr, ".");
		// query device for error

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// retrieve alignment information from CUDA device to host
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);

		char cont = 2;
		do
		{
			fprintf(stderr,".");
			cudaMemcpy (global_alignment_meta_host, global_alignment_meta_device, no_of_sequences*sizeof(alignment_meta_t), cudaMemcpyDeviceToHost);

			// go through the aligned sequeces and decide which ones are finished and which are not
			int aligned=0;
			int alignments = 0;
			for (int i = 0; i < no_of_sequences; i++)
			{
				alignment_meta_t* tmp = global_alignment_meta_host + i;


				if (tmp->no_of_alignments > 0)
				{
					aligned += 1;
					alignments += tmp->no_of_alignments;
					//int seq_id = tmp->sequence_id;

					alignment_meta_t* final = global_alignment_meta_host_final + tmp->sequence_id;


					if (tmp->finished == 1 && final->no_of_alignments == 0) {
					// TODO debug seeding only
					//if (true) {
						memcpy(final, tmp, sizeof(alignment_meta_t)); //simply copy the alignment
#if DEBUG_LEVEL > 3
							printf("stored finished alignment for seq: %d\n", tmp->sequence_id);
#endif
					} else {
						// more processing needed, append if finished or enqueue otherwise
						for (int j = 0; j < tmp->no_of_alignments && j < MAX_NO_OF_ALIGNMENTS; j++)
						{
							if (tmp->finished == 1) {
								// append alignments to an existing entry
								int cur_no_aln = final->no_of_alignments;

								if (cur_no_aln + 1 < MAX_NO_OF_ALIGNMENTS) {
									final->alignment_info[cur_no_aln] = tmp->alignment_info[j];
									final->no_of_alignments = cur_no_aln + 1;
								} else {
									break;
								}

#if DEBUG_LEVEL > 3
								printf("stored finished alignment for seq: %d\n", tmp->sequence_id);
#endif
							} else {
#if DEBUG_LEVEL > 3
								printf("continue with unfinished seq: %d\n", tmp->sequence_id);
#endif
								// otherwise add them to another queue for processing
								int score = tmp->alignment_info[j].score;
								align_store_lst *cur_top = main_store.score_align[score];
								align_store_lst *new_top = (align_store_lst*) malloc( sizeof(align_store_lst) );

								new_top->val = tmp->alignment_info[j];
								new_top->sequence_id = tmp->sequence_id;
								new_top->next = cur_top;
								new_top->start_pos = tmp->start_pos;

								main_store.score_align[score] = new_top;
							}
						}
					}
				}
			}

			#if DEBUG_LEVEL > 0

			fprintf(stderr, "[aln_debug] seq. through: %i \n", aligned);
			fprintf(stderr, "[aln_debug] total alignments: %i \n", alignments);

			#endif

			//print out current new host alignment store
#if DEBUG_LEVEL > 3
			for (int j=0; j<MAX_SCORE; j++) {
					align_store_lst * cur_el = main_store.score_align[j];

					if (cur_el) {
						printf("Alignments with score: %d \n", j);
					}

					while(cur_el) {
						barracuda_aln1_t alignment = cur_el->val;
						int cur_len = main_sequences_index[cur_el->sequence_id].y;
						//print some info
						printf("Sequence: %d,  a:%d, k: %d, l: %d, mm: %d, gape: %d, gapo: %d, length: %d, processed: %d\n",cur_el->sequence_id, alignment.a, alignment.k, alignment.l, alignment.n_mm, alignment.n_gape, alignment.n_gapo, cur_len, cur_el->start_pos);

						cur_el = cur_el->next;
					}


			}
			printf("\n");
#endif



			int max_process = (1ul<<(buffer-3)); //taken from the line allocating the memory, maximum we can do in a single run

			int last_index = -1;


			//remove items from the list and free memory accordingly
			for(int i=0; i<MAX_SCORE && max_process > last_index+1; i++) {
				align_store_lst * cur_el = main_store.score_align[i];
				align_store_lst * tmp;

				while(cur_el  && max_process > last_index+1) {
					barracuda_aln1_t alignment = cur_el->val;


					// add alignment to the new store
					last_index++;
					alignment_meta_t* store_entry = global_alignment_meta_host + (last_index);

					// increment start_pos
					store_entry->start_pos = cur_el->start_pos + pass_length;

					store_entry->sequence_id = cur_el->sequence_id;
	//				store_entry->init.best_cnt = alignment.best_cnt;
	//				store_entry->init.best_diff = alignment.best_diff;
					store_entry->init.cur_n_gape = alignment.n_gape;
					store_entry->init.cur_n_gapo = alignment.n_gapo;
					store_entry->init.cur_n_mm = alignment.n_mm;
					store_entry->init.lim_k = alignment.k;
					store_entry->init.lim_l = alignment.l;
					store_entry->init.sequence_type = alignment.a;
					store_entry->no_of_alignments = 0; //change to 1 to see the prev. alignment

					tmp = cur_el;
					cur_el = cur_el->next;

					// update the main store to point to the new element
					main_store.score_align[i] = cur_el;

					free(tmp);
				}

			}

			no_of_sequences = last_index + 1;


			if (no_of_sequences > 0) {

#if DEBUG_LEVEL > 3
				printf("aligning %d sequences\n", no_of_sequences);
#endif

				// how many blocks in the current run
				gridsize = GRID_UNIT * (1 + int (((no_of_sequences/blocksize) + ((no_of_sequences%blocksize)!=0))/GRID_UNIT));
				dimGrid = gridsize;

				// transfer the data to the card again
				cudaMemcpy (global_alignment_meta_device,global_alignment_meta_host, no_of_sequences*sizeof(alignment_meta_t), cudaMemcpyHostToDevice);

				//run kernel again
				cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);

			}
			else {
#if DEBUG_LEVEL > 3
				printf("Nothing to align, finished \n");
#endif
				cont = 0;
			}

		} while(cont);
		// end of kernel loop

#endif

#if DEBUG_LEVEL > 0
		if(opt->bwa_output)
			fprintf(stderr,"[aln_debug] Writing alignment to disk in BWA compatible format...");
		else
			fprintf(stderr,"[aln_debug] Writing alignment to disk in old barracuda format...");
#endif


		#if STDOUT_STRING_RESULT == 1
		for (int i = 0; i < run_no_sequences; i++)
		{
			alignment_meta_t* tmp = global_alignment_meta_host + i;

			if (tmp->no_of_alignments > 0)
			{
			printf("Sequence %d", i);//tmp->sequence_id);
			printf(", no of alignments: %d\n", tmp->no_of_alignments);

				barracuda_aln1_t *tmp_aln = global_alns_host_final + i;
				for (int j = 0; j < tmp->no_of_alignments && j < MAX_NO_OF_ALIGNMENTS; j++)
				{
					//printf("  Aligned read %d, ",j+1);
					printf("  Aligned read, ");
					//printf("a: %d, ", tmp->alignment_info[j].a);
					printf("n_mm: %d, ", tmp_aln[j].n_mm);
					printf("n_gape: %d, ", tmp_aln[j].n_gape);
					printf("n_gapo: %d, ", tmp_aln[j].n_gapo);
					printf("k: %llu, ", tmp_aln[j].k);
					printf("l: %llu, ", tmp_aln[j].l);
					printf("score: %u\n", tmp_aln[j].score);
				}
			}


		}
		//}
		#endif // STDOUT_STRING_RESULT == 1

		#if STDOUT_BINARY_RESULT == 1
		for (int  i = 0; i < run_no_sequences; ++i)
		{
#if USE_PETR_SPLIT_KERNEL > 0
			alignment_meta_t* tmp = global_alignment_meta_host_final + i;
#else
			alignment_meta_t* tmp = global_alignment_meta_host + i;
#endif
			err_fwrite(&tmp->no_of_alignments, 4, 1, stdout);
			if (tmp->no_of_alignments)
			{
				unsigned long long aln_offset = i*MAX_NO_OF_ALIGNMENTS;
				barracuda_aln1_t * tmp_aln;
				if(opt->bwa_output)
				{
					bwt_aln1_t * output;
					output = (bwt_aln1_t*)malloc(tmp->no_of_alignments*sizeof(bwt_aln1_t));

					for (int j = 0; j < tmp->no_of_alignments; j++)
					{
						tmp_aln = global_alns_host_final + aln_offset + j;
						bwt_aln1_t * temp_output = output + j;
						//temp_output->a = tmp->alignment_info[j].a;
						temp_output->k = tmp_aln->k;
						temp_output->l = tmp_aln->l;
						temp_output->n_mm = tmp_aln->n_mm;
						temp_output->n_gapo = tmp_aln->n_gapo;
						temp_output->n_gape = tmp_aln->n_gape;
						temp_output->score = tmp_aln->score;
					}
					if(tmp->no_of_alignments > 1) aln_quicksort(output,0,tmp->no_of_alignments-1);
					err_fwrite(output, sizeof(bwt_aln1_t), tmp->no_of_alignments, stdout);
					free(output);
				}else
				{
					barracuda_aln1_t * output;
					output = (barracuda_aln1_t*)malloc(tmp->no_of_alignments*sizeof(barracuda_aln1_t));

					for (int j = 0; j < tmp->no_of_alignments; j++)
					{
						tmp_aln = global_alns_host_final + aln_offset + j;
						barracuda_aln1_t * temp_output = output + j;
						//temp_output->a = tmp_aln->a;
						temp_output->k = tmp_aln->k;
						temp_output->l = tmp_aln->l;
						temp_output->n_mm = tmp_aln->n_mm;
						temp_output->n_gapo = tmp_aln->n_gapo;
						temp_output->n_gape = tmp_aln->n_gape;
						temp_output->score = tmp_aln->score;
					}
					fwrite(output, sizeof(barracuda_aln1_t), tmp->no_of_alignments, stdout);
					free(output);
				}
			}
		}

		#endif // STDOUT_BINARY_RESULT == 1

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		//fprintf(stderr, "Finished outputting alignment information... %0.2fs.\n\n", time_used);
		fprintf (stderr, ".");
		total_no_of_base_pair+=read_size;
		total_no_of_sequences+=run_no_sequences;
		gettimeofday (&start, NULL);
		loopcount ++;
	}
	fprintf(stderr, "\n");

	//report if there is any CUDA error
	cudaError_t cuda_err = cudaGetLastError();
	if(int(cuda_err))
	{
		fprintf(stderr, "[aln_core] CUDA ERROR(s) reported at end of core loop! Message: %s\n", cudaGetErrorString(cuda_err));
	}

#if DEBUG_LEVEL > 0
	fprintf(stderr, "[aln_debug] ERROR message: %s\n", cudaGetErrorString( cudaGetLastError() ) );
#endif

	fprintf(stderr, "[aln_core] Finished!\n[aln_core] Total no. of sequences: %u, size in base pair: %u bp, average length %0.2f bp/sequence.\n", (unsigned int)total_no_of_sequences, (unsigned int)total_no_of_base_pair, (float)total_no_of_base_pair/(unsigned int)total_no_of_sequences);
	fprintf(stderr, "[aln_core] Alignment Speed: %0.2f sequences/sec or %0.2f bp/sec.\n", (float)(total_no_of_sequences/total_time_used), (float)(total_no_of_base_pair/total_time_used));
	fprintf(stderr, "[aln_core] Total program time: %0.2fs.\n", (float)total_time_used);

	//Free memory
	cudaFree(global_sequences);
	free(main_sequences);
	free(main_suffixes);
	cudaFree(global_sequences_index);
	free(main_sequences_index);
	cudaFree(global_alignment_meta_device);
	cudaFree(global_alns_device);
	cudaFree(global_seq_flag_device);
	free(global_alignment_meta_host);
	free(global_alns_host);
	free(global_alignment_meta_host_final);
	free(global_alns_host_final);
	free(global_seq_flag_host);

	return;
}


void cuda_alignment_core(const char *prefix, bwa_seqio_t *ks,  gap_opt_t *opt)
//Determines the availability of CUDA devices and
//calls core_kernel_loop();

//invokes CUDA kernels cuda_inexact_match_caller
{
	// For timing purpose only
	struct timeval start, end;
	double time_used;
	double total_time_used = 0;


	fprintf(stderr,"[aln_core] Running CUDA mode.\n");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CUDA options
	if (opt->max_entries < 0 )
	{
		opt->max_entries = 150000;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Pick Cuda device
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	size_t mem_available = 0, total_mem = 0; //, max_mem_available = 0;
	cudaDeviceProp properties;
	int num_devices;
	cudaGetDeviceCount(&num_devices);

	if (!num_devices)
	{
		fprintf(stderr,"[aln_core] Cannot find a suitable CUDA device! aborting!\n");
	}


	int sel_device = 0;
	if (opt->cuda_device == -1)
	{
		sel_device = detect_cuda_device();
		if(sel_device >= 0)
		{
			cudaSetDevice(sel_device);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		else
		{
			fprintf(stderr,"[aln_core] Cannot find a suitable CUDA device! aborting!\n");
			return;
		}
	}
	else if (opt->cuda_device >= 0)
	{
		 sel_device = opt->cuda_device;
		 cudaSetDevice(sel_device);
		 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		 cudaGetDeviceProperties(&properties, sel_device);
		 cudaMemGetInfo(&mem_available, &total_mem);

		 fprintf(stderr, "[aln_core] Using specified CUDA device %d, memory available %d MB.\n", sel_device, int(mem_available>>20));

	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy bwt occurrences array to from HDD to CPU then to GPU
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	gettimeofday (&start, NULL);
	bwtint_t seq_len;

	// pointer to bwt occurrence array in GPU
	uint32_t * global_bwt = 0;
	// total number of bwt_occ structure read in bytes
	unsigned long long bwt_read_size = 0;

	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[aln_core] Loading BWTs, please wait..\n");

	bwt_read_size = copy_bwts_to_cuda_memory(prefix, &global_bwt, mem_available>>20, &seq_len)>>20;

	// copy_bwt_to_cuda_memory
	// returns 0 if error occurs
	// mem_available in MiB not in bytes

	if (!bwt_read_size) return; //break

	gettimeofday (&end, NULL);
	time_used = diff_in_seconds(&end,&start);
	total_time_used += time_used;
	fprintf(stderr, "[aln_core] Finished loading reference sequence assembly, %u MB in %0.2fs (%0.2f MB/s).\n", (unsigned int)bwt_read_size, time_used, ((unsigned int)bwt_read_size)/time_used );


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// allocate GPU working memory
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Set memory buffer according to memory available
	cudaMemGetInfo(&mem_available, &total_mem);

	#if DEBUG_LEVEL > 0
	fprintf(stderr,"[aln_debug] mem left: %d MiB\n", int(mem_available>>20));
	#endif

	//stop if there isn't enough memory available

	int buffer = SEQUENCE_TABLE_SIZE_EXPONENTIAL;
	if ((mem_available>>20) < CUDA_TESLA)
	{
		buffer = buffer - 1; //this will half the memory usage by half to 675MB
		if(mem_available>>20 < (CUDA_TESLA >> 1))
		{
			fprintf(stderr,"[aln_core] Not enough memory to perform alignment (min: %d MB).\n", CUDA_TESLA >> 1);
			return;
		}
	}
	else
	{
		fprintf(stderr,"[aln_core] Sweet! Running with an enlarged buffer.\n");
	}

	//calls core_kernel_loop
	core_kernel_loop(sel_device, buffer, opt, ks, total_time_used, global_bwt);

	free_bwts_from_cuda_memory(global_bwt);

	return;
}

//////////////////////////////////////////
// CUDA detection code
//////////////////////////////////////////

int bwa_deviceQuery()
// Detect CUDA devices available on the machine, for quick CUDA test and for multi-se/multi-pe shell scripts
{
	int device, num_devices;
	cudaGetDeviceCount(&num_devices);
	if (num_devices)
		{
			  //fprintf(stderr,"[deviceQuery] Querying CUDA devices:\n");
			  for (device = 0; device < num_devices; device++)
			  {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  fprintf(stdout, "%d ", device);
					  fprintf(stdout, "%d %d%d\n", int(properties.totalGlobalMem>>20), int(properties.major),  int(properties.minor));

			  }
			  //fprintf(stderr,"[total] %d\n", device);
		}
	return 0;
}

int detect_cuda_device()
// Detect CUDA devices available on the machine, used in aln_core and samse_core
{
	int num_devices, device = 0;
	size_t mem_available = 0,
		   //total_mem = 0,
		   max_mem_available = 0;
	cudaGetDeviceCount(&num_devices);
	cudaDeviceProp properties;
	int sel_device = -1;

	if (num_devices >= 1)
	{
	     fprintf(stderr, "[detect_cuda_device] Querying CUDA devices:\n");
		 int max_cuda_cores = 0, max_device = 0;
		 for (device = 0; device < num_devices; device++)
		 {
			  cudaGetDeviceProperties(&properties, device);
			  mem_available = properties.totalGlobalMem;
			  //cudaMemGetInfo(&mem_available, &total_mem);
			  fprintf(stderr, "[detect_cuda_device]   Device %d ", device);
			  for (int i = 0; i < 256; i++)
			  {
				  fprintf(stderr,"%c", properties.name[i]);
			  }
			  //calculated by multiprocessors * 8 for 1.x and multiprocessors * 32 for 2.0, *48 for 2.1 and *192 for 3.0
			  //determine amount of memory available
				  int cuda_cores = 0;
				  if (properties.major == 1){
						  cuda_cores = properties.multiProcessorCount*8;

				  }else if (properties.major == 2){
					  if (properties.minor == 0){
						  cuda_cores = properties.multiProcessorCount*32;
					  }else{
						  cuda_cores = properties.multiProcessorCount*48;
					  }
				  }else if (properties.major ==3)
				  {
					  cuda_cores = properties.multiProcessorCount*192;
				  }

			  fprintf(stderr,", CUDA cores %d, global memory size %d MB, compute capability %d.%d.\n", int(cuda_cores), int(mem_available>>20), int(properties.major),  int(properties.minor));
			  if (max_cuda_cores <= cuda_cores) //choose the one with highest number of processors
			  {
					  max_cuda_cores = cuda_cores;
					  if (max_mem_available < mem_available) //choose the one with max memory
					  {
						      max_mem_available = mem_available;
							  max_device = device;
					  }
			  }
 		 }
		 if (max_mem_available>>20 >= MIN_MEM_REQUIREMENT)
		 {
			 sel_device = max_device;
			 fprintf(stderr, "[detect_cuda_device] Using CUDA device %d, global memory size %d MB.\n", max_device, int(max_mem_available>>20));
			 }
		 else
		 {
			 fprintf(stderr,"[detect_cuda_device] Cannot find a suitable CUDA device with > %d MB of memory available! aborting!\n", MIN_MEM_REQUIREMENT);
			 return -1;
		 }
	}
	else
	{
		 fprintf(stderr,"[detect_cuda_device] No CUDA device found! aborting!\n");
		 return -1;
	}
	return sel_device;
}

//////////////////////////////////////////
// End CUDA detection code
//////////////////////////////////////////

//////////////////////////////////////////
// Below is code for BarraCUDA CUDA samse_core
//////////////////////////////////////////
#if CUDA_SAMSE == 1



void report_cuda_error_CPU(const char * message)
{
	fprintf(stderr,"%s\n",message);
	exit(1);
}


// Texture.
texture<bwtint_t, 1, cudaReadModeElementType> sa_tex;
texture<bwtint_t, 1, cudaReadModeElementType> bwt_sa_tex;
texture<bwtint_t, 1, cudaReadModeElementType> rbwt_sa_tex;
texture<int, 1, cudaReadModeElementType> g_log_n_tex;

// Variables for information to do with GPU or software (e.g., no. of blocks).

const static int BLOCK_SIZE2 = 128;

static bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_ho;
static bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de;
static bwtint_t *seqs_sa_ho;
static bwtint_t *seqs_sa_de;
static uint8_t *seqs_mapQ_ho;
static uint8_t *seqs_mapQ_de;
static bwtint_t *seqs_pos_ho;
static bwtint_t *seqs_pos_de;


int prepare_bwa_cal_pac_pos_cuda1(
    unsigned int **global_bwt,
    unsigned int **global_rbwt,
    const char *prefix,
    bwtint_t **bwt_sa_de,
    bwtint_t **rbwt_sa_de,
    const bwt_t *bwt,
    const bwt_t *rbwt,
    const int *g_log_n_ho,
    int **g_log_n_de,
    const int g_log_n_len,
    int device)
{
    // mem_available in bytes not MiB
    size_t mem_available,total_mem;

    cudaSetDevice(device);
    cudaMemGetInfo(&mem_available, &total_mem);

    ////////////////////////////////////////////////////////////
    // Load BWT to GPU.
    ////////////////////////////////////////////////////////////

    // copy_bwt_occ_array_to_cuda_memory

	unsigned long long size_read = 0;

		if ( bwt != 0 )
		{
			//Original BWT
			size_read += bwt->bwt_size*sizeof(uint32_t);

			mem_available = mem_available - size_read;

			if(mem_available > 0)
			{
				//Allocate memory for bwt
				cudaMalloc((void**)global_bwt, bwt->bwt_size*sizeof(uint32_t));
				//copy bwt occurrence array from host to device and dump the bwt to save CPU memory
				cudaMemcpy (*global_bwt, bwt->bwt, bwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
				//bind global variable bwt to texture memory bwt_occ_array
				cudaBindTexture(0, bwt_occ_array, *global_bwt, bwt->bwt_size*sizeof(uint32_t));
				//copy bwt structure data to constant memory bwt_cuda structure
				cudaMemcpyToSymbol ( bwt_cuda, bwt, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
			}
			else
			{
				fprintf(stderr,"[samse_core] Not enough device memory to continue.\n");
				return 0;
			}


	#if DEBUG_LEVEL > 0
			fprintf(stderr,"[samse_debug] bwt loaded, mem left: %d MB\n", (int)(mem_available>>20));
	#endif
		}
		if ( rbwt != 0 )
		{
			//Reversed BWT
			size_read += bwt->bwt_size*sizeof(uint32_t);
			mem_available = mem_available - (bwt->bwt_size*sizeof(uint32_t));


	#if DEBUG_LEVEL > 0
			fprintf(stderr,"[samse_debug] rbwt loaded mem left: %d MB\n", (int)(mem_available>>20));
	#endif

			if (mem_available > 0)
			{
				//Allocate memory for rbwt
				cudaMalloc((void**)global_rbwt, rbwt->bwt_size*sizeof(uint32_t));
				//copy reverse bwt occurrence array from host to device and dump the bwt to save CPU memory
				cudaMemcpy (*global_rbwt, rbwt->bwt, rbwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
				//bind global variable rbwt to texture memory rbwt_occ_array
				cudaBindTexture(0, rbwt_occ_array, *global_rbwt, rbwt->bwt_size*sizeof(uint32_t));
				//copy rbwt structure data to constant memory bwt_cuda structure
				cudaMemcpyToSymbol ( rbwt_cuda, rbwt, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
			}
			else
			{
				fprintf(stderr,"[samse_core] Not enough device memory to continue.\n");
				return 0;
			}

		}

	// returns 0 if error occurs

    ////////////////////////////////////////////////////////////
    // Copy input data in "g_log_n" to device, and bind texture of "g_log_n_de".
    ////////////////////////////////////////////////////////////
    // Reserve memory.
    cudaMalloc((void**)g_log_n_de,sizeof(int)*g_log_n_len);
    report_cuda_error_GPU("[samse_core] Error reserving memory for \"g_log_n_de\".\n");

    // Copy data from host to device.
    cudaMemcpy(*g_log_n_de,g_log_n_ho,sizeof(int)*g_log_n_len,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("[samse_core] Error copying to \"g_log_n_de\".\n");

    // Bind texture.
    cudaBindTexture(0,g_log_n_tex,*g_log_n_de,sizeof(int)*g_log_n_len);
    report_cuda_error_GPU("[samse_core] Error binding texture to \"g_log_n_tex\".\n");

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy "sa" data of BWT and RBWT to device.
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    //fprintf(stderr,"[debug] bwt->n_sa: %u\n", bwt->n_sa);

    // Reserve memory for SA (BWT) on device.
    cudaMalloc(*&bwt_sa_de,sizeof(bwtint_t)*bwt->n_sa);
    report_cuda_error_GPU("Error reserving memory for \"bwt_sa_de\".\n");

    // Copy SA (BWT) to device.
    cudaMemcpy(*bwt_sa_de,bwt->sa,sizeof(bwtint_t)*bwt->n_sa,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("[samse_core] Error copying to \"bwt_sa_de\".\n");

    // Bind texture.
    cudaBindTexture(0,bwt_sa_tex,*bwt_sa_de,sizeof(bwtint_t)*bwt->n_sa);
    report_cuda_error_GPU("[samse_core] Error binding texture to \"bwt_sa_tex\".\n");

    // Reserve memory for SA (RBWT) on device.
    cudaMalloc(*&rbwt_sa_de,sizeof(bwtint_t)*rbwt->n_sa);
    report_cuda_error_GPU("[samse_core] Error reserving memory for \"rbwt_sa_de\".\n");

    // Copy SA (RBWT) to device.
    cudaMemcpy(*rbwt_sa_de,rbwt->sa,sizeof(bwtint_t)*rbwt->n_sa,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("[samse_core] Error copying to \"rbwt_sa_de\".\n");

    // Bind texture.
    cudaBindTexture(0,rbwt_sa_tex,*rbwt_sa_de,sizeof(bwtint_t)*rbwt->n_sa);
    report_cuda_error_GPU("[samse_core] Error binding texture to \"rbwt_sa_tex\".\n");

	cudaMemGetInfo(&mem_available, &total_mem);
#if DEBUG_LEVEL > 0
	fprintf(stderr,"[samse_debug] sa/rsa loaded mem left: %d MB\n", (int)(mem_available>>20));
#endif
    return 1;
}


void prepare_bwa_cal_pac_pos_cuda2(int n_seqs_max)
{
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate memory and copy reads in "seqs" to "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho".
	///////////////////////////////////////////////////////////////////////////////////////////////////
	seqs_maxdiff_mapQ_ho = (bwa_maxdiff_mapQ_t *) malloc(sizeof(bwa_maxdiff_mapQ_t)*n_seqs_max);
	if (seqs_maxdiff_mapQ_ho == NULL) report_cuda_error_CPU("[samse_core] Error reserving memory for \"seqs_maxdiff_mapq_ho\".\n");
	seqs_sa_ho = (bwtint_t *) malloc(sizeof(bwtint_t)*n_seqs_max);
	if (seqs_sa_ho == NULL) report_cuda_error_CPU("[samse_core] Error reserving memory for \"seqs_sa_ho\".\n");

    ///////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input data in "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho" to device, and bind texture of
	// "seqs_sa_de".
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Reserve memory.
	cudaMalloc(&seqs_maxdiff_mapQ_de,sizeof(bwa_maxdiff_mapQ_t)*n_seqs_max);
	report_cuda_error_GPU("Error reserving memory for \"seqs_maxdiff_mapQ_de\".\n");

	// Reserve memory.
	cudaMalloc(&seqs_sa_de,sizeof(bwtint_t)*n_seqs_max);
	report_cuda_error_GPU("Error reserving memory for \"seqs_sa_de\".\n");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Reserve memory for output data variables in "seqs_mapQ_ho" and "seqs_pos_ho" to host.
	///////////////////////////////////////////////////////////////////////////////////////////////////
	//cudaGetLastError();
	// Reserve memory for return data "mapQ_ho" and "pos_ho" on the host.
	seqs_mapQ_ho = (uint8_t *) malloc(sizeof(uint8_t)*n_seqs_max);
	if (seqs_mapQ_ho == NULL) report_cuda_error_CPU("[samse_core] Error reserving memory for \"seqs_mapQ_ho\".\n");
	seqs_pos_ho = (bwtint_t *) malloc(sizeof(bwtint_t)*n_seqs_max);
	if (seqs_pos_ho == NULL) report_cuda_error_CPU("[samse_core] Error reserving memory for \"seqs_pos_ho\".\n");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Reserve memory for output data variables "seqs_mapQ_de" and "seqs_pos_de" on device.
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMalloc(&seqs_mapQ_de,sizeof(uint8_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core] Error reserving memory for \"seqs_mapQ_de\".\n");
	cudaMalloc(&seqs_pos_de,sizeof(bwtint_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core] Error reserving memory for \"seqs_pos_de\".\n");

	size_t mem_available, total_mem;

	cudaMemGetInfo(&mem_available, &total_mem);

#if DEBUG_LEVEL > 0
	fprintf(stderr,"[samse_debug] sequence loaded loaded mem left: %d MB\n", (int)(mem_available>>20));
#endif

}

void free_bwa_cal_pac_pos_cuda1(
    unsigned int *global_bwt,
    unsigned int *global_rbwt,
    bwtint_t *bwt_sa_de,
    bwtint_t *rbwt_sa_de,
    int *g_log_n_de)
{

    ////////////////////////////////////////////////////////////
    // Clean up data.
    ////////////////////////////////////////////////////////////
    // Erase BWT on GPU device.
    free_bwts_from_cuda_memory(global_bwt,global_rbwt);

    // Delete memory used.
    cudaFree(bwt_sa_de);
    cudaFree(rbwt_sa_de);
    cudaFree(g_log_n_de);

    // Unbind texture to reads.
    //cudaUnbindTexture(sa_tex);

    // Unbind texture to "g_log_n_tex".
    cudaUnbindTexture(g_log_n_tex);

    // Unbind textures to BWT and RBWT.
    cudaUnbindTexture(bwt_sa_tex);
    cudaUnbindTexture(rbwt_sa_tex);

}

void free_bwa_cal_pac_pos_cuda2()
{
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Clean up data.
	///////////////////////////////////////////////////////////////////////////////////////////////////
	free(seqs_maxdiff_mapQ_ho);
	cudaFree(seqs_maxdiff_mapQ_de);
	free(seqs_sa_ho);
	cudaFree(seqs_sa_de);
	free(seqs_pos_ho);
	cudaFree(seqs_pos_de);
	free(seqs_mapQ_ho);
	cudaFree(seqs_mapQ_de);
}



// This function is meant to be a GPU implementation of bwa_cal_pac_pos(). Currently,
// only the forward strand is being tested for bwt_sa(). After that, test the reverse
// strand. Lastly, make GPU implementations of bwa_cal_maxdiff() and bwa_approx_mapQ().
void launch_bwa_cal_pac_pos_cuda(
	const char *prefix,
	int n_seqs,
	bwa_seq_t *seqs,
	int max_mm,
	float fnr,
	int device)
{

	//fprintf(stderr, "bwt->n_sa: %u %i\n",bwt->n_sa, int(sizeof(bwt->n_sa)));
	//fprintf(stderr, "bwt->sa_intv: %u %i\n",bwt->sa_intv, int(sizeof(bwt->sa_intv)));
	//fprintf(stderr, "rbwt->sa_intv: %u %i\n",rbwt->sa_intv, int(sizeof(rbwt->sa_intv)));

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Declare and initiate variables.
	///////////////////////////////////////////////////////////////////////////////////////////////////

	cudaDeviceProp prop;
	int n_block;
	int n_seq_per_block;
	int block_mod;

	// Obtain information on CUDA devices.
	cudaGetDeviceProperties(&prop, device);


	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate memory and copy reads in "seqs" to "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho".
	///////////////////////////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < n_seqs; i++)
	{
		seqs_maxdiff_mapQ_ho[i].len = seqs[i].len;
		//seqs_maxdiff_mapQ_ho[i].strand_type = ((seqs[i].strand<<2) | seqs[i].type);
		seqs_maxdiff_mapQ_ho[i].strand = seqs[i].strand;
		seqs_maxdiff_mapQ_ho[i].type = seqs[i].type;
		seqs_maxdiff_mapQ_ho[i].n_mm = seqs[i].n_mm;
		seqs_maxdiff_mapQ_ho[i].c1 = seqs[i].c1;
		seqs_maxdiff_mapQ_ho[i].c2 = seqs[i].c2;
		seqs_sa_ho[i] = seqs[i].sa;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input data in "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho" to device, and bind texture of
	// "seqs_sa_de".
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy data from host to device.
	cudaMemcpy(seqs_maxdiff_mapQ_de,seqs_maxdiff_mapQ_ho,sizeof(bwa_maxdiff_mapQ_t)*n_seqs,cudaMemcpyHostToDevice);
	report_cuda_error_GPU("Error copying to \"seqs_maxdiff_mapQ_de\".\n");

	// Copy data from host to device.
	cudaMemcpy(seqs_sa_de,seqs_sa_ho,sizeof(bwtint_t)*n_seqs,cudaMemcpyHostToDevice);
	report_cuda_error_GPU("Error copying to \"seqs_sa_de\".\n");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Process bwa_cal_pac_pos_cuda()
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// No. of blocks.
	n_block = 2048;
	// No of sequences per block.
	n_seq_per_block = n_seqs / n_block;
	// Extra sequences for the last block.
	block_mod = n_seqs - n_seq_per_block * n_block;

	//fprintf(stderr,"N_MP %i n_block %i n_seq_per_block %i block_mod %i\n", N_MP, n_block, n_seq_per_block, block_mod);
	//fprintf(stderr,"n_seqs %i\n", n_seqs);


	// Set block and grid sizes.
	dim3 dimBlock(BLOCK_SIZE2);
	dim3 dimGrid(n_block);

	// Execute bwt_sa function.
	cuda_bwa_cal_pac_pos_parallel2 <<<dimGrid, dimBlock>>>(
		seqs_mapQ_de,
		seqs_pos_de,
		seqs_maxdiff_mapQ_de,
		seqs_sa_de,
		n_seqs,
		n_block,
		n_seq_per_block,
		block_mod,
		max_mm,
		fnr);

	report_cuda_error_GPU("[samse_core] Error running \"cuda_bwa_cal_pac_pos()\".\n");
	cudaThreadSynchronize();
	report_cuda_error_GPU("[samse_core] Error synchronizing after \"cuda_bwa_cal_pac_pos()\".\n");


	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy data of output data variables in "seqs_mapQ_de" and "seqs_pos_de" to host.
	///////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaGetLastError();
	// Return data to host.
	cudaMemcpy(seqs_mapQ_ho, seqs_mapQ_de, sizeof(uint8_t)*n_seqs, cudaMemcpyDeviceToHost);
	report_cuda_error_GPU("[samse_core] Error copying to \"seqs_mapQ_ho\".\n");
	cudaMemcpy(seqs_pos_ho, seqs_pos_de, sizeof(bwtint_t)*n_seqs, cudaMemcpyDeviceToHost);
	report_cuda_error_GPU("[samse_core] Error copying to \"seqs_pos_ho\".\n");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Save output data variables to "seqs".
	///////////////////////////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < n_seqs; i++)
	{
	 	seqs[i].mapQ = seqs_mapQ_ho[i];
		seqs[i].seQ = seqs_mapQ_ho[i];
		seqs[i].pos = seqs_pos_ho[i];
	}
}

// This function does not work because the pointer b->bwt is not set.
__device__ uint32_t _bwt_bwt(const bwt_t *b, bwtint_t k)
{
	return ((b)->bwt[(k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL)/16]);
	//return ((b)->bwt[(k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL >> 4)]);
}

__device__ uint32_t _bwt_bwt2(bwtint_t k)
{
	//int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL) / 16;
	int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL >> 4);
	uint4 four_integers = tex1Dfetch(bwt_occ_array,pos>>2);
	uint32_t one_integer;

	switch (pos & 0x3)
	{
		case 0: one_integer = four_integers.x; break;
		case 1: one_integer = four_integers.y; break;
		case 2: one_integer = four_integers.z; break;
		case 3: one_integer = four_integers.w; break;
	}

	return one_integer;
}



__device__ uint32_t _bwt_bwt3(bwtint_t k, texture<uint4, 1, cudaReadModeElementType> *b)
{
	//int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL) / 16;
	int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL >> 4);
	uint4 four_integers = tex1Dfetch(*b,pos>>2);
	uint32_t one_integer;

	switch (pos & 0x3)
	{
		case 0: one_integer = four_integers.x; break;
		case 1: one_integer = four_integers.y; break;
		case 2: one_integer = four_integers.z; break;
		case 3: one_integer = four_integers.w; break;
	}

	return one_integer;
}



__device__ uint32_t _rbwt_bwt2(bwtint_t k)
{
    //int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL) / 16;
    int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL >> 4);
    uint4 four_integers = tex1Dfetch(rbwt_occ_array,pos>>2);
    uint32_t one_integer;

    switch (pos & 0x3)
    {
        case 0: one_integer = four_integers.x; break;
        case 1: one_integer = four_integers.y; break;
        case 2: one_integer = four_integers.z; break;
        case 3: one_integer = four_integers.w; break;
    }

    return one_integer;
}



// This function does not work because the pointer b->bwt is not set.
__device__ ubyte_t _bwt_B0(const bwt_t *b, bwtint_t k)
{
	uint32_t tmp = _bwt_bwt(b,k)>>((~(k)&0xf)<<1)&3;
	ubyte_t c = ubyte_t(tmp);
	return c;
}



__device__ ubyte_t _bwt_B02(bwtint_t k)
{
	uint32_t tmp = _bwt_bwt2(k)>>((~(k)&0xf)<<1)&3;
	ubyte_t c = ubyte_t(tmp);
	return c;
}



__device__ ubyte_t _rbwt_B02(bwtint_t k)
{
    uint32_t tmp = _rbwt_bwt2(k)>>((~(k)&0xf)<<1)&3;
    ubyte_t c = ubyte_t(tmp);
    return c;
}



__device__ ubyte_t _bwt_B03(bwtint_t k, texture<uint4, 1, cudaReadModeElementType> *b)
{
	//uint32_t tmp = _bwt_bwt3(k,b)>>((~(k)&0xf)<<1)&3;
	//ubyte_t c = ubyte_t(tmp);
    //return c;
    return ubyte_t(_bwt_bwt3(k,b)>>((~(k)&0xf)<<1)&3);
}



__device__ uint32_t* _bwt_occ_intv(const bwt_t *b, bwtint_t k)
{
	return ((b)->bwt + (k)/OCC_INTERVAL*12);
}



__device__
int cuda_bwa_cal_maxdiff(int l, double err, double thres)
{
    double elambda = exp(-l * err);
    double sum, y = 1.0;
    int k, x = 1;
    for (k = 1, sum = elambda; k < 1000; ++k) {
        y *= l * err;
        x *= k;
        sum += elambda * y / x;
        if (1.0 - sum < thres) return k;
    }
    return 2;
}



__device__
int cuda_bwa_approx_mapQ(const bwa_maxdiff_mapQ_t *p, int mm)
{
    int n, g_log;
    if (p->c1 == 0) return 23;
    if (p->c1 > 1) return 0;
    if (p->n_mm == mm) return 25;
    if (p->c2 == 0) return 37;
    n = (p->c2 >= 255)? 255 : p->c2;
    g_log = tex1Dfetch(g_log_n_tex,n);

    return (23 < g_log)? 0 : 23 - g_log;
}



__device__
void update_indices(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty)
{
    (*n_sa_processed)++;
    (*n_sa_remaining)--;
    (*n_sa_in_buf)--;
    (*n_sa_buf_empty)++;
}



__device__
void update_indices_in_parallel(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty)
{
    atomicAdd(*&n_sa_processed,1);
    atomicSub(*&n_sa_remaining,1);
    atomicSub(*&n_sa_in_buf,1);
    atomicAdd(*&n_sa_buf_empty,1);
}



__device__
void fetch_read_new_in_parallel(
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf,
    int16_t *sa_origin,
    const bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de,
    const int offset,
    int *n_sa_in_buf,
    int *n_sa_buf_empty,
    int *n_sa_processed,
    int *n_sa_remaining,
    int *sa_next_no,
    const int n_sa_total,
    const char strand)
{
    while (*sa_next_no < n_sa_total)
    {
        int read_no_new = atomicAdd(*&sa_next_no,1);

        if (read_no_new < n_sa_total)
        {
            // Get new read from global memory.
            *maxdiff_mapQ_buf = seqs_maxdiff_mapQ_de[offset+read_no_new];
            //sa_buf_arr[tid] = seqs_sa_de[offset+read_no_new];
            // Check whether read can be used.
            if ((*maxdiff_mapQ_buf).strand == strand && ((*maxdiff_mapQ_buf).type == BWA_TYPE_UNIQUE ||
                (*maxdiff_mapQ_buf).type == BWA_TYPE_REPEAT))
            {
                *sa_origin = read_no_new;
                //sa_return[tid] = 0;
                atomicAdd(*&n_sa_in_buf,1);
                atomicSub(*&n_sa_buf_empty,1);
                break;
            }
            else
            {
                atomicAdd(*&n_sa_processed,1);
                atomicSub(*&n_sa_remaining,1);
                // Show that read is not being used.
            }
        }
    }
}



__device__
void sort_reads(
    bwtint_t *sa_buf_arr,
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf_arr,
    int16_t *sa_origin,
    bwtint_t *sa_return,
    const int *n_sa_in_buf,
    int *n_sa_in_buf_prev)
{
    int sa_empty_no = *n_sa_in_buf_prev;
    *n_sa_in_buf_prev = *n_sa_in_buf;

    for (int j = 0; j < sa_empty_no; j++)
    {
        if (sa_origin[j] == -1)
        {
            for (int k = sa_empty_no-1; k > j; k--)
            {
                sa_empty_no--;
                if (sa_origin[k] != -1)
                {
                    sa_buf_arr[j] = sa_buf_arr[k];
                    maxdiff_mapQ_buf_arr[j] = maxdiff_mapQ_buf_arr[k];
                    sa_origin[j] = sa_origin[k];
                    sa_return[j] = sa_return[k];
                    sa_origin[k] = -1;
                    break;
                }
            }
        }
    }
}

// This function can process a maximum of 2**15 reads per block.
// bwt_sa() with texture reads (alignment 1).
// BWT and RBWT are separated by order (run in succession).
__global__
void cuda_bwa_cal_pac_pos_parallel2(
    uint8_t *seqs_mapQ_de,
    bwtint_t *seqs_pos_de,
    const bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de,
    const bwtint_t *seqs_sa_de,
    int n_seqs,
    int n_block,
    int n_seq_per_block,
    int block_mod,
    int max_mm,
    float fnr)
{
    // Declare and initialize variables.
    // Thread ID and offset.
    const int tid = threadIdx.x;
    const int offset = blockIdx.x < block_mod ? (n_seq_per_block+1)*blockIdx.x : (n_seq_per_block+1)*block_mod + n_seq_per_block*(blockIdx.x-block_mod);
    const int n_sa_total = n_seq_per_block + (blockIdx.x < block_mod ? 1 : 0);

    int bwt_sa_intv = bwt_cuda.sa_intv;
    int rbwt_sa_intv = rbwt_cuda.sa_intv;

    __shared__ int n_sa_processed;
    __shared__ int n_sa_remaining;
    __shared__ int n_sa_in_buf;
    __shared__ int n_sa_in_buf_prev;
    __shared__ int n_sa_buf_empty;
    __shared__ int sa_next_no;

    __shared__ bwtint_t sa_buf_arr[BLOCK_SIZE2];    // Array of "sa".
    __shared__ bwa_maxdiff_mapQ_t maxdiff_mapQ_buf_arr[BLOCK_SIZE2];    // Array of "maxdiff" elements.
    __shared__ int16_t sa_origin[BLOCK_SIZE2];  // Index of reads.
    __shared__ bwtint_t sa_return[BLOCK_SIZE2]; // Return value.

    // "n_sa_total" is the total number of reads of the block, "n_sa_processed" is the number of finished
    // reads: "n_total = n_sa_processed + n_sa_remaining". "n_sa_in_buf" (<= BLOCK_SIZE2) is the number of
    // reads in process in the buffer, and "n_sa_buf_empty" is the number of empty elements in the buffer:
    // BUFFER_SIZE2 = n_sa_in_buf + n_sa_buf_empty". "sa_next_no" (< "n_total") is the number of the read
    // to fetch next from global or texture memory.

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Run BWT.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    n_sa_processed = 0;
    n_sa_remaining = n_sa_total;
    n_sa_in_buf = min(n_sa_total,BLOCK_SIZE2);
    n_sa_in_buf_prev = n_sa_in_buf;
    n_sa_buf_empty = BLOCK_SIZE2 - n_sa_in_buf;
    sa_next_no = n_sa_in_buf;

    __syncthreads();

    // Fill arrays with initial values. (Do this first to reduce latency as reading from global
    // memory is time-consuming).
    if (tid < n_sa_in_buf)
    {
        maxdiff_mapQ_buf_arr[tid] = seqs_maxdiff_mapQ_de[offset+tid];
        sa_buf_arr[tid] = seqs_sa_de[offset+tid];
    }

    // Set the position in the position array and state which threads are not in use (-1).
    sa_origin[tid] = tid < n_sa_in_buf ? tid : -1;

    // Initialize the return values
    sa_return[tid] = 0;

    // Get new reads on the right strand.
    if (tid < n_sa_in_buf &&
        !(maxdiff_mapQ_buf_arr[tid].strand && (maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_UNIQUE ||
        maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_REPEAT)))
    {
        update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);
        sa_origin[tid] = -1;

        fetch_read_new_in_parallel(
            &maxdiff_mapQ_buf_arr[tid],
            &sa_origin[tid],
            seqs_maxdiff_mapQ_de,
            offset,
            &n_sa_in_buf,
            &n_sa_buf_empty,
            &n_sa_processed,
            &n_sa_remaining,
            &sa_next_no,
            n_sa_total,
            1);

        if (sa_origin[tid] != -1)
        {
            sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
                    //tex1Dfetch(sa_tex,offset+sa_origin[tid]);
            sa_return[tid] = 0;
        }
    }

    // Get rid of reads that are on the wrong strand, fetch new ones.
    __syncthreads();

    if (n_sa_in_buf < BLOCK_SIZE2 && tid == 0)
    {
        sort_reads(
            &sa_buf_arr[0],
            &maxdiff_mapQ_buf_arr[0],
            &sa_origin[0],
            &sa_return[0],
            &n_sa_in_buf,
            &n_sa_in_buf_prev);
    }

    __syncthreads();

    // Start bwt_sa() in a loop until all reads have been processed.
    while (true)
    {
        // Return finished reads, fetch new reads if possible. Run in parallel, not sequentially.
        if //(sa_origin[tid] != -1)
           (tid < n_sa_in_buf)
        {
            char continuation = 1;
            if (sa_buf_arr[tid] % bwt_sa_intv == 0) {continuation = 0;}
            else if (sa_buf_arr[tid] == bwt_cuda.primary)
            {
                sa_return[tid]++;
                sa_buf_arr[tid] = 0;
                continuation = 0;
            }

            if (!continuation)
            {
                int max_diff = cuda_bwa_cal_maxdiff(maxdiff_mapQ_buf_arr[tid].len,BWA_AVG_ERR,fnr);
                uint8_t mapQ = cuda_bwa_approx_mapQ(&maxdiff_mapQ_buf_arr[tid],max_diff);

                // Return read that is finished.
                seqs_pos_de[offset+sa_origin[tid]] = sa_return[tid] + tex1Dfetch(bwt_sa_tex,sa_buf_arr[tid]/bwt_sa_intv);
                // Return "mapQ".
                seqs_mapQ_de[offset+sa_origin[tid]] = mapQ;
                sa_origin[tid] = -1;

                // Update indices.
                update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);

                // Get new read.
                fetch_read_new_in_parallel(
                    &maxdiff_mapQ_buf_arr[tid],
                    &sa_origin[tid],
                    seqs_maxdiff_mapQ_de,
                    offset,
                    &n_sa_in_buf,
                    &n_sa_buf_empty,
                    &n_sa_processed,
                    &n_sa_remaining,
                    &sa_next_no,
                    n_sa_total,
                    1);

                if (sa_origin[tid] != -1)
                {
                    sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
                            //tex1Dfetch(sa_tex,offset+sa_origin[tid]);
                    sa_return[tid] = 0;
                }
            }
        }

        __syncthreads();

        if (n_sa_remaining <= 0) break;

        // This section puts reads in the buffer first to allow full warps to be run.
        if (n_sa_in_buf < BLOCK_SIZE2)
        {
            if (tid == 0)
            {
                sort_reads(
                    &sa_buf_arr[0],
                    &maxdiff_mapQ_buf_arr[0],
                    &sa_origin[0],
                    &sa_return[0],
                    &n_sa_in_buf,
                    &n_sa_in_buf_prev);
            }

            __syncthreads();
        }

        sa_return[tid]++;

        if (tid < n_sa_in_buf)
        {
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Start bwt_sa (bwtint_t bwt_sa(const bwt_t *bwt, bwtint_t k))
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////

            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Start #define bwt_invPsi(bwt, k)
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // First conditional expression.
            // Moved to the section above where "else if (sa_arr[k] == bwt_cuda.primary)".

            // Second conditional expression.
            bwtint_t invPsi1 = sa_buf_arr[tid] < bwt_cuda.primary ? sa_buf_arr[tid] : sa_buf_arr[tid]-1;
            ubyte_t invPsi2 = _bwt_B02(invPsi1);
            invPsi1 = bwt_cuda_occ(sa_buf_arr[tid],invPsi2);
            sa_buf_arr[tid] = bwt_cuda.L2[invPsi2]+invPsi1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Run RBWT.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //seqs_pos_de[offset+sa_origin[tid]] = bwt_cuda.seq_len - (maxdiff_mapQ_buf_arr[tid].len +
    //    sa_return[tid] + tex1Dfetch(rbwt_sa_tex,sa_buf_arr[tid]/rbwt_sa_intv));
    __syncthreads();

    n_sa_processed = 0;
    n_sa_remaining = n_sa_total;
    n_sa_in_buf = min(n_sa_total,BLOCK_SIZE2);
    n_sa_in_buf_prev = n_sa_in_buf;
    n_sa_buf_empty = BLOCK_SIZE2 - n_sa_in_buf;
    sa_next_no = n_sa_in_buf;

    __syncthreads();

    // Fill arrays with initial values. (Do this first to reduce latency as reading from global
    // memory is time-consuming).
    if (tid < n_sa_in_buf)
    {
        maxdiff_mapQ_buf_arr[tid] = seqs_maxdiff_mapQ_de[offset+tid];
        sa_buf_arr[tid] = seqs_sa_de[offset+tid];
    }

    // Set the position in the return array.
    sa_origin[tid] = tid < n_sa_in_buf ? tid : -1;

    // Initialize the return values.
    sa_return[tid] = 0;

    // Get new reads on the right strand.
    if (tid < n_sa_in_buf &&
        !(!maxdiff_mapQ_buf_arr[tid].strand && (maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_UNIQUE ||
        maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_REPEAT)))
    {
        update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);
        sa_origin[tid] = -1;

        fetch_read_new_in_parallel(
            &maxdiff_mapQ_buf_arr[tid],
            &sa_origin[tid],
            seqs_maxdiff_mapQ_de,
            offset,
            &n_sa_in_buf,
            &n_sa_buf_empty,
            &n_sa_processed,
            &n_sa_remaining,
            &sa_next_no,
            n_sa_total,
            0);

        if (sa_origin[tid] != -1)
        {
            sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
            sa_return[tid] = 0;
        }
    }

    // Sort reads.
    __syncthreads();

    if (tid == 0)
    {
        sort_reads(
            &sa_buf_arr[0],
            &maxdiff_mapQ_buf_arr[0],
            &sa_origin[0],
            &sa_return[0],
            &n_sa_in_buf,
            &n_sa_in_buf_prev);
    }

    __syncthreads();

    // Start bwt_sa() in a loop until all reads have been processed.
    while (true)
    {
        // Return finished reads, fetch new reads if possible. Run in parallel, not sequentially.
        if //(sa_origin[tid] != -1)
           (tid < n_sa_in_buf)
        {
            char continuation = 1;
            if (sa_buf_arr[tid] % rbwt_sa_intv == 0) {continuation = 0;}
            else if (sa_buf_arr[tid] == rbwt_cuda.primary)
            {
                sa_return[tid]++;
                sa_buf_arr[tid] = 0;
                continuation = 0;
            }

            if (!continuation)
            {
                int max_diff = cuda_bwa_cal_maxdiff(maxdiff_mapQ_buf_arr[tid].len,BWA_AVG_ERR,fnr);
                uint8_t mapQ = cuda_bwa_approx_mapQ(&maxdiff_mapQ_buf_arr[tid],max_diff);

                // Return read that is finished.
                //seqs_pos_de[offset+sa_origin[tid]] = sa_return[tid] + tex1Dfetch(bwt_sa_tex,sa_buf_arr[tid]/bwt_sa_intv);
                seqs_pos_de[offset+sa_origin[tid]] = bwt_cuda.seq_len - (maxdiff_mapQ_buf_arr[tid].len +
                    sa_return[tid] + tex1Dfetch(rbwt_sa_tex,sa_buf_arr[tid]/rbwt_sa_intv));
                // Return "mapQ".
                seqs_mapQ_de[offset+sa_origin[tid]] = mapQ;
                sa_origin[tid] = -1;

                // Update indices.
                update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);

                // Get new read.
                fetch_read_new_in_parallel(
                    &maxdiff_mapQ_buf_arr[tid],
                    &sa_origin[tid],
                    seqs_maxdiff_mapQ_de,
                    offset,
                    &n_sa_in_buf,
                    &n_sa_buf_empty,
                    &n_sa_processed,
                    &n_sa_remaining,
                    &sa_next_no,
                    n_sa_total,
                    0);

                if (sa_origin[tid] != -1)
                {
                    sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
                    sa_return[tid] = 0;
                }
            }
        }

        __syncthreads();

        if (n_sa_remaining <= 0) break;

        // This section puts reads in the buffer first to allow full warps to be run.
        if (n_sa_in_buf < BLOCK_SIZE2)
        {
            if (tid == 0)
            {
                sort_reads(
                    &sa_buf_arr[0],
                    &maxdiff_mapQ_buf_arr[0],
                    &sa_origin[0],
                    &sa_return[0],
                    &n_sa_in_buf,
                    &n_sa_in_buf_prev);
            }

            __syncthreads();
        }

        sa_return[tid]++;

        if (tid < n_sa_in_buf)
        {
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Start bwt_sa (bwtint_t bwt_sa(const bwt_t *bwt, bwtint_t k))
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////

            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Start #define bwt_invPsi(bwt, k)
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            // First conditional expression.
            // Moved to the section above where "else if (sa_arr[k] == bwt_cuda.primary)".

            // Second conditional expression.
            bwtint_t invPsi1 = sa_buf_arr[tid] < rbwt_cuda.primary ? sa_buf_arr[tid] : sa_buf_arr[tid]-1;
            ubyte_t invPsi2 = _rbwt_B02(invPsi1);
            invPsi1 = rbwt_cuda_occ(sa_buf_arr[tid],invPsi2);
            sa_buf_arr[tid] = rbwt_cuda.L2[invPsi2]+invPsi1;
        }
    }
}

#endif
///////////////////////////////////////////////////////////////
// End CUDA samse_core
///////////////////////////////////////////////////////////////


