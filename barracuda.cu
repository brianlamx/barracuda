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

/* (0.7.0) beta: 
  27 Feb 2015 WBL remove bulk loopcount==0 debug
  26 Feb 2015 WBL swap back from bwt_cuda_occ4.cuh to stub bwt_cuda_occ4()
  25 Feb 2015 WBL skip r1.32, r1.33(sequence_shift81 perhaps do later?),
r1.34(sequence_global), r1.35-43, apply r1.43(no nulls in properties.name),
skip r1.45-46, apply r1.47-50(pad bwt to 16 int), 
skip r1.51-55(many_blocks) r1.56(threads_per_sequence) skip r1.57
skip r1.58-59(TotalCores, include helper_cuda.h)
apply r1.60-63(d_mycache4, include read_mycache.cuh) skip r1.64
skip r1.65-69(cache_threads, kl_split, kl_par)
  29 Dec 2014 WBL Avoid binary files by removing nulls in properties.name
  21 Feb 2015 WBL reduce volume of debug output
  19 Feb 2015 WBL still no progress... have wound back to r1.89
  try adding huge debug to each kernel launch
  12 Feb 2015 WBL Add displaying timing info for cuda_inexact_match_caller
  fix r1.85 performance problem with include bwt_cuda_occ4.cuh
Split history barracuda_src.cu,v barracuda.cu,v
  11 Feb 2015 WBL Add same_length to copy_sequences_to_cuda_memory
  replaces r1.30 14 Dec 2014 WBL for direct_index force all sequences to be same length
  Add stub bwt_cuda_occ4 add direct_sequence
  move remaining cuda device code for cuda_split_inexact_match_caller etc to cuda2.cuh
  10 Feb 2015 WBL Split history barracuda_src.cu,v barracuda.cu,v
Re-apply r1.25 free kl_host/kl_device, size_t, remove bwtkl_t (now in barracuda.h),
improve "[aln_debug] bwt loaded %lu bytes, <assert.h> include cuda.cuh
  25 Nov 2014 WBL Re-enable cuda_find_exact_matches changes. Note where sequence matches exactly once no longer report other potential matches
  21 Nov 2014 WBL disable cuda_find_exact_matches changes and add <<<>>> logging comments
                  Add header to text .sai file
  19 Nov 2014 WBL merge text and binary output, ie add stdout_aln_head stdout_barracuda_aln1
                  Explicitly clear unused parts of alignment records in binary .sai output file
  13 Nov 2014 WBL try re-enabling cuda_find_exact_matches
  13 Nov 2014 WBL ensure check status of all host cuda calls
  Ensure all kernels followed by cudaDeviceSynchronize so they can report asynchronous errors
*/

#define PACKAGE_VERSION "0.7.0 beta $Revision: 1.101 $"
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include "bwtaln.h"
#include "bwtgap.h"
#include "utils.h"
#include "barracuda.h"

#define d_mycache4 const uint4* mycache0
#define d_mycache8 const uint2* mycache0
#define d_mycache16 const uint32_t* mycache0

#define max_mycache 1
#include "read_mycache.cuh"
#undef max_mycache

#undef d_mycache4
#undef d_mycache8
#undef d_mycache16

#define d_mycache4 const uint4* mycache0
#define d_mycache8 const uint2* mycache0
#define d_mycache16 const uint32_t* mycache0,const uint32_t* mycache1

#define max_mycache 2
#include "read_mycache.cuh"
#undef max_mycache

#undef d_mycache4
#undef d_mycache8
#undef d_mycache16

#define d_mycache4 const uint4* mycache0
#define d_mycache8 const uint2* mycache0,const uint2* mycache1
#define d_mycache16 const uint32_t* mycache0,const uint32_t* mycache1,const uint32_t* mycache2,const uint32_t* mycache3

#define max_mycache 4
#include "read_mycache.cuh"
#undef max_mycache

#undef d_mycache4
#undef d_mycache8
#undef d_mycache16

#define d_mycache4 const uint4* mycache0,const uint4* mycache1
#define d_mycache8 const uint2* mycache0,const uint2* mycache1,const uint2* mycache2,const uint2* mycache3
#define d_mycache16 const uint32_t* mycache0,const uint32_t* mycache1,const uint32_t* mycache2,const uint32_t* mycache3,const uint32_t* mycache4,const uint32_t* mycache5,const uint32_t* mycache6,const uint32_t* mycache7

#define max_mycache 8
#include "read_mycache.cuh"
#undef max_mycache

#undef d_mycache4
#undef d_mycache8
#undef d_mycache16

#define d_mycache4 const uint4* mycache0,const uint4* mycache1,const uint4* mycache2,const uint4* mycache3
#define d_mycache8 const uint2* mycache0,const uint2* mycache1,const uint2* mycache2,const uint2* mycache3,const uint2* mycache4,const uint2* mycache5,const uint2* mycache6,const uint2* mycache7
#define d_mycache16 const uint32_t* mycache0,const uint32_t* mycache1,const uint32_t* mycache2,const uint32_t* mycache3,const uint32_t* mycache4,const uint32_t* mycache5,const uint32_t* mycache6,const uint32_t* mycache7,const uint32_t* mycache8,const uint32_t* mycache9,const uint32_t* mycache10,const uint32_t* mycache11,const uint32_t* mycache12,const uint32_t* mycache13,const uint32_t* mycache14,const uint32_t* mycache15

#define max_mycache 16
#include "read_mycache.cuh"
#undef max_mycache

#undef d_mycache4
#undef d_mycache8
#undef d_mycache16

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


//macro was beyound comprehension
inline void write_to_half_byte_array(unsigned char * array, const int index, const int data) {
  const int wordindex = index>>3;
  const int byteindex = wordindex*4 + ((index>>1) & 0x3);
  if((index)&0x1) array[byteindex] = (array[byteindex]&0xF0) | (data &0x0F);
  else            array[byteindex] = (array[byteindex]&0x0F) | (data<<4);
}

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


size_t copy_bwts_to_cuda_memory(const char * prefix, uint32_t ** bwt, uint32_t mem_available, bwtint_t* seq_len)
// bwt occurrence array to global and bind to texture, bwt structure to constant memory
// this function only load part of the bwt for alignment only.  SA is not loaded.
// mem available in MiB (not bytes)
{
	bwt_t * bwt_src;
	size_t size_read = 0;

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
			fprintf(stderr,"[aln_debug] bwt loaded %lu bytes to CPU \n", size_read);
	#endif
	size_read = bwt_src->bwt_size*sizeof(uint32_t);
	mem_available = mem_available - uint32_t (size_read>>20); // mem available in MiB (not bytes)
	*seq_len = bwt_src->seq_len;

	if(mem_available > 0)
	{
		//Allocate memory for bwt
		const int bwt_size = (bwt_src->bwt_size + 15) & (~0xf); //ensure multiple of 16 ints
		//printf("bwt_size %d padded to %d uint32_t for FIXED_MAX_global_bwt\n",bwt_src->bwt_size,bwt_size);
		cudaMalloc((void**)bwt, bwt_size*sizeof(uint32_t));
		report_cuda_error_GPU("[aln_core] Error allocating memory for \"bwt_occurrence array\".\n");
		cudaMemset(&((*bwt)[bwt_size-16]), 0, 16*sizeof(uint32_t));
		report_cuda_error_GPU("[aln_core] Error clearing padding in \"bwt_occurrence array\".\n");
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
		int *same_length,
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
error need to set same_length...
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
	barracuda_query_array_t *seqs = barracuda_read_seqs(bs,  buffer, &n_seqs, 0, 0, &accumulated_length, max_length, same_length);
	//TODO: insert  sort here!!!!
	//TODO: Arran: put the clumping code here.
	barracuda_write_to_half_byte_array(seqs, main_sequences, main_sequences_index, n_seqs);
	number_of_sequences = (unsigned int) n_seqs;

#endif

    //copy main_sequences_width from host to device
    cudaUnbindTexture(sequences_index_array);
    report_cuda_error_GPU("[aln_core] Error freeing texture \"sequences_index_array\".");
    cudaMemcpy(global_sequences_index, main_sequences_index, (number_of_sequences)*sizeof(uint2), cudaMemcpyHostToDevice);
    report_cuda_error_GPU("[aln_core] Error copying to \"global_sequences_index\" on GPU");
    cudaBindTexture(0, sequences_index_array, global_sequences_index, (number_of_sequences)*sizeof(uint2));
    report_cuda_error_GPU("[aln_core] Error binding texture \"sequences_index_array\".\n");

    //copy main_sequences from host to device, sequences array length should be accumulated_length/2
    cudaUnbindTexture(sequences_array);
    report_cuda_error_GPU("[aln_core] Error freeing texture \"sequences_array\".");
    cudaMemcpy(global_sequences, main_sequences, (1ul<<(buffer))*sizeof(unsigned char), cudaMemcpyHostToDevice);
    report_cuda_error_GPU("[aln_core] Error copying to \"global_sequences\" on GPU");
    cudaBindTexture(0, sequences_array, global_sequences, (1ul<<(buffer))*sizeof(unsigned char));
    report_cuda_error_GPU("[aln_core] Error binding texture to \"sequences_array\".\n");

    if ( read_size ) *read_size = accumulated_length;
    free (seqs);
    return number_of_sequences;
}

//CUDA DEVICE CODE STARTING FROM THIS LINE
/////////////////////////////////////////////////////////////////////////////

/*WBL 12 Feb 2015 performance much worse for barracuda r1.85
 * for timebeing try using full old version of bwt_cuda_occ4*/
#include "bwt_cuda_occ4.cuh"

#include "cuda.cuh"


/*WBL 11 feb 2015 dummy stub fix cuda_dfs_match() properly later**
__device__ ulong4 bwt_cuda_occ4(uint32_t *global_bwt, bwtint_t k) {
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
#endif //scache_global_bwt
  ulong4 n;
  n.x = bwt_cuda_occ(global_bwt, k, 0, 0, &last,l_mycache0);
  n.y = bwt_cuda_occ(global_bwt, k, 1, 0, &last,l_mycache0);
  n.z = bwt_cuda_occ(global_bwt, k, 2, 0, &last,l_mycache0);
  n.w = bwt_cuda_occ(global_bwt, k, 3, 0, &last,l_mycache0);
  return n;
}
*/

//configuration options for GP to tune
#undef direct_sequence
#include "bwt_cuda_match_exact.cuh"


#include "cuda2.cuh"
//END CUDA DEVICE CODE

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

void stdout_aln_head(const int id, const int* no_of_alignments) {
#if STDOUT_STRING_RESULT == 1
  //output even if no_of_alignments <=0 
  printf("Sequence %d", id);
  printf(", no of alignments: %d\n", *no_of_alignments);
#else
  err_fwrite(no_of_alignments, 4, 1, stdout);
#endif
}

#if STDOUT_STRING_RESULT == 1
//ignore nmemb < MAX_NO_OF_ALIGNMENTS limit
#define stdout_aln1(type,opt_best_cnt)	\
  for(size_t i=0; i<nmemb; i++) {\
    printf("  Aligned read, ");\
    printf("n_mm: %d, ",   aln[i].n_mm);\
    printf("n_gape: %d, ", aln[i].n_gape);\
    printf("n_gapo: %d, ", aln[i].n_gapo);\
    printf("k: %llu, ",    aln[i].k);\
    printf("l: %llu, ",    aln[i].l);\
    printf("score: %d",    aln[i].score);\
    opt_best_cnt;\
    printf("\n");\
  }
#else
#define stdout_aln1(type,opt_best_cnt) \
  err_fwrite(aln, sizeof(type), nmemb, stdout);
#endif

void stdout_bwt_aln1(      const bwt_aln1_t       *aln, const size_t nmemb) {
  //fprintf(stderr,"stdout_bwt_aln1(*aln, %d) %dbytes\n",nmemb,sizeof(bwt_aln1_t));
  stdout_aln1(bwt_aln1_t,);
}
void stdout_barracuda_aln1(const barracuda_aln1_t *aln, const size_t nmemb) {
  //fprintf(stderr,"stdout_barracuda_aln1(*aln, %d) %dbytes\n",nmemb,sizeof(barracuda_aln1_t));
  stdout_aln1(barracuda_aln1_t,printf("best_cnt: %d", aln[i].best_cnt));
}
#undef stdout_aln1

/*WBL for debug
void print_global_alns(const int no_to_process, const int max_no_partial_hits, const barracuda_aln1_t * global_alns_device) {
  const size_t nbytes = max_no_partial_hits*no_to_process*sizeof(barracuda_aln1_t);
  barracuda_aln1_t * global_alns_host = (barracuda_aln1_t*)malloc(nbytes);
  assert(global_alns_host);
  cudaMemcpy(global_alns_host, global_alns_device, nbytes, cudaMemcpyDeviceToHost);
  report_cuda_error_GPU("[aln_core] Error reading \"global_alns_host\" from GPU for print.");

  for(int i=0;i<no_to_process;i++) {
    printf("alns %d ",i);
    for(int j=0;j<max_no_partial_hits;j++) {
      printf("%d %d %d ",
	     int(global_alns_host[i].n_mm),
	     int(global_alns_host[i].n_gapo),
	     int(global_alns_host[i].n_gape));
      printf("%lu %lu ",global_alns_host[i].k,global_alns_host[i].l);
      printf("%d %d",global_alns_host[i].score,global_alns_host[i].best_cnt);
      if(j<max_no_partial_hits-1) printf(", ");
    }
    printf("\n");
  }

  free(global_alns_host);
}*/
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
		int same_length;
		// Flag: are all sequences the same length
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
		bwtkl_t * kl_device, *kl_host;



	//CPU and GPU memory allocations
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	gettimeofday (&start, NULL);
		//allocate global_sequences memory in device
		cudaMalloc((void**)&global_sequences, (1ul<<(buffer))*sizeof(unsigned char));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_sequences\".");
		main_sequences = (unsigned char *)malloc((1ul<<(buffer))*sizeof(unsigned char));
		//suffixes for clumping
		main_suffixes = (unsigned long long *)malloc((1ul<<(buffer-3))*sizeof(unsigned long long));
		//allocate global_sequences_index memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		cudaMalloc((void**)&global_sequences_index, (1ul<<(buffer-3))*sizeof(uint2));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_sequences_index\".");
		main_sequences_index = (uint2*)malloc((1ul<<(buffer-3))*sizeof(uint2));
		//allocate and copy options (opt) to device constant memory
		cudaMalloc((void**)&options, sizeof(gap_opt_t));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"options\".");
		cudaMemcpy ( options, opt, sizeof(gap_opt_t), cudaMemcpyHostToDevice);
		report_cuda_error_GPU("[aln_core] Error cudaMemcpy to \"options\" on GPU");
		cudaMemcpyToSymbol ( options_cuda, opt, sizeof(gap_opt_t), 0, cudaMemcpyHostToDevice);
		report_cuda_error_GPU("[aln_core] Error in cudaMemcpyToSymbol to \"options_cuda\" on GPU");
		//allocate alignment stores for host and device
		cudaMalloc((void**)&global_alignment_meta_device, (1ul<<(buffer-3))*sizeof(alignment_meta_t));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_alignment_meta_device\".");
		cudaMalloc((void**)&global_alns_device, MAX_NO_PARTIAL_HITS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_alns_device\".");
		cudaMalloc((void**)&global_init_device, (1ul<<(buffer-3))*sizeof(init_info_t));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_init_device\".");
		cudaMalloc((void**)&global_w_b_device, (1ul<<(buffer-3))*sizeof(widths_bids_t));
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_w_b_device\".");
		cudaMalloc((void**)&global_seq_flag_device, (1ul<<(buffer-3))*sizeof(char));	
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"global_seq_flag_device\".");

		cudaMalloc((void**)&kl_device, (1ul<<(buffer-3))*sizeof(bwtkl_t));	
		report_cuda_error_GPU("[core] Error allocating cuda memory for \"kl_device\".");

	//allocate alignment store memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		global_alignment_meta_host = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
		assert(global_alignment_meta_host);//better than segfault later
		global_alns_host = (barracuda_aln1_t*)malloc(MAX_NO_PARTIAL_HITS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		assert(global_alns_host);
		global_alignment_meta_host_final = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
		assert(global_alignment_meta_host_final);
		global_alns_host_final = (barracuda_aln1_t*)malloc(MAX_NO_OF_ALIGNMENTS*(1ul<<(buffer-3))*sizeof(barracuda_aln1_t));
		assert(global_alns_host_final);
		global_init_host = (init_info_t*)malloc((1ul<<(buffer-3))*sizeof(init_info_t));
		assert(global_init_host);
		global_seq_flag_host = (char*)malloc((1ul<<(buffer-3))*sizeof(char));
		assert(global_seq_flag_host);
#if USE_PETR_SPLIT_KERNEL > 0
		global_alignment_meta_host_final = (alignment_meta_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_meta_t));
		assert(global_alignment_meta_host_final);
#endif
		kl_host = (bwtkl_t*)malloc((1ul<<(buffer-3))*sizeof(bwtkl_t));
		assert(kl_host);

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
	report_cuda_error_GPU("[core] Error on \"cudaGetDeviceProperties\".");
	if ((int) selected_properties.major > 1) {
		blocksize = 64;
	} else {
		blocksize = 320;
	}

	while ( ( no_of_sequences = copy_sequences_to_cuda_memory(ks, global_sequences_index, main_sequences_index, global_sequences, main_sequences, &read_size, &max_sequence_length, &same_length, buffer, main_suffixes, SUFFIX_CLUMP_WIDTH) ) > 0 )
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
		//fprintf(stderr, "%d sequences max_sequence_length=%d same_length=%d\n", no_of_sequences, max_sequence_length, same_length);
		fprintf(stderr, "l%d", loopcount);
#if STDOUT_STRING_RESULT == 1
		fprintf(stdout, "loopcount %d\n", loopcount);
#endif

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
		//fprintf(stderr,"cuda_prepare_widths<<<(%d,%d,%d)(%d,%d,%d)>>>(global_bwt, %d, global_w_b_device, global_seq_flag_device)\n",
		//	dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_of_sequences);

		cudaDeviceSynchronize();
		cuda_err = cudaGetLastError();
		if(int(cuda_err))
		{
			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during width/bid preparation! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
			return;
		}
		/*if(loopcount==0){
		  const size_t nbytes = no_of_sequences*sizeof(widths_bids_t);
		  widths_bids_t* w_b = (widths_bids_t*)malloc(nbytes);
		  assert(w_b);
		  cudaMemcpy(w_b, global_w_b_device, nbytes, cudaMemcpyDeviceToHost);
		  report_cuda_error_GPU("[aln_core] Error reading \"global_w_b_device\" from GPU.");
		  for(int i=0;i<no_of_sequences;i++) {
		    printf("w_b %d ",i);
		    for(int j=0;j<max_sequence_length+1;j++) {
		      printf("%u %u",w_b[i].widths[j],int(w_b[i].bids[j]));
		      if(j<=max_sequence_length) printf(" ");
		    }
		    printf("\n");
		  }
		  free(w_b);
		}*/

		if(same_length) { /*new cuda_find_exact_matches assumes all sequences are same length*/
		//WBL re-enabled cuda_find_exact_matches with new KL output
		fprintf(stderr, "cuda_find_exact_matches<<<(%d,%d,%d)(%d,%d,%d)>>>(global_bwt, %d, %d, kl_device)\n",
			dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_of_sequences,max_sequence_length);
		struct timeval start2;
		gettimeofday (&start2, NULL);
		cuda_find_exact_matches<<<dimGrid,dimBlock>>>(global_bwt, no_of_sequences, max_sequence_length, kl_device);
		cudaDeviceSynchronize();
		cuda_err = cudaGetLastError();
		if(int(cuda_err))
		{
			fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during exact match pre-check! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
			return;
		}
		gettimeofday (&end, NULL);
		const double time_used = diff_in_seconds(&end,&start2);
		fprintf(stderr, "[aln_core] find_exact_matches Kernel speed: %g sequences/sec or %g bp/sec %g\n", no_of_sequences/time_used, read_size/time_used, time_used);
		}
		cudaMemcpy(kl_host, kl_device, no_of_sequences*sizeof(bwtkl_t), cudaMemcpyDeviceToHost);
		report_cuda_error_GPU("[aln_core] Error reading \"kl_host\" from GPU.");

		/*if(loopcount==0)
		for(int i=0;i<no_of_sequences;i++) {
		  printf("kl %d %lu %lu\n",i,kl_host[i].k,kl_host[i].l);
		}*/

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Exclude exact unique matches and
		// Cull for too many Ns
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		memset(global_init_host, 0, no_of_sequences*sizeof(init_info_t));
		cudaMemcpy(global_seq_flag_host, global_seq_flag_device, no_of_sequences*sizeof(char), cudaMemcpyDeviceToHost);
		report_cuda_error_GPU("[aln_core] Error reading \"global_seq_flag_host\" from GPU.");
		cudaDeviceSynchronize();
		report_cuda_error_GPU("[aln_core] cuda error");
		unsigned int no_to_process = 0;
		for(int i=0; i<no_of_sequences; i++){
		    //use K,L values to note sequences that have a unique exact match - allows setting of best_score=0 and skiping rest of processing
			if(same_length && /*ie cuda_find_exact_matches has been run*/
			   kl_host[i].k == kl_host[i].l) {
		    //save k and l, clear rest (n_mm etc)
				barracuda_aln1_t * tmp_aln = global_alns_host_final + i*MAX_NO_OF_ALIGNMENTS;
				memset(tmp_aln,0,sizeof(barracuda_aln1_t)); //clear n_mm, n_gapo,n_gape, score, best_cnt
				memcpy(&(tmp_aln->k),&kl_host[i].k,sizeof(bwtkl_t));
				//tmp_aln->n_mm = 100+loopcount; //for debug
		    //make sure sequence is marked so not processed again
				memset(global_alignment_meta_host_final + i, 0, sizeof(alignment_meta_t));		    
				global_alignment_meta_host_final[i].no_of_alignments = 1;
				//best_score = 0;
				global_alignment_meta_host_final[i].sequence_id = i;
				global_alignment_meta_host_final[i].best_cnt = 1;
				//char pos = 0;
				global_alignment_meta_host_final[i].finished = 1;
				//fprintf(stderr, "global_alignment_meta_host_final[%d].sequence_id = %d\n",i,global_alignment_meta_host_final[i].sequence_id);
			} else
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

		fprintf(stderr, "'");
		cudaMemcpy(global_init_device, global_init_host, no_to_process*sizeof(init_info_t), cudaMemcpyHostToDevice);
		report_cuda_error_GPU("[aln_core] Error copying \"global_init_host\" to GPU.");
		//cuda_find_exact_matches writes straight to global_init_device so we can launch the first kernel and then deal with global_seq_flag_device

		{struct timeval start2;
		gettimeofday (&start2, NULL);

		cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_to_process, global_alignment_meta_device, global_alns_device, global_init_device, global_w_b_device, best_score, split_engage, SUFFIX_CLUMP_WIDTH>0);
		fprintf(stderr,"1 cuda_inexact_match_caller<<<(%d,%d,%d)(%d,%d,%d)>>>(,%d,,,,,,,%d)\n",
			dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_to_process, SUFFIX_CLUMP_WIDTH);
		fprintf(stderr, "'");

		//***EXACT MATCH CHECK***
		//store knowledge of an exact match to be copied into init struct during partial hit queueing
		//cudaMemcpy(global_seq_flag_host, global_seq_flag_device, no_of_sequences*sizeof(char), cudaMemcpyDeviceToHost);
		//report_cuda_error_GPU("[aln_core] Error reading \"global_seq_flag_host\" from GPU.");
		//for(int i=0; i<no_of_sequences; i++){
//				if(global_seq_flag_host[i]){
//					global_alignment_meta_host_final[i].has_exact = 1;
//				}
		//}


#if DEBUG_LEVEL > 0
		fprintf(stderr,"\n[aln_debug] kernel started, waiting for data... \n", time_used);
#endif
		// Did we get an error running the code? Abort if yes.
		cudaDeviceSynchronize(); //wait until kernel has had a chance to report error
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
		const double time_used = diff_in_seconds(&end,&start2);
		fprintf(stderr, "[aln_core] 1 inexact Kernel speed: %u %g sequences/sec %g\n", no_to_process, no_to_process/time_used, time_used);
		}
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
			report_cuda_error_GPU("[aln_core] cuda error_2.");
			if(!split_loop_count){
				fprintf(stderr, "'");
			}
			cudaMemcpy(global_alignment_meta_host, global_alignment_meta_device, no_to_process*sizeof(alignment_meta_t), cudaMemcpyDeviceToHost);
			report_cuda_error_GPU("[aln_core] Error reading \"global_alignment_meta_host\" from GPU.");
			int max_no_partial_hits = (!split_loop_count ? MAX_NO_SEEDING_PARTIALS : MAX_NO_REGULAR_PARTIALS);
			cudaMemcpy(global_alns_host, global_alns_device, max_no_partial_hits*no_to_process*sizeof(barracuda_aln1_t), cudaMemcpyDeviceToHost);
			report_cuda_error_GPU("[aln_core] Error reading \"global_alns_host\" from GPU.");
			cudaDeviceSynchronize();
			report_cuda_error_GPU("[aln_core] cuda error_3.");
			//if(loopcount==0) print_global_alns(max_no_partial_hits,no_to_process,global_alns_device);
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
				report_cuda_error_GPU("[aln_core] Error_2 copying \"global_init_host\" to GPU.");

				int gridsize = GRID_UNIT * (1 + int (((no_to_process/blocksize) + ((no_to_process%blocksize)!=0))/GRID_UNIT));
				dim3 dimGrid(gridsize);
				struct timeval start2;
				gettimeofday (&start2, NULL);
				cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_to_process, global_alignment_meta_device, global_alns_device, global_init_device, global_w_b_device, best_score, split_engage, 0);
				fprintf(stderr,"2 cuda_inexact_match_caller<<<(%d,%d,%d)(%d,%d,%d)>>>(,%d,,,,,,,0)\n",
					dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_to_process);
				cudaDeviceSynchronize(); //wait until kernel has had a chance to report error
				cuda_err = cudaGetLastError();
				if(int(cuda_err))
				{
					fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported during split kernel run! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
					return;
				}
				//if(loopcount==0) print_global_alns(no_to_process,(!split_loop_count ? MAX_NO_SEEDING_PARTIALS : MAX_NO_REGULAR_PARTIALS),global_alns_device);
				gettimeofday (&end, NULL);
				const double time_used = diff_in_seconds(&end,&start2);
				fprintf(stderr, "[aln_core] 2 inexact Kernel speed: %u %g sequences/sec %g\n", no_to_process, no_to_process/time_used, time_used);
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
		report_cuda_error_GPU("[aln_core] Error copying \"global_alignment_meta_host\" to GPU.");

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
			assert(0);//should we be here?
			cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);
			fprintf(stderr,"cuda_split_inexact_match_caller<<<(%d,%d,%d)(%d,%d,%d)>>>(%d, %d, global_alignment_meta_device, 0)\n",
				dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_of_sequences,max_sequence_length);
		} else {
			assert(0);//should we be here?
		  //WBL 21 Nov 2014 looks odd cuda_inexact_match_caller arguments do not match
			cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(global_bwt, no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);
			fprintf(stderr,"3 cuda_inexact_match_caller<<<(%d,%d,%d)(%d,%d,%d)>>>(global_bwt, %d, %d, global_alignment_meta_device, 0)\n",
				dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_of_sequences,max_sequence_length);
		}
		fprintf(stderr,"[aln_debug] kernels return \n", time_used);

		// Did we get an error running the code? Abort if yes.
		cudaDeviceSynchronize(); //wait until kernel has had a chance to report error
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
			report_cuda_error_GPU("[aln_core] Error_2 reading \"global_alignment_meta_host\" from GPU.");

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
				report_cuda_error_GPU("[aln_core] Error_2 copying \"global_alignment_meta_host\" to GPU.");

				//run kernel again
				assert(0);//should we be here?
				cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_meta_device, 0);
				fprintf(stderr,"cuda_split_inexact_match_caller<<<(%d,%d,%d)(%d,%d,%d)>>>(%d, %d, global_alignment_meta_device, 0)\n",
					dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,no_of_sequences, max_sequence_length);

				// Did we get an error running the code? Abort if yes.
				cudaDeviceSynchronize(); //wait until kernel has had a chance to report error
				cudaError_t cuda_err = cudaGetLastError();
				if(int(cuda_err))
				  {
				    fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
				    return;
				  }

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
		for (int  i = 0; i < run_no_sequences; ++i)
		{
#if USE_PETR_SPLIT_KERNEL > 0
			alignment_meta_t* tmp = global_alignment_meta_host_final + i;
#else
			alignment_meta_t* tmp = global_alignment_meta_host + i;
#endif
			stdout_aln_head(i,&tmp->no_of_alignments);
			if (tmp->no_of_alignments)
			{
				unsigned long long aln_offset = i*MAX_NO_OF_ALIGNMENTS;
				barracuda_aln1_t * tmp_aln;
				if(opt->bwa_output)
				{
					bwt_aln1_t * output;
					output = (bwt_aln1_t*)malloc(tmp->no_of_alignments*sizeof(bwt_aln1_t));
					memset(output,0,tmp->no_of_alignments*sizeof(bwt_aln1_t));//avoid undefined bytes in .sai files

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
					stdout_bwt_aln1(output, tmp->no_of_alignments);
					free(output);
				}else
				{
					barracuda_aln1_t * output;
					output = (barracuda_aln1_t*)malloc(tmp->no_of_alignments*sizeof(barracuda_aln1_t));
					memset(output,0,tmp->no_of_alignments*sizeof(barracuda_aln1_t)); //avoid undefined bytes in .sai files

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
					stdout_barracuda_aln1(output, tmp->no_of_alignments);
					free(output);
				}
			}
		}

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
	cudaFree(kl_device);
	free(main_sequences_index);
	free(kl_host);
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


	fprintf(stderr,"[aln_core] Running %s CUDA mode.\n",PACKAGE_VERSION);
#if STDOUT_STRING_RESULT == 1
	fprintf(stdout,"[aln_core] Running %s CUDA mode.\n",PACKAGE_VERSION);
#endif
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
	report_cuda_error_GPU("[core] Error on cudaGetDeviceCount");

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
			report_cuda_error_GPU("[core] Error on cudaSetDevice");
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			report_cuda_error_GPU("[core] Error on cudaFuncCachePreferL1");
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
		 report_cuda_error_GPU("[core] Error_2 on cudaSetDevice");
		 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		 report_cuda_error_GPU("[core] Error_2 on cudaFuncCachePreferL1");
		 cudaGetDeviceProperties(&properties, sel_device);
		 report_cuda_error_GPU("[core] Error on cudaGetDeviceProperties");
		 cudaMemGetInfo(&mem_available, &total_mem);
		 report_cuda_error_GPU("[core] Error on cudaMemGetInfo");

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
		 report_cuda_error_GPU("[core] Error_2 on cudaMemGetInfo");
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
	report_cuda_error_GPU("[core] Error_3 on cudaMemGetInfo");

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
	report_cuda_error_GPU("[core] Error_2 on cudaGetDeviceCount");
	if (num_devices)
		{
			  //fprintf(stderr,"[deviceQuery] Querying CUDA devices:\n");
			  for (device = 0; device < num_devices; device++)
			  {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  report_cuda_error_GPU("[core] Error_3 on cudaGetDeviceProperties");
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
	report_cuda_error_GPU("[detect_cuda_device] Error_3 on cudaGetDeviceCount");
	cudaDeviceProp properties;
	int sel_device = -1;

	if (num_devices >= 1)
	{
	     fprintf(stderr, "[detect_cuda_device] Querying CUDA devices:\n");
		 int max_cuda_cores = 0, max_device = 0;
		 for (device = 0; device < num_devices; device++)
		 {
			  cudaGetDeviceProperties(&properties, device);
			  report_cuda_error_GPU("[detect_cuda_device] Error_4 on cudaGetDeviceCount");
			  mem_available = properties.totalGlobalMem;
			  //cudaMemGetInfo(&mem_available, &total_mem);
			  fprintf(stderr, "[detect_cuda_device]   Device %d ", device);
			  fprintf(stderr,"%s ", properties.name);
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
    report_cuda_error_GPU("[samse_core] Error on cudaSetDevice");
    cudaMemGetInfo(&mem_available, &total_mem);
    report_cuda_error_GPU("[samse_core] Error on cudaMemGetInfo");

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
				report_cuda_error_GPU("[samse_core] Error allocating memory for \"global_bwt\"");
				//copy bwt occurrence array from host to device and dump the bwt to save CPU memory
				cudaMemcpy (*global_bwt, bwt->bwt, bwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
				report_cuda_error_GPU("[samse_core] Error copying \"bwt\" to GPU.\n");
				//bind global variable bwt to texture memory bwt_occ_array
				cudaBindTexture(0, bwt_occ_array, *global_bwt, bwt->bwt_size*sizeof(uint32_t));
				report_cuda_error_GPU("[samse_core] Error binding texture \"bwt_occ_array\".\n");
				//copy bwt structure data to constant memory bwt_cuda structure
				cudaMemcpyToSymbol ( bwt_cuda, bwt, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
				report_cuda_error_GPU("[samse_core] Error \"copy bwt\" to GPU constant memory");
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
				report_cuda_error_GPU("[samse_core] Error on allocating cuda memory for \"global_rbwt\"");
				//copy reverse bwt occurrence array from host to device and dump the bwt to save CPU memory
				cudaMemcpy (*global_rbwt, rbwt->bwt, rbwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
				report_cuda_error_GPU("[samse_core] Error copying rbwt->bwt to GPU.");
				//bind global variable rbwt to texture memory rbwt_occ_array
				cudaBindTexture(0, rbwt_occ_array, *global_rbwt, rbwt->bwt_size*sizeof(uint32_t));
				report_cuda_error_GPU("[samse_core] Error binding texture \"rbwt_occ_array\".");
				//copy rbwt structure data to constant memory bwt_cuda structure
				cudaMemcpyToSymbol ( rbwt_cuda, rbwt, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
				report_cuda_error_GPU("[samse_core] Error copying \"rbwt\" to GPU constant memory");
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
	report_cuda_error_GPU("[samse_core] Error_2 on cudaMemGetInfo");

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
	report_cuda_error_GPU("[samse_core] Error on \"cudaGetDeviceProperties\".");

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
	cudaDeviceSynchronize(); //wait until kernel has had a chance to report error
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

//CUDA DEVICE CODE STARTING FROM THIS LINE
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

//END CUDA DEVICE CODE


