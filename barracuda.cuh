#ifndef BWTALN_CUH
#define BWTALN_CUH

#include "bwt.h"
#include "bwtaln.h"
#include "barracuda.h"


///////////////////////////////////////////////////////////////
// Begin struct (Dag's test)
///////////////////////////////////////////////////////////////


// This struct is for use in CUDA implementation of the functions 
// bwa_cal_max_diff() and bwa_approx_mapQ(). In the variable
// "strand_type", bits 1-2 are "type", bit 3 is "strand" in 
// corresponding to CPU struct "bwa_seq_t".  
// 
typedef struct __align__(16)
{
	uint32_t len;
	//uint8_t strand_type;
	uint8_t strand;
	uint8_t type;
	uint8_t n_mm;
	uint32_t c1;
	uint32_t c2;
	
} bwa_maxdiff_mapQ_t;




///////////////////////////////////////////////////////////////
// End struct (Dag's test)
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
// Begin bwa_cal_pac_pos_cuda (Dag's test)
///////////////////////////////////////////////////////////////

void print_cuda_info();

__device__ ubyte_t _bwt_B0(const bwt_t *b, bwtint_t k);

__device__ ubyte_t _bwt_B02(bwtint_t k);

__device__ ubyte_t _rbwt_B02(bwtint_t k);

__device__ ubyte_t _bwt_B03(bwtint_t k, texture<uint4, 1, cudaReadModeElementType> *b);

__device__ uint32_t _bwt_bwt(const bwt_t *b, bwtint_t k);

__device__ uint32_t _bwt_bwt2(bwtint_t k);

__device__ uint32_t _rbwt_bwt2(bwtint_t k);

__device__ uint32_t _bwt_bwt3(bwtint_t k, texture<uint4, 1, cudaReadModeElementType> *b);

__device__ inline uint32_t* _bwt_occ_intv(const bwt_t *b, bwtint_t k);

__device__ int cuda_bwa_approx_mapQ(const bwa_maxdiff_mapQ_t *p, int mm);

__device__ int cuda_bwa_cal_maxdiff(int l, double err, double thres);



__device__ 
void update_indices(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty);



__device__ 
void update_indices_in_parallel(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty);



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
    const int n_sa_total);



__device__ 
void sort_reads(
    bwtint_t *sa_buf_arr,
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf_arr,
    int16_t *sa_origin,
    bwtint_t *sa_return,
    const int *n_sa_in_buf,
    int *n_sa_in_buf_prev);



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
    float fnr);
    

///////////////////////////////////////////////////////////////
// End bwa_cal_pac_pos_cuda (Dag's test)
///////////////////////////////////////////////////////////////







#endif // BWTALN_CUH
