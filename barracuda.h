
/*
 * barracuda.h $Revision: 1.9 $ 
 *
 *  Created on: 8 Jun 2012
 *      Author: yhbl2
 *
 * WBL 28 Feb 2015 Remove deadcode for alternative sequences_array layout
 * WBL 11 Feb 2015 For SVN, retain bwtkl_t
 * WBL 16 Dec 2014 Add pack_length next_packed
 * WBL  4 Dec 2014 surpress gcc warnings
 */


#ifndef BARRACUDA_H_
#define BARRACUDA_H_

#include <stdint.h>
#include "bwt.h"
#include "bwtaln.h"

// For multikernel design
#define MAX_PASS_LENGTH 32 //must be maximum of the next 2
#define MAX_SEEDING_PASS_LENGTH 32
#define MAX_REGULAR_PASS_LENGTH 19 // rule of thumb for best speed = (seq_len-MAX_SEEDING_PASS_LENGTH)/2

#define SPLIT_ENGAGE MAX_PASS_LENGTH //when splitting starts to happen
#define SEQUENCE_HOLDER_LENGTH MAX_PASS_LENGTH //set this to max(MAX_PASS_LENGTH, SPLIT_ENGAGE) - this minimises the memory usage of a kernel
#define MAX_SEED_LENGTH 50 // not tested beyond 50

#define SUFFIX_CLUMP_WIDTH 0 //0 to disable

#define MAX_SEQUENCE_LENGTH 100
#define MAX_ALN_LENGTH 100 //Max length for alignment kernel, cannot go beyond 225 (ptx error)
#define MAX_NO_OF_ALIGNMENTS 10

#define MAX_NO_PARTIAL_HITS	25 //must be the maximum of the next 2
#define MAX_NO_SEEDING_PARTIALS 25
#define MAX_NO_REGULAR_PARTIALS 10



//for barracuda_read_seqs. In both cases the return is in nominal bases. barracuda_read_seqs() assumes 2 bases per byte
static //removes warning: no previous prototype for and linker error multiple definition of `pack_length'
inline unsigned int pack_length(const int number_of_bases) { 
  return number_of_bases; //gave up on new layout for texture sequences_array
}

static 
inline unsigned int next_packed(const int n_tot, const int n_seqs, const int first_length) {
  return n_tot + 250; //gave up on new layout for texture sequences_array
}

#ifdef __cplusplus
extern "C" {
#endif

#define aln_score2(m,o,e,p) ((m)*(p).s_mm + (o)*(p).s_gapo + (e)*(p).s_gape)

	////////////////////////////////////
	// Begin BarraCUDA specific structs
	////////////////////////////////////

	typedef struct {
		// The first 2 bytes is length, length = a[0]<<8|a[1]
		// alphabet is pack using only 4 bits, so 1 char contain 2 alphabet.
		unsigned char character[MAX_SEQUENCE_LENGTH/2];
	} bwt_sequence_t;

	typedef struct {
		bwtint_t width[MAX_ALN_LENGTH];
		char bid[MAX_ALN_LENGTH];
	} bwt_sequence_width_t;

	typedef struct {
		bwtint_t lim_k;
		bwtint_t lim_l;
		unsigned char cur_n_mm, cur_n_gapo,cur_n_gape;
		int best_diff;
		char start_pos;
		int score;
		int sequence_id;
		int best_cnt;
	} init_info_t;

	typedef struct {
		unsigned char n_mm, n_gapo,n_gape;
		bwtint_t k, l;
		int score;
		//int best_diff;
		int best_cnt;
	} barracuda_aln1_t;

	//host-device transit structure
	typedef struct
	{
		int no_of_alignments;
		int best_score; //marks best score achieved for particular sequence in the forward run - not updated for backward atm
		unsigned int sequence_id;
		int best_cnt;
		char pos;
		char finished;
	} alignment_meta_t;

	typedef struct {
		unsigned int widths[MAX_SEQUENCE_LENGTH+1];
		unsigned char bids[MAX_SEQUENCE_LENGTH+1];
	} widths_bids_t;

	struct align_store_lst_t
	{
	   barracuda_aln1_t val;
	   int sequence_id;
	   int start_pos;
	   char finished;
	   struct align_store_lst_t * next;
	};

	typedef struct align_store_lst_t align_store_lst;

	//a linked list of linked lists containing sequences with matching (variable length) suffixes - used for clumping
	struct suffix_bin_list_t {
		struct suffix_seq_list_t * seq;
		struct suffix_bin_list_t * next;
	};

	typedef struct suffix_bin_list_t suffix_bin_list;

	struct suffix_seq_list_t {
		unsigned int sequence_id;
		struct suffix_seq_list_t * next;
	};

	typedef struct suffix_seq_list_t suffix_seq_list;


	struct init_bin_list_t {
		struct init_list_t * aln_list;
		int score;
		char processed;
		struct init_bin_list_t * next;
	};

	typedef struct init_bin_list_t init_bin_list;

	struct init_list_t {
		init_info_t init;
		struct init_list_t * next;
	};

	typedef struct init_list_t init_list;

typedef struct {
  bwtint_t k;
  bwtint_t l;
} bwtkl_t;

	////////////////////////////////////
	// End BarraCUDA specific structs
	////////////////////////////////////

	////////////////////////////////////
	// Start BarraCUDA specific functions
	////////////////////////////////////

	int detect_cuda_device(void);

	int bwa_deviceQuery(void);

	void aln_quicksort2(bwt_aln1_t *aln, int m, int n);

	double diff_in_seconds(struct timeval *finishtime, struct timeval * starttime);

	//gap_opt_t *gap_init_bwaopt(gap_opt_t * opt);

	//gap_opt_t *gap_init_opt();

	unsigned long copy_bwts_to_cuda_memory(
			const char * prefix,
			uint32_t ** bwt,
			uint32_t mem_available,
			bwtint_t* seq_len);

	void barracuda_bwa_aln_core(const char *prefix,
			const char *fn_fa,
			gap_opt_t *opt);

	int bwa_read_seq_one_half_byte (
			bwa_seqio_t *bs,
			unsigned char * half_byte_array,
			unsigned int start_index,
			unsigned short * length,
			unsigned long long * clump_array,
			unsigned char clump_len,
			unsigned int sequence_id);

	int bwa_read_seq_one (bwa_seqio_t *bs,
			unsigned char * byte_array,
			unsigned short * length);

	void cuda_alignment_core(const char *prefix,
			bwa_seqio_t *ks,
			gap_opt_t *opt);


	///////////////////////////////////////////////////////////////
	// End CUDA ALN
	///////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif /* BARRACUDA_H_ */
