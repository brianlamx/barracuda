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

/* (0.7.0) beta: $Revision: 1.12 $
  27 Feb 2015 WBL Remove all changes to cuda_dfs_match
  25 Feb 2015 WBL Apply r1.55-62 scache_global_bwt
  21 Feb 2015 WBL Apply r1.25 mycache only NOT sequence_global sequence_stride
  redo r1.47 __align__ mycache
  11 Feb 2015 WBL taken from barracuda.cu r1.82
  Revert to r1.2 and provide old bwt_cuda_match_exact instead
*/


__device__ inline unsigned int numbits(unsigned int i, unsigned char c)
// with y of 32 bits which is a string sequence encoded with 2 bits per alphabet,
// count the number of occurrence of c ( one pattern of 2 bits alphabet ) in y
{
	i = ((c&2)?i:~i)>>1&((c&1)?i:~i)&0x55555555;
	i = (i&0x33333333)+(i>>2&0x33333333);
	return((i+(i>>4)&0x0F0F0F0F)*0x01010101)>>24;
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
				k = bwt_cuda.L2[c] + ((k==0)?0:bwt_cuda_occ(global_bwt, k - 1, c, 0, &last,l_mycache0)) + 1;
				//printf("Calculating l\n");
				//unsigned long long startL = l;
				//unsigned long long tmpL = bwt_cuda_occ(global_bwt, l, c);
				l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 1, &last,l_mycache0);
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
WBL 26 Feb 2016 not in use.... /*
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

#if USE_PETR_SPLIT_KERNEL > 0
WBL 26 Feb 2016 not in use.... /*
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

