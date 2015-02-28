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

/* (0.7.0) beta: $Revision: 1.9 $
  27 Feb 2015 WBL Back to r1.1 remove all printf
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

//NICKED from cuda.cuh r1.13

#ifndef direct_sequence
__device__ int bwt_cuda_match_exact( uint32_t * global_bwt, unsigned int length, const unsigned char * str, bwtint_t *k0, bwtint_t *l0)
#else
__device__ int bwt_cuda_match_exact( uint32_t * global_bwt, unsigned int length, const unsigned int sequence_offset, bwtint_t *k0, bwtint_t *l0)
#endif
//exact match algorithm
{
	//printf("in exact match function\n");
	int i;
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
		else  k = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, k - 1, c, 0) + 1;

		l = bwt_cuda.L2[c] + bwt_cuda_occ(global_bwt, l, c, 0);
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
