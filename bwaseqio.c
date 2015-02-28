/* $Revision: 1.10 $ 
WBL 16 Dec 2014 use next_packed and pack_length
WBL  4 Dec 2014 add some checks that calloc was ok
 */

#include <zlib.h>
#include <ctype.h>
#include <assert.h>
#include <limits.h>
#include "bwtaln.h"
#include "utils.h"
#include "bamlite.h"
#include "barracuda.h"

#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

extern unsigned char nst_nt4_table[256];
static char bam_nt16_nt4_table[] = { 4, 0, 1, 4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4 };

struct __bwa_seqio_t {
	// for BAM input
	int is_bam, which; // 1st bit: read1, 2nd bit: read2, 3rd: SE
	bamFile fp;
	// for fastq input
	kseq_t *ks;
};

bwa_seqio_t *bwa_bam_open(const char *fn, int which)
{
	bwa_seqio_t *bs;
	bam_header_t *h;
	bs = (bwa_seqio_t*)calloc(1, sizeof(bwa_seqio_t));
	bs->is_bam = 1;
	bs->which = which;
	bs->fp = bam_open(fn, "r");
	h = bam_header_read(bs->fp);
	bam_header_destroy(h);
	return bs;
}

bwa_seqio_t *bwa_seq_open(const char *fn)
{
	gzFile fp;
	bwa_seqio_t *bs;
	bs = (bwa_seqio_t*)calloc(1, sizeof(bwa_seqio_t));
	fp = xzopen(fn, "r");
	bs->ks = kseq_init(fp);
	return bs;
}

void bwa_seq_close(bwa_seqio_t *bs)
{
	if (bs == 0) return;
	if (bs->is_bam) bam_close(bs->fp);
	else {
		gzclose(bs->ks->f->f);
		kseq_destroy(bs->ks);
	}
	free(bs);
}

void seq_reverse(int len, ubyte_t *seq, int is_comp)
{
	int i;
	if (is_comp) {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			if (tmp < 4) tmp = 3 - tmp;
			seq[len-1-i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
			seq[i] = tmp;
		}
		if (len&1) seq[i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
	} else {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			seq[len-1-i] = seq[i]; seq[i] = tmp;
		}
	}
}

int barracuda_trim_read(int trim_qual, barracuda_query_array_t *p, ubyte_t *qual);//surpress gcc warning
int barracuda_trim_read(int trim_qual, barracuda_query_array_t *p, ubyte_t *qual)
{
	int s = 0, l, max = 0, max_l = p->len;
	if (trim_qual < 1 || qual == 0) return 0;
	for (l = p->len - 1; l >= BWA_MIN_RDLEN; --l) {
		s += trim_qual - (qual[l] - 33);
		if (s < 0) break;
		if (s > max) max = s, max_l = l;
	}
	return p->len - max_l;
}

int bwa_trim_read(int trim_qual, bwa_seq_t *p)
{
	int s = 0, l, max = 0, max_l = p->len;
	if (trim_qual < 1 || p->qual == 0) return 0;
	for (l = p->len - 1; l >= BWA_MIN_RDLEN; --l) {
		s += trim_qual - (p->qual[l] - 33);
		if (s < 0) break;
		if (s > max) max = s, max_l = l;
	}
	p->clip_len = p->len = max_l;
	return p->full_len - p->len;
}

static bwa_seq_t *bwa_read_bam(bwa_seqio_t *bs, int n_needed, int *n, int is_comp, int trim_qual)
{
	bwa_seq_t *seqs, *p;
	int n_seqs, l, i;
	long n_trimmed = 0, n_tot = 0;
	bam1_t *b;

	b = bam_init1();
	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(n_needed, sizeof(bwa_seq_t));
	while (bam_read1(bs->fp, b) >= 0) {
		uint8_t *s, *q;
		int go = 0;
		if ((bs->which & 1) && (b->core.flag & BAM_FREAD1)) go = 1;
		if ((bs->which & 2) && (b->core.flag & BAM_FREAD2)) go = 1;
		if ((bs->which & 4) && !(b->core.flag& BAM_FREAD1) && !(b->core.flag& BAM_FREAD2))go = 1;
		if (go == 0) continue;
		l = b->core.l_qseq;
		p = &seqs[n_seqs++];
		p->tid = -1; // no assigned to a thread
		p->qual = 0;
		p->full_len = p->clip_len = p->len = l;
		n_tot += p->full_len;
		s = bam1_seq(b); q = bam1_qual(b);
		p->seq = (ubyte_t*)calloc(p->len + 1, 1);
		p->qual = (ubyte_t*)calloc(p->len + 1, 1);
		for (i = 0; i != p->full_len; ++i) {
			p->seq[i] = bam_nt16_nt4_table[(int)bam1_seqi(s, i)];
			p->qual[i] = q[i] + 33 < 126? q[i] + 33 : 126;
		}
		if (bam1_strand(b)) { // then reverse 
			seq_reverse(p->len, p->seq, 1);
			seq_reverse(p->len, p->qual, 0);
		}
		if (trim_qual >= 1) n_trimmed += bwa_trim_read(trim_qual, p);
		p->rseq = (ubyte_t*)calloc(p->full_len, 1);
		memcpy(p->rseq, p->seq, p->len);
		seq_reverse(p->len, p->seq, 0); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		seq_reverse(p->len, p->rseq, is_comp);
		p->name = strdup((const char*)bam1_qname(b));
		if (n_seqs == n_needed) break;
	}
	*n = n_seqs;
	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "[bwa_read_seq] %.1f%% bases are trimmed.\n", 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		free(seqs);
		bam_destroy1(b);
		return 0;
	}
	bam_destroy1(b);
	return seqs;
}

static barracuda_query_array_t *barracuda_read_bam(bwa_seqio_t *bs, unsigned int buffer, int *n, int trim_qual, unsigned int* acc_length)
{
	bwa_seq_t *seqs, *p;
	unsigned int n_seqs;
	int l, i;
	long n_trimmed = 0;
	unsigned int n_tot = 0;
	bam1_t *b;
	unsigned int memory_allocation = 0x160000;
	b = bam_init1();
	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(memory_allocation, sizeof(bwa_seq_t));
	while ((n_tot + 250) < (1ul<<(buffer+1))) {

		int reads = bam_read1(bs->fp, b);
		if (reads < 0) break;
		uint8_t *s, *q;
		int go = 0;
		if ((bs->which & 1) && (b->core.flag & BAM_FREAD1)) go = 1;
		if ((bs->which & 2) && (b->core.flag & BAM_FREAD2)) go = 1;
		if ((bs->which & 4) && !(b->core.flag& BAM_FREAD1) && !(b->core.flag& BAM_FREAD2))go = 1;
		if (go == 0) continue;
		l = b->core.l_qseq;
		p = &seqs[n_seqs++];
		p->full_len = p->clip_len = p->len = l;
		n_tot += p->full_len;
		s = bam1_seq(b); q = bam1_qual(b);
		p->seq = (ubyte_t*)calloc(p->len + 1, 1);
		for (i = 0; i != p->full_len; ++i) {
			p->seq[i] = bam_nt16_nt4_table[(int)bam1_seqi(s, i)];
			p->qual[i] = q[i] + 33 < 126? q[i] + 33 : 126;
		}
		if (bam1_strand(b)) { // then reverse
			seq_reverse(p->len, p->seq, 1);
			seq_reverse(p->len, p->qual, 0);
		}
		if (trim_qual >= 1) n_trimmed += bwa_trim_read(trim_qual, p);
		if (n_seqs == memory_allocation) break;
	}
	*n = n_seqs;
	*acc_length = n_tot;
	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "[bwa_read_seq] %.1f%% bases are trimmed.\n", 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		free(seqs);
		bam_destroy1(b);
		return 0;
	}
	bam_destroy1(b);

	barracuda_query_array_t *seqs2;
	seqs2 = (barracuda_query_array_t*)calloc(memory_allocation, sizeof(barracuda_query_array_t));
	for (i = 0; i < (int) n_seqs; i++){
		bwa_seq_t *p1 = seqs + i;
		barracuda_query_array_t *p2 = seqs2 + i;
		memcpy(p2->seq, p1->seq, p1->len);
		p2->len = p1->len;
	}
	return seqs2;
}

#define BARCODE_LOW_QUAL 13


#define write_to_half_byte_array(array,index,data) \
	(array)[(index)>>1]=(unsigned char)(((index)&0x1)?(((array)[(index)>>1]&0xF0)|((data)&0x0F)):(((data)<<4)|((array)[(index)>>1]&0x0F)))

//TODO adapt changes to bwa_read_seq
// read one sequence (reversed) from fastq file to half_byte_array (i.e. 4bit for one base pair )
int bwa_read_seq_one_half_byte (bwa_seqio_t *bs, unsigned char * half_byte_array, unsigned int start_index, unsigned short * length, unsigned long long * clump_array, unsigned char clump_len, unsigned int sequence_id)
{
	kseq_t *seq = bs->ks;
	int len, i;
	unsigned char c;

	if (((len = kseq_read(seq)) >= 0)) // added to process only when len is longer than mid tag
	{
		//To cut the length of the sequence
		if ( len > MAX_SEQUENCE_LENGTH) len = MAX_SEQUENCE_LENGTH;

		unsigned long long suffix = 0;
		for (i = 0; i < len; i++)
		{
			//fprintf(stderr,"now writing at position %i, character %i\n", start_index+i,nst_nt4_table[(int)seq->seq.s[len-i-1]]);
			c = nst_nt4_table[(int)seq->seq.s[len-i-1]];
			write_to_half_byte_array(half_byte_array,start_index+i,c);
			if(i>=len-clump_len){
				suffix = suffix*4 + c;
			}
		}
		//printf("index: %i\n", start_index);
		clump_array[sequence_id] = suffix;

		*length = len;
	}
	else
	{
		*length = 0;
	}

	return len;
}

static int variable_length_reported=0;
barracuda_query_array_t *barracuda_read_seqs(bwa_seqio_t *bs,  unsigned int buffer,  int *n, int mode, int trim_qual, unsigned int *acc_length, unsigned short *max_length, int *same_length)
{
	barracuda_query_array_t *seqs, *p;
	kseq_t *seq = bs->ks;
	unsigned int		n_seqs = 0,
						i,
						is_64 = mode&BWA_MODE_IL13,
						l_bc = mode>>24;

	*max_length  = 0;
	*same_length = 1;
	int first_length;
	int query_length = 0;
	bwtint_t		n_trimmed = 0;
	unsigned int n_tot = 0;
	size_t memory_allocation = 0x160000;

	if (l_bc > BWA_MAX_BCLEN) {
		fprintf(stderr, "[%s] the maximum barcode length is %d.\n", __func__, BWA_MAX_BCLEN);
		return 0;
	}

	if (bs->is_bam) return barracuda_read_bam(bs, buffer, n, trim_qual, acc_length); // l_bc has no effect for BAM input

	seqs = (barracuda_query_array_t*)calloc(memory_allocation, sizeof(bwa_seq_t));
	assert(seqs); //better than segfault

	
      //while space in buffers for next group of queries
      //In case same_length becomes false use constant rather than first_length
      //pre Dec 2014 assumed sequences up to 250
	while (next_packed(n_tot,n_seqs,first_length) < (1ul<<(buffer+1))){

		ubyte_t *seq_qual;
		query_length = kseq_read(seq);
		if(query_length < 0)
		{
			//fprintf(stderr,"eof reached, query_length: %i\n", query_length);
			//query_length = 0;
			break;
		}

		if ((mode & BWA_MODE_CFY) && (seq->comment.l != 0)) {
			// skip reads that are marked to be filtered by Casava
			char *s = index(seq->comment.s, ':');
			if (s && *(++s) == 'Y') {
				continue;
			}
		}
		if (is_64 && seq->qual.l)
			for (i = 0; i < seq->qual.l; ++i) seq->qual.s[i] -= 31;
		if (seq->seq.l <= l_bc) continue; // sequence length equals or smaller than the barcode length

		p = &seqs[n_seqs++];

		if (l_bc) { // then trim barcode
			for (i = l_bc; i < seq->seq.l; ++i)
				seq->seq.s[i - l_bc] = seq->seq.s[i];
			seq->seq.l -= l_bc; seq->seq.s[seq->seq.l] = 0;
			query_length = seq->seq.l;
			if (seq->qual.l) {
				for (i = l_bc; i < seq->qual.l; ++i)
					seq->qual.s[i - l_bc] = seq->qual.s[i];
					seq->qual.l -= l_bc; seq->qual.s[seq->qual.l] = 0;
				}
		}
		//TODO: TRIMMING!!!!
		if (trim_qual >= 1 && seq->qual.l) { // copy quality
			seq_qual = (ubyte_t*)strdup((char*)seq->qual.s);
			n_trimmed += barracuda_trim_read(trim_qual, p, seq_qual);
		}
		if(n_seqs==1) first_length =  query_length;
		else       if(first_length != query_length) {
		  if(!variable_length_reported) {
		    fprintf(stderr, "[barracuda_read_seq] Better if sequences of the same length. %d\n",n_seqs);
		    variable_length_reported=1;
		  }
		  *same_length = 0;
		}
		if(query_length > *max_length) {
		  assert(query_length<=USHRT_MAX);
		  *max_length = query_length;
		}

		p->len = query_length;
		n_tot += pack_length(p->len);
		p->seq = (char*)calloc(p->len, 1);
		assert(p->seq); //better than segfault
		for (i = 0; i != p->len; ++i)
			p->seq[i] = nst_nt4_table[(int)seq->seq.s[i]];
		//sequence now reversed later in barracuda_write_to_half_byte_array

		if (n_seqs == memory_allocation) {
			break;
		}
	}

	*n = n_seqs;
	*acc_length = n_tot;

	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "[barracuda_read_seq] %.1f%% bases are trimmed.\n", 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		free(seqs);
		return 0;
	}

	return seqs;
}


bwa_seq_t *bwa_read_seq(bwa_seqio_t *bs, int n_needed, int *n, int mode, int trim_qual)
{
	bwa_seq_t *seqs, *p;
	kseq_t *seq = bs->ks;
	int n_seqs, l, i, is_comp = mode&BWA_MODE_COMPREAD, is_64 = mode&BWA_MODE_IL13, l_bc = mode>>24;
	long n_trimmed = 0, n_tot = 0;

	if (l_bc > BWA_MAX_BCLEN) {
		fprintf(stderr, "[%s] the maximum barcode length is %d.\n", __func__, BWA_MAX_BCLEN);
		return 0;
	}
	if (bs->is_bam) return bwa_read_bam(bs, n_needed, n, is_comp, trim_qual); // l_bc has no effect for BAM input

	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(n_needed, sizeof(bwa_seq_t));
	while ((l = kseq_read(seq)) >= 0) {

		if ((mode & BWA_MODE_CFY) && (seq->comment.l != 0)) {
			// skip reads that are marked to be filtered by Casava
			char *s = index(seq->comment.s, ':');
			if (s && *(++s) == 'Y') {
				continue;
			}
		}
		if (is_64 && seq->qual.l)
			for (i = 0; i < seq->qual.l; ++i) seq->qual.s[i] -= 31;
		if (seq->seq.l <= l_bc) continue; // sequence length equals or smaller than the barcode length
		p = &seqs[n_seqs++];
		if (l_bc) { // then trim barcode
			for (i = 0; i < l_bc; ++i)
				p->bc[i] = (seq->qual.l && seq->qual.s[i]-33 < BARCODE_LOW_QUAL)? tolower(seq->seq.s[i]) : toupper(seq->seq.s[i]);
			p->bc[i] = 0;
			for (; i < seq->seq.l; ++i)
				seq->seq.s[i - l_bc] = seq->seq.s[i];
			seq->seq.l -= l_bc; seq->seq.s[seq->seq.l] = 0;
			if (seq->qual.l) {
				for (i = l_bc; i < seq->qual.l; ++i)
					seq->qual.s[i - l_bc] = seq->qual.s[i];
				seq->qual.l -= l_bc; seq->qual.s[seq->qual.l] = 0;
			}
			l = seq->seq.l;
		} else p->bc[0] = 0;
		p->tid = -1; // no assigned to a thread
		p->qual = 0;
		p->full_len = p->clip_len = p->len = l;
		n_tot += p->full_len;
		p->seq = (ubyte_t*)calloc(p->len, 1);
		for (i = 0; i != p->full_len; ++i)
			p->seq[i] = nst_nt4_table[(int)seq->seq.s[i]];
		if (seq->qual.l) { // copy quality
			p->qual = (ubyte_t*)strdup((char*)seq->qual.s);
			if (trim_qual >= 1) n_trimmed += bwa_trim_read(trim_qual, p);
		}
		p->rseq = (ubyte_t*)calloc(p->full_len, 1);
		memcpy(p->rseq, p->seq, p->len);
		seq_reverse(p->len, p->seq, 0); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		seq_reverse(p->len, p->rseq, is_comp);
		p->name = strdup((const char*)seq->name.s);
		{ // trim /[12]$
			int t = strlen(p->name);
			if (t > 2 && p->name[t-2] == '/' && (p->name[t-1] == '1' || p->name[t-1] == '2')) p->name[t-2] = '\0';
		}
		if (n_seqs == n_needed) {
			//printf("breaking because get n_needed reads %i\n", n_needed);
			break;
		}
	}
	*n = n_seqs;
	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "[bwa_read_seq] %.1f%% bases are trimmed.\n", 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		//printf("finished!\n");
		free(seqs);
		return 0;
	}

	//printf("n_tot %i\n", n_tot);
	return seqs;
}

void bwa_free_read_seq(int n_seqs, bwa_seq_t *seqs)
{
	int i, j;
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p = seqs + i;
		for (j = 0; j < p->n_multi; ++j)
			if (p->multi[j].cigar) free(p->multi[j].cigar);
		free(p->name);
		free(p->seq); free(p->rseq); free(p->qual); free(p->aln); free(p->md); free(p->multi);
		free(p->cigar);
	}
	free(seqs);
}
