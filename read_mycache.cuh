/*WBL 7 Jan 2015 (0.7.0) beta: $Revision: 1.6 $
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: read_mycache.cuh helper for CUDA version
   WBL 25 Feb 2015 Augment MY_DEBUG with assert
 */

#if max_mycache >= 4
__device__ inline uint32_t read_mycache_uint4(d_mycache4, const int x) {
#ifdef MY_DEBUG
  //if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) printf("read_mycache_uint4(%d) TESTING\n",x);
/*getting 
Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
with printf string  ... max_mycache=%d range error\n",x,max_mycache
*/
#endif /*MY_DEBUG*/
  switch(x){
  case 0: return mycache0->x;
  case 1: return mycache0->y;
  case 2: return mycache0->z;
  case 3: return mycache0->w;
#if max_mycache > 4
  case 4: return mycache1->x;
  case 5: return mycache1->y;
  case 6: return mycache1->z;
  case 7: return mycache1->w;
#if max_mycache > 8
  case 8: return mycache2->x;
  case 9: return mycache2->y;
  case 10: return mycache2->z;
  case 11: return mycache2->w;
  case 12: return mycache3->x;
  case 13: return mycache3->y;
  case 14: return mycache3->z;
  case 15: return mycache3->w;
#endif
#endif
default: 
#ifdef MY_DEBUG
//if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) printf("read_mycache_uint4(%d) max_mycache=%d range error\n",x,max_mycache);
  if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) printf("read_mycache_uint4(%d) max_mycache=??? range error\n",x);
#endif /*MY_DEBUG*/
    assert(0);
    return 0;
  }
}
#endif

#if max_mycache >= 2
__device__ inline uint32_t read_mycache_uint2(d_mycache8, const int x) {
  switch(x){
  case 0: return mycache0->x;
  case 1: return mycache0->y;
#if max_mycache > 2
  case 2: return mycache1->x;
  case 3: return mycache1->y;
#if max_mycache > 4
  case 4: return mycache2->x;
  case 5: return mycache2->y;
  case 6: return mycache3->x;
  case 7: return mycache3->y;
#if max_mycache > 8
  case 8: return mycache4->x;
  case 9: return mycache4->y;
  case 10: return mycache5->x;
  case 11: return mycache5->y;
  case 12: return mycache6->x;
  case 13: return mycache6->y;
  case 14: return mycache7->x;
  case 15: return mycache7->y;
#endif
#endif
#endif
  default: 
#ifdef MY_DEBUG
if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) printf("read_mycache_uint2(%d) max_mycache=%d range error\n",x,max_mycache);
#endif /*MY_DEBUG*/
    assert(0);
    return 0;
  }
}
#endif

__device__ inline uint32_t read_mycache_uint(d_mycache16, const int x) {
  switch(x){
  case 0: return *mycache0;
#if max_mycache > 1
  case 1: return *mycache1;
#if max_mycache > 2
  case 2: return *mycache2;
  case 3: return *mycache3;
#if max_mycache > 4
  case 4: return *mycache4;
  case 5: return *mycache5;
  case 6: return *mycache6;
  case 7: return *mycache7;
#if max_mycache > 8
  case 8: return *mycache8;
  case 9: return *mycache9;
  case 10: return *mycache10;
  case 11: return *mycache11;
  case 12: return *mycache12;
  case 13: return *mycache13;
  case 14: return *mycache14;
  case 15: return *mycache15;
#endif
#endif
#endif
#endif
  default: 
#ifdef MY_DEBUG
if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) printf("read_mycache_uint(%d) max_mycache=%d range error\n",x,max_mycache);
#endif /*MY_DEBUG*/
    assert(0);
    return 0;
  }
}
