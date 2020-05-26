
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <helper_cuda.h>
#include <sys/time.h>
#include <library_daq.h>
#include <CUDA_Core.h>

//
// kernel routine
// 

// __global__ identifier says it's a kernel function
__global__ void kernel_correct_times_and_get_n_pmts_per_time_bin(histogram_t *ct){

  int time_bin = get_time_bin();

  if( time_bin < 0 ) return;

#if defined __HISTOGRAM_UINT__
  atomicAdd(&ct[time_bin],1);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif

  //  printf( " hit %d (nh %d) id %d t %d; vertex %d (nv %d) tof %f  %d \n", hit_index, constant_n_hits, ids[hit_index], t[hit_index], vertex_index, constant_n_test_vertices, tof, ct[time_index]);

  return;

}


// __global__ identifier says it's a kernel function
__global__ void kernel_correct_times_and_get_n_pmts_per_time_bin_and_direction_bin(unsigned int *ct, bool * dirs){

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int tid = tid_y * gridDim.x * blockDim.x + tid_x;

  // tid runs from 0 to n_test_vertices * n_hits:
  //      vertex 0           vertex 1       ...     vertex m
  // (hit 0, ..., hit n; hit 0, ..., hit n; ...; hit 0, ..., hit n);

  unsigned int vertex_index = (int)(tid/constant_n_hits);
  unsigned int hit_index = tid - vertex_index * constant_n_hits;

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return ;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return ;

  int time_bin = get_time_bin_for_vertex_and_hit(vertex_index, hit_index) - constant_n_time_bins*vertex_index;

  if( time_bin < 0 ) return;

  for(unsigned int idir = 0; idir < constant_n_direction_bins; idir++){

    unsigned int dir_index = device_get_direction_index_at_pmt(
							       tex1Dfetch(tex_ids,hit_index), 
							       vertex_index, 
							       idir
							       );
    
    //    bool good_direction = (bool)tex1Dfetch(tex_directions_for_vertex_and_pmt, dir_index);

    bool good_direction = dirs[dir_index];


    if( good_direction ){
      atomicAdd(&ct[device_get_direction_index_at_time(time_bin, vertex_index, idir)],1);
    }
    
  }





  //  printf( " hit %d (nh %d) id %d t %d; vertex %d (nv %d) tof %f  %d \n", hit_index, constant_n_hits, ids[hit_index], t[hit_index], vertex_index, constant_n_test_vertices, tof, ct[time_index]);

  return;

}


__global__ void kernel_correct_times(unsigned int *ct){


  int time_bin = get_time_bin();

  if( time_bin < 0 ) return;

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int tid = tid_y * gridDim.x * blockDim.x + tid_x;

  ct[tid] = time_bin;


  return;

}

__device__ int get_time_bin_for_vertex_and_hit(unsigned int vertex_index, unsigned int hit_index){

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return -1;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return -1;

  unsigned int vertex_block = constant_n_time_bins*vertex_index;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  return device_get_time_index(
			       (tex1Dfetch(tex_times,hit_index)
				- tex1Dfetch(tex_times_of_flight,
					     device_get_distance_index(
								       tex1Dfetch(tex_ids,hit_index),
								       vertex_block2
								       )
					     )
				+ constant_time_offset)/constant_time_step_size
			       ,
			       vertex_block
			       );
  

}


__device__ float get_light_dx_for_vertex_and_hit(unsigned int vertex_index, unsigned int hit_index){

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return -1;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return -1;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  return tex1Dfetch(tex_light_dx,
		    device_get_distance_index(
					      tex1Dfetch(tex_ids,hit_index),
					      vertex_block2
					      )
		    );


}


__device__ float get_light_dy_for_vertex_and_hit(unsigned int vertex_index, unsigned int hit_index){

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return -1;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return -1;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  return tex1Dfetch(tex_light_dy,
		    device_get_distance_index(
					      tex1Dfetch(tex_ids,hit_index),
					      vertex_block2
					      )
		    );


}


__device__ float get_light_dz_for_vertex_and_hit(unsigned int vertex_index, unsigned int hit_index){

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return -1;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return -1;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  return tex1Dfetch(tex_light_dz,
		    device_get_distance_index(
					      tex1Dfetch(tex_ids,hit_index),
					      vertex_block2
					      )
		    );


}


__device__ float get_light_dr_for_vertex_and_hit(unsigned int vertex_index, unsigned int hit_index){

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return -1;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return -1;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  return tex1Dfetch(tex_light_dr,
		    device_get_distance_index(
					      tex1Dfetch(tex_ids,hit_index),
					      vertex_block2
					      )
		    );


}



__device__ int get_time_bin(){

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int tid = tid_y * gridDim.x * blockDim.x + tid_x;

  // tid runs from 0 to n_test_vertices * n_hits:
  //      vertex 0           vertex 1       ...     vertex m
  // (hit 0, ..., hit n; hit 0, ..., hit n; ...; hit 0, ..., hit n);

  unsigned int vertex_index = (int)(tid/constant_n_hits);
  unsigned int hit_index = tid - vertex_index * constant_n_hits;

  return get_time_bin_for_vertex_and_hit(vertex_index, hit_index);


}






__global__ void kernel_histo_one_thread_one_vertex( unsigned int *ct, histogram_t *histo ){

  
  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;

  unsigned int vertex_index = tid_x;
  unsigned int bin ;
  unsigned int max = constant_n_test_vertices*constant_n_hits;
  unsigned int size = vertex_index * constant_n_hits;

  for( unsigned int ihit=0; ihit<constant_n_hits; ihit++){
    bin = size + ihit;
    if( bin < max)
#if defined __HISTOGRAM_UINT__
      atomicAdd(&histo[ct[bin]],1);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif
      ;
  }
  
}

__global__ void kernel_histo_stride( unsigned int *ct, histogram_t *histo){

  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while( i < constant_n_hits*constant_n_test_vertices ){
#if defined __HISTOGRAM_UINT__
    atomicAdd( &histo[ct[i]], 1);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif
    i += stride;
  }


}



__global__ void kernel_histo_iterated( unsigned int *ct, histogram_t *histo, unsigned int offset ){

  
  extern __shared__ unsigned int temp[];
  unsigned int index = threadIdx.x + offset;
  temp[index] = 0;
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int size = blockDim.x * gridDim.x;
  unsigned int max = constant_n_hits*constant_n_test_vertices;
  while( i < max ){
    atomicAdd( &temp[ct[i]], 1);
    i += size;
  }
  __syncthreads();
#if defined __HISTOGRAM_UINT__
    atomicAdd( &(histo[index]), temp[index] );
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif


}


__global__ void kernel_histo_stride_2d( unsigned int *ct, histogram_t *histo){

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  unsigned int size = blockDim.x * gridDim.x;
  unsigned int max = constant_n_hits*constant_n_test_vertices;

  // map the two 2D indices to a single linear, 1D index
  int tid = tid_y * size + tid_x;

  /*
  unsigned int vertex_index = (int)(tid/constant_n_time_bins);
  unsigned int time_index = tid - vertex_index * constant_n_time_bins;

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return;

  // skip if thread is assigned to nonexistent hit
  if( time_index >= constant_n_time_bins ) return;

  unsigned int vertex_block = constant_n_time_bins*vertex_index;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;
  */

  unsigned int stride = blockDim.y * gridDim.y*size;

  while( tid < max ){
#if defined __HISTOGRAM_UINT__
    atomicAdd( &histo[ct[tid]], 1);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif
    tid += stride;
  }


}


__global__ void kernel_histo_per_vertex( unsigned int *ct, histogram_t *histo){

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  if( tid_x >= constant_n_test_vertices ) return;

  unsigned int vertex_offset = tid_x*constant_n_hits;
  unsigned int bin;
  unsigned int stride = blockDim.y*gridDim.y;
  unsigned int ihit = vertex_offset + tid_y;

  while( ihit<vertex_offset+constant_n_hits){

    bin = ct[ihit];
    //histo[bin]++;
#if defined __HISTOGRAM_UINT__
    atomicAdd( &histo[bin], 1);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif
    ihit += stride;

  }
  __syncthreads();
}

__global__ void kernel_histo_per_vertex_shared( unsigned int *ct, histogram_t *histo){
  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  if( tid_x >= constant_n_test_vertices ) return;

  unsigned int vertex_offset = tid_x*constant_n_hits;
  unsigned int bin;
  unsigned int stride = blockDim.y*gridDim.y;
  unsigned int stride_block = blockDim.y;
  unsigned int ihit = vertex_offset + tid_y;
  unsigned int time_offset = tid_x*constant_n_time_bins;

  unsigned int local_ihit = threadIdx.y;
  extern __shared__ unsigned int temp[];
  while( local_ihit<constant_n_time_bins ){
    temp[local_ihit] = 0;
    local_ihit += stride_block;
  }

  __syncthreads();

  while( ihit<vertex_offset+constant_n_hits){

    bin = ct[ihit];
    atomicAdd(&temp[bin - time_offset],1);
    ihit += stride;

  }

  __syncthreads();

  local_ihit = threadIdx.y;
  while( local_ihit<constant_n_time_bins ){
#if defined __HISTOGRAM_UINT__
    atomicAdd( &histo[local_ihit+time_offset], temp[local_ihit]);
#else
  printf("This function needs checking before using it without _HISTOGRAM_UINT_ defined"); 
  return;
#endif
    local_ihit += stride_block;
  }


}

__global__ void kernel_correct_times_and_get_histo_per_vertex_shared(histogram_t *ct){

  unsigned int vertex_index = blockIdx.x;
  if( vertex_index >= constant_n_test_vertices ) return;

  unsigned int local_ihit_initial = threadIdx.x + threadIdx.y*blockDim.x;
  unsigned int local_ihit = local_ihit_initial;
  unsigned int stride_block = blockDim.x*blockDim.y;
  unsigned int stride = stride_block*gridDim.y;
  unsigned int hit_index = local_ihit + stride_block*blockIdx.y;

  unsigned int bin;
  unsigned int time_offset = vertex_index*constant_n_time_bins;

  extern __shared__ unsigned int temp[];
  while( local_ihit<constant_n_time_bins ){
    temp[local_ihit] = 0;
    local_ihit += stride_block;
  }

  __syncthreads();

  // unsigned int vertex_block = const_n_time_bins*vertex_index;
  // unsigned int vertex_block2 = const_n_PMTs*vertex_index;
  // unsigned int v1, v2, v4;
  // float v3;
  while( hit_index<constant_n_hits){
    // v1 = __ldg(times + hit_index);
    // v2 = *(ids + hit_index) + vertex_block2 - 1;
    // v3 = __ldg(times_of_flight + v2);
    // v4 = (v1 - v3 + const_time_offset)/const_time_step_size;
    // bin = (v4+vertex_block);
    bin = get_time_bin_for_vertex_and_hit(vertex_index, hit_index);
    atomicAdd(&temp[bin - time_offset],1);
    hit_index += stride;

  }

  __syncthreads();

  local_ihit = local_ihit_initial;
  while( local_ihit<constant_n_time_bins ){
    //    atomicAdd( &ct[local_ihit+time_offset], temp[local_ihit]);
#if defined __HISTOGRAM_UCHAR__
    ct[local_ihit+time_offset] = min(255, ct[local_ihit+time_offset] + temp[local_ihit]);
#else
    ct[local_ihit+time_offset] += temp[local_ihit];
#endif
    local_ihit += stride_block;
  }


}


__global__ void kernel_correct_times_and_get_histo_per_vertex_shared(histogram_t *ct, unsigned int* times, unsigned int* ids, unsigned short* times_of_flight,
     unsigned int const_n_test_vertices, unsigned int const_n_time_bins, unsigned int const_n_hits, 
     unsigned int const_n_PMTs, offset_t const_time_offset, unsigned int const_time_step_size)
{

  unsigned int vertex_index = blockIdx.x;
  if( vertex_index >= const_n_test_vertices ) return;
  unsigned int local_ihit_initial = threadIdx.x + threadIdx.y*blockDim.x;
  unsigned int local_ihit = local_ihit_initial;
  unsigned int stride_block = blockDim.x*blockDim.y;
  unsigned int stride = stride_block*gridDim.y;
  unsigned int hit_index = threadIdx.x + threadIdx.y*blockDim.x + stride_block*blockIdx.y;
  unsigned int bin;
  unsigned int time_offset = vertex_index*const_n_time_bins;
  extern __shared__ unsigned int temp[];
  while( local_ihit<const_n_time_bins ){
    temp[local_ihit] = 0;
    local_ihit += stride_block;
  }
  __syncthreads();
  unsigned int vertex_block = const_n_time_bins*vertex_index;
  unsigned int vertex_block2 = const_n_PMTs*vertex_index;
  unsigned int v1, v2, v4;
  float v3;
  while( hit_index<const_n_hits){
    v1 = __ldg(times + hit_index);
    v2 = *(ids + hit_index) + vertex_block2 - 1;
    v3 = __ldg(times_of_flight + v2);
    v4 = (v1 - v3 + const_time_offset)/const_time_step_size;
    bin = (v4+vertex_block);
    atomicAdd(&temp[bin - time_offset],1);
    hit_index += stride;
  }
  __syncthreads();
  local_ihit = local_ihit_initial;
  while( local_ihit<const_n_time_bins ){
#if defined __HISTOGRAM_UCHAR__
    ct[local_ihit+time_offset] = min(255, ct[local_ihit+time_offset] + temp[local_ihit]);
#else
    ct[local_ihit+time_offset] += temp[local_ihit];
#endif
    local_ihit += stride_block;
  }
}


__global__ void kernel_correct_times_calculate_averages_and_get_histo_per_vertex_shared(histogram_t *ct,float *dx,float *dy,float *dz, unsigned int *ncone){

  //  number_of_kernel_blocks_3d = (n test vertices, 1)
  //  gridDim.x = n test vertices,  gridDim.y = 1
  //  number_of_threads_per_block_3d = (1, 1024)
  //  blockDim.x = 1,  blockDim.y = 1024
  //  grid size = n test vertices * 1024

  unsigned int vertex_index = blockIdx.x;
  if( vertex_index >= constant_n_test_vertices ) return;

  // unique thread id = initial hit index
  unsigned int local_ihit_initial = threadIdx.x + threadIdx.y*blockDim.x;
  unsigned int local_ihit = local_ihit_initial;

  // stride block = 1024
  unsigned int stride_block = blockDim.x*blockDim.y;

  // stride = 1024
  unsigned int stride = stride_block*gridDim.y;

  // hit index = initial hit index
  unsigned int hit_index = local_ihit + stride_block*blockIdx.y;

  unsigned int bin;
  unsigned int time_offset = vertex_index*constant_n_time_bins;

  float light_dx, light_dy, light_dz, light_dr;

  // init shared memory to zero
  extern __shared__ unsigned int temp_ct[];
  while( local_ihit<constant_n_time_bins ){
    temp_ct[local_ihit] = 0;
    local_ihit += stride_block;
  }

  __syncthreads();

  // count hits per time bin in shared memory
  // add dx, dy, dz
  while( hit_index<constant_n_hits){

    bin = get_time_bin_for_vertex_and_hit(vertex_index, hit_index);
    atomicAdd(&temp_ct[bin - time_offset],1);

    light_dx = get_light_dx_for_vertex_and_hit(vertex_index, hit_index);
    atomicAdd(&dx[bin],light_dx);

    light_dy = get_light_dy_for_vertex_and_hit(vertex_index, hit_index);
    atomicAdd(&dy[bin],light_dy);

    light_dz = get_light_dz_for_vertex_and_hit(vertex_index, hit_index);
    atomicAdd(&dz[bin],light_dz);

    hit_index += stride;

  }

  __syncthreads();

  // count hits per time bin
  local_ihit = local_ihit_initial;
  while( local_ihit<constant_n_time_bins ){
    //    atomicAdd( &ct[local_ihit+time_offset], temp_ct[local_ihit]);
#if defined __HISTOGRAM_UCHAR__
    ct[local_ihit+time_offset] = min(255, ct[local_ihit+time_offset] + temp_ct[local_ihit]);
#else
    ct[local_ihit+time_offset] += temp_ct[local_ihit];
#endif
    local_ihit += stride_block;
  }

  __syncthreads();

  local_ihit = local_ihit_initial;
  while( local_ihit<constant_n_time_bins ){
    temp_ct[local_ihit] = 0;
    local_ihit += stride_block;
  }

  __syncthreads();


  float axis_dx, axis_dy, axis_dz, axis_dr, costheta;

  hit_index = local_ihit_initial + stride_block*blockIdx.y;
  while( hit_index<constant_n_hits){

    bin = get_time_bin_for_vertex_and_hit(vertex_index, hit_index);
    
    light_dx = get_light_dx_for_vertex_and_hit(vertex_index, hit_index);
    light_dy = get_light_dy_for_vertex_and_hit(vertex_index, hit_index);
    light_dz = get_light_dz_for_vertex_and_hit(vertex_index, hit_index);
    light_dr = get_light_dr_for_vertex_and_hit(vertex_index, hit_index);

    axis_dx = dx[bin];
    axis_dy = dy[bin];
    axis_dz = dz[bin];
    axis_dr = sqrt(pow(axis_dx,2) + pow(axis_dy,2) + pow(axis_dz,2));

    costheta = (light_dx*axis_dx + light_dy*axis_dy + light_dz*axis_dz)/(light_dr*axis_dr);
    if( fabs( costheta - constant_cerenkov_costheta) < constant_costheta_cone_cut )
      atomicAdd(&temp_ct[bin - time_offset],1);

    hit_index += stride;

  }

  __syncthreads();

  local_ihit = local_ihit_initial;
  while( local_ihit<constant_n_time_bins ){
    atomicAdd( &ncone[local_ihit+time_offset], temp_ct[local_ihit]);
    local_ihit += stride_block;
  }




}



 

// __global__ identifier says it's a kernel function
__global__ void kernel_nhits(unsigned int *ct, unsigned int start_time, unsigned int nhits_window){

  // get unique id for each thread in each block
  unsigned int hit_index = threadIdx.x + blockDim.x*blockIdx.x;

  int stride = blockDim.x * gridDim.x;
  unsigned int time_bin;
  //  unsigned int old_nhits;

  while( hit_index < constant_n_hits ){

    time_bin = tex1Dfetch(tex_times,hit_index);
    
    
    if( time_bin >= start_time &&
	time_bin <= start_time + nhits_window){
      atomicAdd(&ct[0],1);
      //      time[old_nhits] = time_bin;
    }

    hit_index += stride;
    
  }


  return;

}
