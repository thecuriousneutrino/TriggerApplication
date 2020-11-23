
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
#include <algorithm>    // std::sort
#include <GPUFunctions.h>
#include <CUDA_Core.h>


int GPU_daq::nhits_initialize(){

  int argc = 0;
  const char* n_argv[] = {};
  const char **argv = n_argv;

  /////////////////////
  // initialise card //
  /////////////////////
  findCudaDevice(argc, argv);


  // initialise CUDA timing
  use_timing = true;
  if( use_timing ){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_stop);
  elapsed_parameters = 0; elapsed_pmts = 0; elapsed_detector = 0; elapsed_vertices = 0;
  elapsed_threads = 0; elapsed_tof = 0; elapsed_memory_tofs_dev = 0; elapsed_memory_candidates_host = 0; elapsed_tofs_copy_dev = 0;
  elapsed_input = 0; elapsed_memory_dev = 0; elapsed_copy_dev = 0; elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin = 0; 
  elapsed_threads_candidates = 0; elapsed_candidates_memory_dev = 0; elapsed_candidates_kernel = 0;
  elapsed_candidates_copy_host = 0; choose_candidates = 0; elapsed_coalesce = 0; elapsed_gates = 0; elapsed_free = 0; elapsed_total = 0;
  elapsed_tofs_free = 0; elapsed_reset = 0;
  use_verbose = false;


  ////////////////////
  // inspect device //
  ////////////////////
  // set: max_n_threads_per_block, max_n_blocks
  print_gpu_properties();




  ////////////////
  // read PMTs  //
  ////////////////
  // set: n_PMTs, PMT_x, PMT_y, PMT_z
  if( use_timing )
    start_c_clock();
  event_file_base = "all_hits_";
  event_file_suffix = ".txt";
  detector_file = "detector.txt";
  pmts_file = "all_pmts.txt";
  output_file_base = "all_hits_emerald_threshold_";
  if( !read_the_pmts() ) return 0;
  if( use_timing )
    elapsed_pmts = stop_c_clock();


  ///////////////////////
  // define parameters //
  ///////////////////////
  if( use_timing )
    start_c_clock();
  read_user_parameters_nhits();
  if( use_verbose ){
    printf(" [2] --- user parameters \n");
    printf(" [2] distance between test vertices = %f cm \n", distance_between_vertices);
    printf(" [2] time step size = %d ns \n", time_step_size);
    printf(" [2] water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
    printf(" [2] coalesce_time = %f ns \n", coalesce_time);
    printf(" [2] trigger_gate_up = %f ns \n", trigger_gate_up);
    printf(" [2] trigger_gate_down = %f ns \n", trigger_gate_down);
    printf(" [2] max_n_hits_per_job = %d \n", max_n_hits_per_job);
  }
  if( use_timing )
    elapsed_parameters = stop_c_clock();




  /////////////////////
  // read detector ////
  /////////////////////
  // set: detector_height, detector_radius, pmt_radius
  if( use_timing )
    start_c_clock();
  if( !read_the_detector() ) return 0;
  if( use_timing )
    elapsed_detector = stop_c_clock();




  ////////////////////////////////
  // fill tofs memory on device //
  ////////////////////////////////
  // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
  // memcpy: device_times_of_flight, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
  // texture: tex_times_of_flight
  if( use_timing )
    start_cuda_clock();
  fill_tofs_memory_on_device_nhits();
  if( use_timing )
    elapsed_tofs_copy_dev = stop_cuda_clock();


  ///////////////////////
  // initialize output //
  ///////////////////////
  initialize_output_nhits();


  return 1;

}

int GPU_daq::nhits_initialize_ToolDAQ(std::string ParameterFile,int nPMTs, int fTriggerSearchWindow, int fTriggerSearchWindowStep, int fTriggerThreshold, int fTriggerSaveWindowPre, int fTriggerSaveWindowPost){

  int argc = 0;
  const char* n_argv[] = {};
  const char **argv = n_argv;

  /////////////////////
  // initialise card //
  /////////////////////
  findCudaDevice(argc, argv);


  // initialise CUDA timing
  use_timing = true;
  if( use_timing ){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_stop);
  elapsed_parameters = 0; elapsed_pmts = 0; elapsed_detector = 0; elapsed_vertices = 0;
  elapsed_threads = 0; elapsed_tof = 0; elapsed_memory_tofs_dev = 0; elapsed_memory_candidates_host = 0; elapsed_tofs_copy_dev = 0;
  elapsed_input = 0; elapsed_memory_dev = 0; elapsed_copy_dev = 0; elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin = 0; 
  elapsed_threads_candidates = 0; elapsed_candidates_memory_dev = 0; elapsed_candidates_kernel = 0;
  elapsed_candidates_copy_host = 0; choose_candidates = 0; elapsed_coalesce = 0; elapsed_gates = 0; elapsed_free = 0; elapsed_total = 0;
  elapsed_tofs_free = 0; elapsed_reset = 0;
  use_verbose = true;


  ////////////////////
  // inspect device //
  ////////////////////
  // set: max_n_threads_per_block, max_n_blocks
  print_gpu_properties();




  ////////////////
  // read PMTs  //
  ////////////////
  // set: n_PMTs
  if( use_timing )
    start_c_clock();
  output_file_base = "all_hits_emerald_threshold_";
  //  if( !read_the_pmts() ) return 0;
  {
    printf(" [2] --- read pmts \n");
    n_PMTs = nPMTs;
      
    if( !n_PMTs ) return 0;
    printf(" [2] detector contains %d PMTs \n", n_PMTs);
  }
  if( use_timing )
    elapsed_pmts = stop_c_clock();


  ///////////////////////
  // define parameters //
  ///////////////////////
  if( use_timing )
    start_c_clock();
  //  read_user_parameters_nhits();
  {
    std::string parameter_file = ParameterFile;
    
    twopi = 2.*acos(-1.);
    speed_light_water = 29.9792/1.3330; // speed of light in water, cm/ns
    //double speed_light_water = 22.490023;
    
    //    time_step_size = read_value_from_file("nhits_step_size", parameter_file); // ns
    time_step_size = fTriggerSearchWindowStep;
    //    nhits_window = read_value_from_file("nhits_window", parameter_file); // ns
    nhits_window = fTriggerSearchWindow;
    //    trigger_gate_up = read_value_from_file("trigger_gate_up", parameter_file); // ns
    trigger_gate_up = fTriggerSaveWindowPost;
    //    trigger_gate_down = read_value_from_file("trigger_gate_down", parameter_file); // ns
    trigger_gate_down = - fTriggerSaveWindowPre;
    max_n_hits_per_job = read_value_from_file("max_n_hits_per_job", parameter_file);
    output_txt = (bool)read_value_from_file("output_txt", parameter_file);
    correct_mode = read_value_from_file("correct_mode", parameter_file);
    number_of_kernel_blocks_3d.y = read_value_from_file("num_blocks_y", parameter_file);
    number_of_threads_per_block_3d.y = read_value_from_file("num_threads_per_block_y", parameter_file);
    number_of_threads_per_block_3d.x = read_value_from_file("num_threads_per_block_x", parameter_file);
    //    nhits_threshold_min = read_value_from_file("nhits_threshold_min", parameter_file);
    //    nhits_threshold_max = read_value_from_file("nhits_threshold_max", parameter_file);
    nhits_threshold_min = fTriggerThreshold;
    nhits_threshold_max = fTriggerThreshold;
  }
  if( use_verbose ){
    printf(" [2] --- user parameters \n");
    printf(" [2] time step size = %d ns \n", time_step_size);
    printf(" [2] trigger_gate_up = %f ns \n", trigger_gate_up);
    printf(" [2] trigger_gate_down = %f ns \n", trigger_gate_down);
    printf(" [2] max_n_hits_per_job = %d \n", max_n_hits_per_job);
    printf(" [2] nhits_window = %d \n", nhits_window);
    printf(" [2] nhits_threshold_min = %d, max = %d \n", nhits_threshold_min, nhits_threshold_max);
  }
  if( use_timing )
    elapsed_parameters = stop_c_clock();







  ////////////////////////////////
  // fill tofs memory on device //
  ////////////////////////////////
  // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
  // memcpy: device_times_of_flight, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
  // texture: tex_times_of_flight
  if( use_timing )
    start_cuda_clock();
  fill_tofs_memory_on_device_nhits();
  if( use_timing )
    elapsed_tofs_copy_dev = stop_cuda_clock();


  ///////////////////////
  // initialize output //
  ///////////////////////
  initialize_output_nhits();


  return 1;

}

int GPU_daq::nhits_execute(){

  start_total_cuda_clock();

  int n_events = 0;
  unsigned int nthresholds = nhits_threshold_max - nhits_threshold_min + 1;
  bool * triggerfound = (bool*)malloc(nthresholds * sizeof(bool));
  unsigned int * ntriggers = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));
  unsigned int * start_times = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));

  while( set_input_file_for_event(n_events) ){

    printf(" [2] ------ analyzing event %d with %d nthresholds (min %d, max %d) \n", 
	   n_events+1, nthresholds, nhits_threshold_min, nhits_threshold_max);

    memset(ntriggers, 0, nthresholds*sizeof(unsigned int));
    memset(start_times, 0, nthresholds*sizeof(unsigned int));

    ////////////////
    // read input //
    ////////////////
    // set: n_hits, host_ids, host_times, time_offset, n_time_bins
    // use: time_offset, n_test_vertices
    // memcpy: constant_n_time_bins, constant_n_hits
    if( use_timing )
      start_c_clock();
    if( !read_the_input() ){
      if( use_timing )
	elapsed_input += stop_c_clock();
      n_events ++;
      continue;
    }
    if( use_timing )
      elapsed_input += stop_c_clock();
  


    ////////////////////////////////////////
    // allocate candidates memory on host //
    ////////////////////////////////////////
    // use: n_time_bins
    // malloc: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    allocate_candidates_memory_on_host();
    if( use_timing )
      elapsed_memory_candidates_host += stop_cuda_clock();


    ////////////////////////////////////////////////
    // set number of blocks and threads per block //
    ////////////////////////////////////////////////
    // set: number_of_kernel_blocks, number_of_threads_per_block
    // use: n_test_vertices, n_hits
    if( use_timing )
      start_c_clock();
    if( !setup_threads_nhits() ) return 0;
    if( use_timing )
      elapsed_threads += stop_c_clock();


    ///////////////////////////////////////
    // allocate correct memory on device //
    ///////////////////////////////////////
    // use: n_test_vertices, n_hits, n_time_bins
    // cudamalloc: device_ids, device_times, device_n_pmts_per_time_bin
    if( use_timing )
      start_cuda_clock();
    allocate_correct_memory_on_device_nhits();
    if( use_timing )
      elapsed_memory_dev += stop_cuda_clock();


    //////////////////////////////////////
    // copy input into device variables //
    //////////////////////////////////////
    // use: n_hits
    // memcpy: device_ids, device_times, constant_time_offset
    // texture: tex_ids, tex_times
    if( use_timing )
      start_cuda_clock();
    fill_correct_memory_on_device();
    if( use_timing )
      elapsed_copy_dev += stop_cuda_clock();



    ////////////////////
    // execute kernel //
    ////////////////////
    if( use_timing )
      start_cuda_clock();
    unsigned int start_time = 0;
    unsigned int min_time = 0;

    printf(" [2] --- execute kernel nhits \n");
    while(start_time <= the_max_time) {
      memset(triggerfound, false, nthresholds*sizeof(bool));
      checkCudaErrors(cudaMemset(device_n_pmts_nhits, 0, 1*sizeof(unsigned int)));
      //      checkCudaErrors(cudaMemset(device_time_nhits, 0, (nhits_window/time_step_size + 1)*sizeof(unsigned int)));
      kernel_nhits<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_nhits, start_time, nhits_window);
      cudaThreadSynchronize();
      getLastCudaError("kernel_nhits execution failed\n");

      checkCudaErrors(cudaMemcpy(host_n_pmts_nhits,
				 device_n_pmts_nhits,
				 1*sizeof(unsigned int),
				 cudaMemcpyDeviceToHost));

      min_time = the_max_time+1;

      for(unsigned int u=0; u<nthresholds; u++){
	if( start_times[u] <= start_time ){ // initially true as both zero

	  //	  printf(" [2] %d n_digits found in trigger window [%d, %d] \n", host_n_pmts_nhits[0], start_time, start_time + nhits_window);


	  if( host_n_pmts_nhits[0] > nhits_threshold_min + u ){
	    triggerfound[u] = true;
	    ntriggers[u] ++;
	    //	    printf(" [2] trigger! n triggers %d \n", ntriggers[u]);
	  }
	  
	  if( triggerfound[u] )
	    start_times[u] = start_time + trigger_gate_up;
	  else
	    start_times[u] = start_time + time_step_size;
	}

	if( min_time > start_times[u] )
	  min_time = start_times[u];

      }

      start_time = min_time;

    }
    if( use_timing )
      elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin += stop_cuda_clock();

    for(unsigned int u=0; u<nthresholds; u++)
      printf(" [2] --- threshold %d found %d triggers \n", nhits_threshold_min+u, ntriggers[u]);

    //////////////////
    // write output //
    //////////////////
    if( use_timing )
      start_cuda_clock();
    write_output_nhits(ntriggers);
    if( use_timing )
      elapsed_write_output += stop_cuda_clock();

    /////////////////////////////
    // deallocate event memory //
    /////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- deallocate memory \n");
    free_event_memories_nhits();
    if( use_timing )
      elapsed_free += stop_cuda_clock();

    n_events ++;

  }

  elapsed_total += stop_total_cuda_clock();
  free(triggerfound);
  free(ntriggers);
  free(start_times);


  printf(" [2] ------ analyzed %d events \n", n_events);

  ///////////////////////
  // normalize timings //
  ///////////////////////
  if( use_timing ){
    elapsed_input /= n_events;
    elapsed_memory_candidates_host /= n_events;
    elapsed_threads /= n_events;
    elapsed_memory_dev /= n_events;
    elapsed_copy_dev /= n_events;
    elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin /= n_events;
    elapsed_threads_candidates /= n_events;
    elapsed_candidates_memory_dev /= n_events;
    elapsed_candidates_kernel /= n_events;
    elapsed_candidates_copy_host /= n_events;
    elapsed_coalesce /= n_events;
    elapsed_gates /= n_events;
    elapsed_write_output /= n_events;
    elapsed_free /= n_events;
  }
  elapsed_total /= n_events;

  return 1;
}

int GPU_daq::nhits_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts){

  start_total_cuda_clock();

  int n_events = 0;
  unsigned int nthresholds = nhits_threshold_max - nhits_threshold_min + 1;
  bool * triggerfound = (bool*)malloc(nthresholds * sizeof(bool));
  unsigned int * ntriggers = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));
  unsigned int * start_times = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));

  //  while( set_input_file_for_event(n_events) ){
  while(1){

    printf(" [2] ------ analyzing event %d with %d nthresholds (min %d, max %d) \n", 
	   n_events+1, nthresholds, nhits_threshold_min, nhits_threshold_max);

    memset(ntriggers, 0, nthresholds*sizeof(unsigned int));
    memset(start_times, 0, nthresholds*sizeof(unsigned int));

    ////////////////
    // read input //
    ////////////////
    // set: n_hits, host_ids, host_times, time_offset, n_time_bins
    // use: time_offset, n_test_vertices
    // memcpy: constant_n_time_bins, constant_n_hits
    int earliest_time = 0;
    if( use_timing )
      start_c_clock();
    if( !read_the_input_ToolDAQ(PMTid, time, &earliest_time) ){
      if( use_timing )
	elapsed_input += stop_c_clock();
      n_events ++;
      break;
    }
    if( use_timing )
      elapsed_input += stop_c_clock();
  


    ////////////////////////////////////////
    // allocate candidates memory on host //
    ////////////////////////////////////////
    // use: n_time_bins
    // malloc: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    allocate_candidates_memory_on_host();
    if( use_timing )
      elapsed_memory_candidates_host += stop_cuda_clock();


    ////////////////////////////////////////////////
    // set number of blocks and threads per block //
    ////////////////////////////////////////////////
    // set: number_of_kernel_blocks, number_of_threads_per_block
    // use: n_test_vertices, n_hits
    if( use_timing )
      start_c_clock();
    if( !setup_threads_nhits() ) return 0;
    if( use_timing )
      elapsed_threads += stop_c_clock();


    ///////////////////////////////////////
    // allocate correct memory on device //
    ///////////////////////////////////////
    // use: n_test_vertices, n_hits, n_time_bins
    // cudamalloc: device_ids, device_times, device_n_pmts_per_time_bin
    if( use_timing )
      start_cuda_clock();
    allocate_correct_memory_on_device_nhits();
    if( use_timing )
      elapsed_memory_dev += stop_cuda_clock();


    //////////////////////////////////////
    // copy input into device variables //
    //////////////////////////////////////
    // use: n_hits
    // memcpy: device_ids, device_times, constant_time_offset
    // texture: tex_ids, tex_times
    if( use_timing )
      start_cuda_clock();
    fill_correct_memory_on_device();
    if( use_timing )
      elapsed_copy_dev += stop_cuda_clock();



    ////////////////////
    // execute kernel //
    ////////////////////
    if( use_timing )
      start_cuda_clock();
    unsigned int start_time = earliest_time;
    unsigned int min_time = 0;

    printf(" [2] --- execute kernel nhits starting from time %d max %d \n", start_time, the_max_time);
    while(start_time <= the_max_time) {
      memset(triggerfound, false, nthresholds*sizeof(bool));
      checkCudaErrors(cudaMemset(device_n_pmts_nhits, 0, 1*sizeof(unsigned int)));
      //      checkCudaErrors(cudaMemset(device_time_nhits, 0, (nhits_window/time_step_size + 1)*sizeof(unsigned int)));
      kernel_nhits<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_nhits, start_time, nhits_window);
      cudaThreadSynchronize();
      getLastCudaError("kernel_nhits execution failed\n");

      checkCudaErrors(cudaMemcpy(host_n_pmts_nhits,
				 device_n_pmts_nhits,
				 1*sizeof(unsigned int),
				 cudaMemcpyDeviceToHost));

      min_time = the_max_time+1;

      // F. Nova verbose output
      //      printf(" [2] interval (%d, %d) has %d hits \n", start_time, start_time + nhits_window, host_n_pmts_nhits[0]);

      for(unsigned int u=0; u<nthresholds; u++){
	if( start_times[u] <= start_time ){ // initially true as both zero

	  //	  printf(" [2] %d n_digits found in trigger window [%d, %d] \n", host_n_pmts_nhits[0], start_time, start_time + nhits_window);


	  if( host_n_pmts_nhits[0] > nhits_threshold_min + u ){
	    triggerfound[u] = true;
	    ntriggers[u] ++;
	    //	    printf(" [2] trigger! n triggers %d \n", ntriggers[u]);

	    // F. Nova verbose output
	    //printf(" [2] found trigger in interval (%d, %d) with %d hits \n", start_time, start_time + nhits_window, host_n_pmts_nhits[0]);
	    trigger_ns->push_back(host_n_pmts_nhits[0]);
	    trigger_ts->push_back(start_time + nhits_window - time_step_size);

	  }
	  
	  if( triggerfound[u] )
	    start_times[u] = start_time + trigger_gate_up;
	  else
	    start_times[u] = start_time + time_step_size;
	}

	if( min_time > start_times[u] )
	  min_time = start_times[u];

      }

      start_time = min_time;

    }
    if( use_timing )
      elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin += stop_cuda_clock();

    for(unsigned int u=0; u<nthresholds; u++)
      printf(" [2] --- threshold %d found %d triggers \n", nhits_threshold_min+u, ntriggers[u]);

    //////////////////
    // write output //
    //////////////////
    if( use_timing )
      start_cuda_clock();
    write_output_nhits(ntriggers);
    if( use_timing )
      elapsed_write_output += stop_cuda_clock();

    /////////////////////////////
    // deallocate event memory //
    /////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- deallocate memory \n");
    free_event_memories_nhits();
    if( use_timing )
      elapsed_free += stop_cuda_clock();

    n_events ++;

    break;
  }

  elapsed_total += stop_total_cuda_clock();
  free(triggerfound);
  free(ntriggers);
  free(start_times);


  printf(" [2] ------ analyzed %d events \n", n_events);

  ///////////////////////
  // normalize timings //
  ///////////////////////
  if( use_timing ){
    elapsed_input /= n_events;
    elapsed_memory_candidates_host /= n_events;
    elapsed_threads /= n_events;
    elapsed_memory_dev /= n_events;
    elapsed_copy_dev /= n_events;
    elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin /= n_events;
    elapsed_threads_candidates /= n_events;
    elapsed_candidates_memory_dev /= n_events;
    elapsed_candidates_kernel /= n_events;
    elapsed_candidates_copy_host /= n_events;
    elapsed_coalesce /= n_events;
    elapsed_gates /= n_events;
    elapsed_write_output /= n_events;
    elapsed_free /= n_events;
  }
  elapsed_total /= n_events;

  return 1;
}

int GPU_daq::nhits_finalize(){


  //////////////////////////////
  // deallocate global memory //
  //////////////////////////////
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" [2] --- deallocate tofs memory \n");
  free_global_memories();
  if( use_timing )
    elapsed_tofs_free = stop_cuda_clock();



  //////////////////
  // reset device //
  //////////////////
  // -- needed to flush the buffer which holds printf from each thread
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" [2] --- reset device \n");
  //  cudaDeviceReset();
  if( use_timing )
    elapsed_reset = stop_cuda_clock();



  //////////////////
  // print timing //
  //////////////////
  if( use_timing ){
    printf(" [2] user parameters time : %f ms \n", elapsed_parameters);
    printf(" [2] read pmts execution time : %f ms \n", elapsed_pmts);
    printf(" [2] read detector execution time : %f ms \n", elapsed_detector);
    printf(" [2] make test vertices execution time : %f ms \n", elapsed_vertices);
    printf(" [2] setup threads candidates execution time : %f ms \n", elapsed_threads_candidates);
    printf(" [2] make table of tofs execution time : %f ms \n", elapsed_tof);
    printf(" [2] allocate tofs memory on device execution time : %f ms \n", elapsed_memory_tofs_dev);
    printf(" [2] fill tofs memory on device execution time : %f ms \n", elapsed_tofs_copy_dev);
    printf(" [2] deallocate tofs memory execution time : %f ms \n", elapsed_tofs_free);
    printf(" [2] device reset execution time : %f ms \n", elapsed_reset);
    printf(" [2] read input execution time : %f ms (%f) \n", elapsed_input, elapsed_input/elapsed_total);
    printf(" [2] allocate candidates memory on host execution time : %f ms (%f) \n", elapsed_memory_candidates_host, elapsed_memory_candidates_host/elapsed_total);
    printf(" [2] setup threads execution time : %f ms (%f) \n", elapsed_threads, elapsed_threads/elapsed_total);
    printf(" [2] allocate memory on device execution time : %f ms (%f) \n", elapsed_memory_dev, elapsed_memory_dev/elapsed_total);
    printf(" [2] fill memory on device execution time : %f ms (%f) \n", elapsed_copy_dev, elapsed_copy_dev/elapsed_total);
    printf(" [2] correct kernel execution time : %f ms (%f) \n", elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin, elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin/elapsed_total);
    printf(" [2] allocate candidates memory on device execution time : %f ms (%f) \n", elapsed_candidates_memory_dev, elapsed_candidates_memory_dev/elapsed_total);
    printf(" [2] copy candidates to host execution time : %f ms (%f) \n", elapsed_candidates_copy_host, elapsed_candidates_copy_host/elapsed_total);
    printf(" [2] choose candidates execution time : %f ms (%f) \n", choose_candidates, choose_candidates/elapsed_total);
    printf(" [2] candidates kernel execution time : %f ms (%f) \n", elapsed_candidates_kernel, elapsed_candidates_kernel/elapsed_total);
    printf(" [2] coalesce triggers execution time : %f ms (%f) \n", elapsed_coalesce, elapsed_coalesce/elapsed_total);
    printf(" [2] separate triggers into gates execution time : %f ms (%f) \n", elapsed_gates, elapsed_gates/elapsed_total);
    printf(" [2] write output execution time : %f ms (%f) \n", elapsed_write_output, elapsed_write_output/elapsed_total);
    printf(" [2] deallocate memory execution time : %f ms (%f) \n", elapsed_free, elapsed_free/elapsed_total);
  }
  printf(" [2] total execution time : %f ms \n", elapsed_total);

  return 1;

}

