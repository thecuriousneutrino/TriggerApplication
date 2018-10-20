
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
    printf(" --- user parameters \n");
    printf(" distance between test vertices = %f cm \n", distance_between_vertices);
    printf(" time step size = %d ns \n", time_step_size);
    printf(" water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
    printf(" coalesce_time = %f ns \n", coalesce_time);
    printf(" trigger_gate_up = %f ns \n", trigger_gate_up);
    printf(" trigger_gate_down = %f ns \n", trigger_gate_down);
    printf(" max_n_hits_per_job = %d \n", max_n_hits_per_job);
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

int GPU_daq::nhits_initialize_ToolDAQ(std::string PMTFile, std::string DetectorFile, std::string ParameterFile){

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
  output_file_base = "all_hits_emerald_threshold_";
  //  if( !read_the_pmts() ) return 0;
  {
    printf(" --- read pmts \n");
    //n_PMTs = read_number_of_pmts();
    {
      FILE *f=fopen(PMTFile.c_str(), "r");
      if (f == NULL){
	printf(" cannot read pmts file %s \n", PMTFile.c_str());
	fclose(f);
	return 0;
      }
      
      unsigned int n_pmts = 0;
      
      for (char c = getc(f); c != EOF; c = getc(f))
	if (c == '\n')
	  n_pmts ++;
      
      fclose(f);
      n_PMTs = n_pmts;
      
    }
    if( !n_PMTs ) return false;
    printf(" detector contains %d PMTs \n", n_PMTs);
    PMT_x = (double *)malloc(n_PMTs*sizeof(double));
    PMT_y = (double *)malloc(n_PMTs*sizeof(double));
    PMT_z = (double *)malloc(n_PMTs*sizeof(double));
    //if( !read_pmts() ) return false;
    {
      FILE *f=fopen(PMTFile.c_str(), "r");
      
      double x, y, z;
      unsigned int id;
      for( unsigned int i=0; i<n_PMTs; i++){
	if( fscanf(f, "%d %lf %lf %lf", &id, &x, &y, &z) != 4 ){
	  printf(" problem scanning pmt %d \n", i);
	  fclose(f);
	  return false;
	}
	PMT_x[id-1] = x;
	PMT_y[id-1] = y;
	PMT_z[id-1] = z;
      }
      
      fclose(f);

    }
    //print_pmts();
    
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
    
    double dark_rate = read_value_from_file("dark_rate", parameter_file); // Hz
    distance_between_vertices = read_value_from_file("distance_between_vertices", parameter_file); // cm
    wall_like_distance = read_value_from_file("wall_like_distance", parameter_file); // units of distance between vertices
    time_step_size = read_value_from_file("nhits_step_size", parameter_file); // ns
    nhits_window = read_value_from_file("nhits_window", parameter_file); // ns
    int extra_threshold = (int)(dark_rate*n_PMTs*nhits_window*1.e-9); // to account for dark current occupancy
    extra_threshold = 0;
    water_like_threshold_number_of_pmts = read_value_from_file("water_like_threshold_number_of_pmts", parameter_file) + extra_threshold;
    wall_like_threshold_number_of_pmts = read_value_from_file("wall_like_threshold_number_of_pmts", parameter_file) + extra_threshold;
    coalesce_time = read_value_from_file("coalesce_time", parameter_file); // ns
    trigger_gate_up = read_value_from_file("trigger_gate_up", parameter_file); // ns
    trigger_gate_down = read_value_from_file("trigger_gate_down", parameter_file); // ns
    max_n_hits_per_job = read_value_from_file("max_n_hits_per_job", parameter_file);
    output_txt = (bool)read_value_from_file("output_txt", parameter_file);
    correct_mode = read_value_from_file("correct_mode", parameter_file);
    number_of_kernel_blocks_3d.y = read_value_from_file("num_blocks_y", parameter_file);
    number_of_threads_per_block_3d.y = read_value_from_file("num_threads_per_block_y", parameter_file);
    number_of_threads_per_block_3d.x = read_value_from_file("num_threads_per_block_x", parameter_file);
    nhits_threshold_min = read_value_from_file("nhits_threshold_min", parameter_file);
    nhits_threshold_max = read_value_from_file("nhits_threshold_max", parameter_file);
  }
  if( use_verbose ){
    printf(" --- user parameters \n");
    printf(" distance between test vertices = %f cm \n", distance_between_vertices);
    printf(" time step size = %d ns \n", time_step_size);
    printf(" water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
    printf(" coalesce_time = %f ns \n", coalesce_time);
    printf(" trigger_gate_up = %f ns \n", trigger_gate_up);
    printf(" trigger_gate_down = %f ns \n", trigger_gate_down);
    printf(" max_n_hits_per_job = %d \n", max_n_hits_per_job);
  }
  if( use_timing )
    elapsed_parameters = stop_c_clock();




  /////////////////////
  // read detector ////
  /////////////////////
  // set: detector_height, detector_radius, pmt_radius
  if( use_timing )
    start_c_clock();
  //if( !read_the_detector() ) return 0;
  {
    printf(" --- read detector \n");
    //    if( !read_detector() ) return false;
    {
      FILE *f=fopen(DetectorFile.c_str(), "r");
      double pmt_radius;
      if( fscanf(f, "%lf %lf %lf", &detector_height, &detector_radius, &pmt_radius) != 3 ){
	printf(" problem scanning detector \n");
	fclose(f);
	return 0;
      }
      
      fclose(f);
    }
    printf(" detector height %f cm, radius %f cm \n", detector_height, detector_radius);
  }
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

int GPU_daq::nhits_execute(){

  start_total_cuda_clock();

  int n_events = 0;
  unsigned int nthresholds = nhits_threshold_max - nhits_threshold_min + 1;
  bool * triggerfound = (bool*)malloc(nthresholds * sizeof(bool));
  unsigned int * ntriggers = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));
  unsigned int * start_times = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));

  while( set_input_file_for_event(n_events) ){

    printf(" ------ analyzing event %d with %d nthresholds (min %d, max %d) \n", 
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

    printf(" --- execute kernel nhits \n");
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

	  //	  printf(" %d n_digits found in trigger window [%d, %d] \n", host_n_pmts_nhits[0], start_time, start_time + nhits_window);


	  if( host_n_pmts_nhits[0] > nhits_threshold_min + u ){
	    triggerfound[u] = true;
	    ntriggers[u] ++;
	    //	    printf(" trigger! n triggers %d \n", ntriggers[u]);
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
      printf(" --- threshold %d found %d triggers \n", nhits_threshold_min+u, ntriggers[u]);

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
      printf(" --- deallocate memory \n");
    free_event_memories_nhits();
    if( use_timing )
      elapsed_free += stop_cuda_clock();

    n_events ++;

  }

  elapsed_total += stop_total_cuda_clock();
  free(triggerfound);
  free(ntriggers);
  free(start_times);


  printf(" ------ analyzed %d events \n", n_events);

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

int GPU_daq::nhits_execute(std::vector<int> PMTid, std::vector<int> time){

  start_total_cuda_clock();

  int n_events = 0;
  unsigned int nthresholds = nhits_threshold_max - nhits_threshold_min + 1;
  bool * triggerfound = (bool*)malloc(nthresholds * sizeof(bool));
  unsigned int * ntriggers = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));
  unsigned int * start_times = (unsigned int*)malloc(nthresholds * sizeof(unsigned int));

  //  while( set_input_file_for_event(n_events) ){
  while(1){

    printf(" ------ analyzing event %d with %d nthresholds (min %d, max %d) \n", 
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
    if( !read_the_input_ToolDAQ(PMTid, time) ){
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
    unsigned int start_time = 0;
    unsigned int min_time = 0;

    printf(" --- execute kernel nhits \n");
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

	  //	  printf(" %d n_digits found in trigger window [%d, %d] \n", host_n_pmts_nhits[0], start_time, start_time + nhits_window);


	  if( host_n_pmts_nhits[0] > nhits_threshold_min + u ){
	    triggerfound[u] = true;
	    ntriggers[u] ++;
	    //	    printf(" trigger! n triggers %d \n", ntriggers[u]);
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
      printf(" --- threshold %d found %d triggers \n", nhits_threshold_min+u, ntriggers[u]);

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
      printf(" --- deallocate memory \n");
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


  printf(" ------ analyzed %d events \n", n_events);

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
    printf(" --- deallocate tofs memory \n");
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
    printf(" --- reset device \n");
  //  cudaDeviceReset();
  if( use_timing )
    elapsed_reset = stop_cuda_clock();



  //////////////////
  // print timing //
  //////////////////
  if( use_timing ){
    printf(" user parameters time : %f ms \n", elapsed_parameters);
    printf(" read pmts execution time : %f ms \n", elapsed_pmts);
    printf(" read detector execution time : %f ms \n", elapsed_detector);
    printf(" make test vertices execution time : %f ms \n", elapsed_vertices);
    printf(" setup threads candidates execution time : %f ms \n", elapsed_threads_candidates);
    printf(" make table of tofs execution time : %f ms \n", elapsed_tof);
    printf(" allocate tofs memory on device execution time : %f ms \n", elapsed_memory_tofs_dev);
    printf(" fill tofs memory on device execution time : %f ms \n", elapsed_tofs_copy_dev);
    printf(" deallocate tofs memory execution time : %f ms \n", elapsed_tofs_free);
    printf(" device reset execution time : %f ms \n", elapsed_reset);
    printf(" read input execution time : %f ms (%f) \n", elapsed_input, elapsed_input/elapsed_total);
    printf(" allocate candidates memory on host execution time : %f ms (%f) \n", elapsed_memory_candidates_host, elapsed_memory_candidates_host/elapsed_total);
    printf(" setup threads execution time : %f ms (%f) \n", elapsed_threads, elapsed_threads/elapsed_total);
    printf(" allocate memory on device execution time : %f ms (%f) \n", elapsed_memory_dev, elapsed_memory_dev/elapsed_total);
    printf(" fill memory on device execution time : %f ms (%f) \n", elapsed_copy_dev, elapsed_copy_dev/elapsed_total);
    printf(" correct kernel execution time : %f ms (%f) \n", elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin, elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin/elapsed_total);
    printf(" allocate candidates memory on device execution time : %f ms (%f) \n", elapsed_candidates_memory_dev, elapsed_candidates_memory_dev/elapsed_total);
    printf(" copy candidates to host execution time : %f ms (%f) \n", elapsed_candidates_copy_host, elapsed_candidates_copy_host/elapsed_total);
    printf(" choose candidates execution time : %f ms (%f) \n", choose_candidates, choose_candidates/elapsed_total);
    printf(" candidates kernel execution time : %f ms (%f) \n", elapsed_candidates_kernel, elapsed_candidates_kernel/elapsed_total);
    printf(" coalesce triggers execution time : %f ms (%f) \n", elapsed_coalesce, elapsed_coalesce/elapsed_total);
    printf(" separate triggers into gates execution time : %f ms (%f) \n", elapsed_gates, elapsed_gates/elapsed_total);
    printf(" write output execution time : %f ms (%f) \n", elapsed_write_output, elapsed_write_output/elapsed_total);
    printf(" deallocate memory execution time : %f ms (%f) \n", elapsed_free, elapsed_free/elapsed_total);
  }
  printf(" total execution time : %f ms \n", elapsed_total);

  return 1;

}

