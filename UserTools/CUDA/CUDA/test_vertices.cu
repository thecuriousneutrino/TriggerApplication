
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
#include <GPUFunctions.h>
#include <CUDA_Core.h>


int GPU_daq::test_vertices_initialize(){

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
  elapsed_threads = 0; elapsed_tof = 0; elapsed_directions = 0; elapsed_memory_tofs_dev = 0; elapsed_memory_directions_dev = 0; elapsed_memory_candidates_host = 0; elapsed_tofs_copy_dev = 0; elapsed_directions_copy_dev = 0;
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
  read_user_parameters();
  set_output_file();
  printf(" [2] --- user parameters \n");
  printf(" [2] dark_rate %f \n", dark_rate);
  printf(" [2] distance between test vertices = %f cm \n", distance_between_vertices);
  printf(" [2] wall_like_distance %f \n", wall_like_distance);
  printf(" [2] water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
  printf(" [2] wall_like_threshold_number_of_pmts %d \n", wall_like_threshold_number_of_pmts);
  printf(" [2] coalesce_time = %f ns \n", coalesce_time);
  printf(" [2] trigger_gate_up = %f ns \n", trigger_gate_up);
  printf(" [2] trigger_gate_down = %f ns \n", trigger_gate_down);
  printf(" [2] max_n_hits_per_job = %d \n", max_n_hits_per_job);
  printf(" [2] output_txt %d \n", output_txt);
  printf(" [2] correct_mode %d \n", correct_mode);
  printf(" [2] num_blocks_y %d \n", number_of_kernel_blocks_3d.y);
  printf(" [2] num_threads_per_block_x %d \n", number_of_threads_per_block_3d.x);
  printf(" [2] num_threads_per_block_y %d \n", number_of_threads_per_block_3d.y);
  printf(" [2] cylindrical_grid %d \n", cylindrical_grid);
  printf(" [2] time step size = %d ns \n", time_step_size);
  printf(" [2] write_output_mode %d \n", write_output_mode);
  if( correct_mode == 9 ){
    printf(" [2] n_direction_bins_theta %d, n_direction_bins_phi %d, n_direction_bins %d \n",
	   n_direction_bins_theta, n_direction_bins_phi, n_direction_bins);
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




  ////////////////////////
  // make test vertices //
  ////////////////////////
  // set: n_test_vertices, n_water_like_test_vertices, vertex_x, vertex_y, vertex_z
  // use: detector_height, detector_radius
  if( use_timing )
    start_c_clock();
  make_test_vertices();
  if( use_timing )
    elapsed_vertices = stop_c_clock();



  //////////////////////////////
  // table of times_of_flight //
  //////////////////////////////
  // set: host_times_of_flight, time_offset
  // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
  // malloc: host_times_of_flight
  if( use_timing )
    start_c_clock();
  make_table_of_tofs();
  if( use_timing )
    elapsed_tof = stop_c_clock();



  if( correct_mode == 9 ){
    //////////////////////////////
    // table of directions //
    //////////////////////////////
    // set: host_directions_phi, host_directions_cos_theta
    // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
    // malloc: host_directions_phi, host_directions_cos_theta
    if( use_timing )
      start_c_clock();
    make_table_of_directions();
    if( use_timing )
      elapsed_directions = stop_c_clock();
  }


  ////////////////////////////////////
  // allocate tofs memory on device //
  ////////////////////////////////////
  // use: n_test_vertices, n_PMTs
  // cudamalloc: device_times_of_flight
  if( use_timing )
    start_cuda_clock();
  allocate_tofs_memory_on_device();
  if( use_timing )
    elapsed_memory_tofs_dev = stop_cuda_clock();


  if( correct_mode == 9 ){
    ////////////////////////////////////
    // allocate direction memory on device //
    ////////////////////////////////////
    // use: n_test_vertices, n_PMTs
    // cudamalloc: device_directions_phi, device_directions_cos_theta
    if( use_timing )
      start_cuda_clock();
    allocate_directions_memory_on_device();
    if( use_timing )
      elapsed_memory_directions_dev = stop_cuda_clock();
  }


  ////////////////////////////////
  // fill tofs memory on device //
  ////////////////////////////////
  // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
  // memcpy: device_times_of_flight, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
  // texture: tex_times_of_flight
  if( use_timing )
    start_cuda_clock();
  fill_tofs_memory_on_device();
  if( use_timing )
    elapsed_tofs_copy_dev = stop_cuda_clock();


  if( correct_mode == 9 ){
    ////////////////////////////////
    // fill directions memory on device //
    ////////////////////////////////
    // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
    // memcpy: device_directions_phi, device_directions_cos_theta, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
    // texture: tex_directions_phi, tex_directions_cos_theta
    if( use_timing )
      start_cuda_clock();
    fill_directions_memory_on_device();
    if( use_timing )
      elapsed_directions_copy_dev = stop_cuda_clock();
  }

  ///////////////////////
  // initialize output //
  ///////////////////////
  initialize_output();


  return 1;

}

int GPU_daq::test_vertices_initialize_ToolDAQ(double f_detector_length, double f_detector_radius, double f_pmt_radius, std::string ParameterFile, std::vector<int> tube_no,std::vector<float> tube_x,std::vector<float> tube_y,std::vector<float> tube_z, float f_dark_rate,
  float f_distance_between_vertices,
  float f_wall_like_distance,
  float f_water_like_threshold_number_of_pmts,
  float f_wall_like_threshold_number_of_pmts,
  float f_coalesce_time,
  float f_trigger_gate_up,
  float f_trigger_gate_down,
  int f_output_txt,
  int f_correct_mode,
  int f_n_direction_bins_theta,
  bool f_cylindrical_grid,
  float f_costheta_cone_cut,
  bool f_select_based_on_cone,
  bool  f_trigger_threshold_adjust_for_noise,
  int f_max_n_hits_per_job,
  int f_num_blocks_y,
  int f_num_threads_per_block_y,
  int f_num_threads_per_block_x,
  int f_write_output_mode
){

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
  elapsed_threads = 0; elapsed_tof = 0; elapsed_directions = 0; elapsed_memory_tofs_dev = 0; elapsed_memory_directions_dev = 0; elapsed_memory_candidates_host = 0; elapsed_tofs_copy_dev = 0; elapsed_directions_copy_dev = 0;
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
  //  detector_file = DetectorFile;
  output_file_base = "all_hits_emerald_threshold_";
  //  if( !read_the_pmts() ) return 0;
  {
    printf(" [2] --- read pmts \n");
    n_PMTs = tube_no.size();
    {
      if( n_PMTs != tube_x.size() ||
	  n_PMTs != tube_y.size() ||
	  n_PMTs != tube_z.size() ){
	printf("pmt problem: n_PMTs %d xs %d ys %d zs %d \n", n_PMTs, tube_x.size(), tube_y.size(), tube_z.size());
	return 0;
      }
    }
    if( !n_PMTs ) return 0;
    PMT_x = (double *)malloc(n_PMTs*sizeof(double));
    PMT_y = (double *)malloc(n_PMTs*sizeof(double));
    PMT_z = (double *)malloc(n_PMTs*sizeof(double));
    for( unsigned int i=0; i<n_PMTs; i++){
      PMT_x[i] = tube_x[i];
      PMT_y[i] = tube_y[i];
      PMT_z[i] = tube_z[i];
    }
    printf(" [2] detector contains %d PMTs \n", n_PMTs);
  }
  if( use_timing )
    elapsed_pmts = stop_c_clock();


  ///////////////////////
  // define parameters //
  ///////////////////////
  if( use_timing )
    start_c_clock();
  //read_user_parameters();
  {
    
    twopi = 2.*acos(-1.);
    speed_light_water = 29.9792/1.3330; // speed of light in water, cm/ns
  //double speed_light_water = 22.490023;

    cerenkov_costheta =1./1.3330;
    cerenkov_angle_water = acos(cerenkov_costheta);
    costheta_cone_cut = f_costheta_cone_cut;
    select_based_on_cone = f_select_based_on_cone;

    dark_rate = f_dark_rate; // Hz
    cylindrical_grid = f_cylindrical_grid;
    distance_between_vertices = f_distance_between_vertices; // cm
    wall_like_distance = f_wall_like_distance; // units of distance between vertices
    time_step_size = (unsigned int)(sqrt(3.)*distance_between_vertices/(4.*speed_light_water)); // ns
    int extra_threshold = 0;
    if( f_trigger_threshold_adjust_for_noise ){
      extra_threshold = (int)(dark_rate*n_PMTs*2.*time_step_size*1.e-9); // to account for dark current occupancy
    }
    water_like_threshold_number_of_pmts = f_water_like_threshold_number_of_pmts + extra_threshold;
    wall_like_threshold_number_of_pmts = f_wall_like_threshold_number_of_pmts + extra_threshold;
    coalesce_time = f_coalesce_time; // ns
    trigger_gate_up = f_trigger_gate_up; // ns
    trigger_gate_down = f_trigger_gate_down; // ns
    max_n_hits_per_job = f_max_n_hits_per_job;
    output_txt = f_output_txt;
    correct_mode = f_correct_mode;
    write_output_mode = f_write_output_mode;
    number_of_kernel_blocks_3d.y = f_num_blocks_y;
    number_of_threads_per_block_3d.y = f_num_threads_per_block_y;
    number_of_threads_per_block_3d.x = f_num_threads_per_block_x;
    
    n_direction_bins_theta = f_n_direction_bins_theta;
    n_direction_bins_phi = 2*(n_direction_bins_theta - 1);
    n_direction_bins = n_direction_bins_phi*n_direction_bins_theta - 2*(n_direction_bins_phi - 1);
    
  }
  //  set_output_file();
  printf(" [2] --- user parameters \n");
  printf(" [2] dark_rate %f \n", dark_rate);
  printf(" [2] distance between test vertices = %f cm \n", distance_between_vertices);
  printf(" [2] wall_like_distance %f \n", wall_like_distance);
  printf(" [2] water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
  printf(" [2] wall_like_threshold_number_of_pmts %d \n", wall_like_threshold_number_of_pmts);
  printf(" [2] coalesce_time = %f ns \n", coalesce_time);
  printf(" [2] trigger_gate_up = %f ns \n", trigger_gate_up);
  printf(" [2] trigger_gate_down = %f ns \n", trigger_gate_down);
  printf(" [2] max_n_hits_per_job = %d \n", max_n_hits_per_job);
  printf(" [2] output_txt %d \n", output_txt);
  printf(" [2] correct_mode %d \n", correct_mode);
  printf(" [2] num_blocks_y %d \n", number_of_kernel_blocks_3d.y);
  printf(" [2] num_threads_per_block_x %d \n", number_of_threads_per_block_3d.x);
  printf(" [2] num_threads_per_block_y %d \n", number_of_threads_per_block_3d.y);
  printf(" [2] cylindrical_grid %d \n", cylindrical_grid);
  printf(" [2] time step size = %d ns \n", time_step_size);
  printf(" [2] write_output_mode %d \n", write_output_mode);
  if( correct_mode == 9 ){
    printf(" [2] n_direction_bins_theta %d, n_direction_bins_phi %d, n_direction_bins %d \n",
	   n_direction_bins_theta, n_direction_bins_phi, n_direction_bins);
  }
  if( use_timing )
    elapsed_parameters = stop_c_clock();




  /////////////////////
  // read detector ////
  /////////////////////
  // set: detector_height, detector_radius, pmt_radius
  if( use_timing )
    start_c_clock();
  //  if( !read_the_detector() ) return 0;
  detector_height = f_detector_length;
  detector_radius = f_detector_radius;
  printf(" [2] detector height %f cm, radius %f cm \n", detector_height, detector_radius);
  //  pmt_radius = f_pmt_radius;
  if( use_timing )
    elapsed_detector = stop_c_clock();




  ////////////////////////
  // make test vertices //
  ////////////////////////
  // set: n_test_vertices, n_water_like_test_vertices, vertex_x, vertex_y, vertex_z
  // use: detector_height, detector_radius
  if( use_timing )
    start_c_clock();
  make_test_vertices();
  if( use_timing )
    elapsed_vertices = stop_c_clock();



  //////////////////////////////
  // table of times_of_flight //
  //////////////////////////////
  // set: host_times_of_flight, time_offset
  // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
  // malloc: host_times_of_flight
  if( use_timing )
    start_c_clock();
  make_table_of_tofs();
  if( use_timing )
    elapsed_tof = stop_c_clock();

  if( correct_mode == 9 ){
    //////////////////////////////
    // table of directions //
    //////////////////////////////
    // set: host_directions_phi, host_directions_cos_theta
    // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
    // malloc: host_directions_phi, host_directions_cos_theta
    if( use_timing )
      start_c_clock();
    make_table_of_directions();
    if( use_timing )
      elapsed_directions = stop_c_clock();
  }


  ////////////////////////////////////
  // allocate tofs memory on device //
  ////////////////////////////////////
  // use: n_test_vertices, n_PMTs
  // cudamalloc: device_times_of_flight
  if( use_timing )
    start_cuda_clock();
  allocate_tofs_memory_on_device();
  if( use_timing )
    elapsed_memory_tofs_dev = stop_cuda_clock();


  if( correct_mode == 9 ){
    ////////////////////////////////////
    // allocate direction memory on device //
    ////////////////////////////////////
    // use: n_test_vertices, n_PMTs
    // cudamalloc: device_directions_phi, device_directions_cos_theta
    if( use_timing )
      start_cuda_clock();
    allocate_directions_memory_on_device();
    if( use_timing )
      elapsed_memory_directions_dev = stop_cuda_clock();
  }


  ////////////////////////////////
  // fill tofs memory on device //
  ////////////////////////////////
  // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
  // memcpy: device_times_of_flight, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
  // texture: tex_times_of_flight
  if( use_timing )
    start_cuda_clock();
  fill_tofs_memory_on_device();
  if( use_timing )
    elapsed_tofs_copy_dev = stop_cuda_clock();


  if( correct_mode == 9 ){
    ////////////////////////////////
    // fill directions memory on device //
    ////////////////////////////////
    // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
    // memcpy: device_directions_phi, device_directions_cos_theta, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
    // texture: tex_directions_phi, tex_directions_cos_theta
    if( use_timing )
      start_cuda_clock();
    fill_directions_memory_on_device();
    if( use_timing )
      elapsed_directions_copy_dev = stop_cuda_clock();
  }

  ///////////////////////
  // initialize output //
  ///////////////////////
  //  initialize_output();


  return 1;

}

int GPU_daq::test_vertices_execute(){

  start_total_cuda_clock();

  n_events = 0;

  while( set_input_file_for_event(n_events) ){

    printf(" [2] ------ analyzing event %d \n", n_events+1);

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
      write_output();
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


    if( correct_mode != 8 && correct_mode != 10 ){
      ////////////////////////////////////////////////
      // set number of blocks and threads per block //
      ////////////////////////////////////////////////
      // set: number_of_kernel_blocks, number_of_threads_per_block
      // use: n_test_vertices, n_hits
      if( use_timing )
	start_c_clock();
      if( !setup_threads_for_tof_2d(n_test_vertices, n_hits) ) return 0;
      if( use_timing )
	elapsed_threads += stop_c_clock();
    }


    ///////////////////////////////////////
    // allocate correct memory on device //
    ///////////////////////////////////////
    // use: n_test_vertices, n_hits, n_time_bins
    // cudamalloc: device_ids, device_times, device_n_pmts_per_time_bin
    if( use_timing )
      start_cuda_clock();
    allocate_correct_memory_on_device();
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
    if( correct_mode == 0 ){
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_n_pmts_per_time_bin<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
    }else if( correct_mode == 1 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      setup_threads_for_histo(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_one_thread_one_vertex<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 2 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      checkCudaErrors(cudaMemcpy(host_time_bin_of_hit,
				 device_time_bin_of_hit,
				 n_hits*n_test_vertices*sizeof(unsigned int),
				 cudaMemcpyDeviceToHost));

      for( unsigned int u=0; u<n_hits*n_test_vertices; u++){
	unsigned int bin = host_time_bin_of_hit[u];
	if( bin < n_time_bins*n_test_vertices )
	  host_n_pmts_per_time_bin[ bin ] ++;
      }

      checkCudaErrors(cudaMemcpy(device_n_pmts_per_time_bin,
				 host_n_pmts_per_time_bin,
				 n_time_bins*n_test_vertices*sizeof(unsigned int),
				 cudaMemcpyHostToDevice));
      

    }else if( correct_mode == 3 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo();
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_stride<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 4 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      unsigned int njobs = n_time_bins*n_test_vertices/max_n_threads_per_block + 1;
      printf(" [2] executing %d njobs to get n pmts per time bin \n", njobs); 
      for( unsigned int iter=0; iter<njobs; iter++){

	setup_threads_for_histo_iterated((bool)(iter + 1 == njobs));

	kernel_histo_iterated<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*n_test_vertices*sizeof(unsigned int) >>>(device_time_bin_of_hit, device_n_pmts_per_time_bin, iter*max_n_threads_per_block);
	cudaThreadSynchronize();
	getLastCudaError("kernel_histo execution failed\n");
      }

    }else if( correct_mode == 5 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      if( !setup_threads_for_tof_2d(n_test_vertices, n_time_bins) ) return 0;

      printf(" [2] executing kernel to get n pmts per time bin \n"); 

      kernel_histo_stride_2d<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d >>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo execution failed\n");

    }else if( correct_mode == 6 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_per_vertex<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 7 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 8 ){
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_correct_times_and_get_histo_per_vertex_shared execution failed\n");
    }else if( correct_mode == 9 ){
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_n_pmts_per_time_bin_and_direction_bin<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_per_time_bin_and_direction_bin, device_directions_for_vertex_and_pmt);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
    }else if( correct_mode == 10 ){
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_calculate_averages_and_get_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_n_pmts_per_time_bin,device_dx_per_time_bin,device_dy_per_time_bin,device_dz_per_time_bin,device_number_of_pmts_in_cone_in_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_correct_times_calculate_averages_and_get_histo_per_vertex_shared execution failed\n");


    }
    if( use_timing )
      elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin += stop_cuda_clock();



    //////////////////////////////////
    // setup threads for candidates //
    //////////////////////////////////
    // set: number_of_kernel_blocks, number_of_threads_per_block
    // use: n_time_bins
    if( use_timing )
      start_c_clock();
    if( !setup_threads_to_find_candidates() ) return 0;
    if( use_timing )
      elapsed_threads_candidates += stop_c_clock();



    //////////////////////////////////////////
    // allocate candidates memory on device //
    //////////////////////////////////////////
    // use: n_time_bins
    // cudamalloc: device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    allocate_candidates_memory_on_device();
    if( use_timing )
      elapsed_candidates_memory_dev += stop_cuda_clock();



    /////////////////////////////////////
    // find candidates above threshold //
    /////////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- execute candidates kernel \n");
    if( correct_mode == 9 ){
      kernel_find_vertex_with_max_npmts_in_timebin_and_directionbin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin_and_direction_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts);
    }else if( correct_mode == 10 ){
      kernel_find_vertex_with_max_npmts_and_center_of_mass_in_timebin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts,device_number_of_pmts_in_cone_in_time_bin,device_max_number_of_pmts_in_cone_in_time_bin);

    }else{
      kernel_find_vertex_with_max_npmts_in_timebin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts);
    }
    getLastCudaError("candidates_kernel execution failed\n");
    if( use_timing )
      elapsed_candidates_kernel += stop_cuda_clock();



    /////////////////////////////////////////
    // copy candidates from device to host //
    /////////////////////////////////////////
    // use: n_time_bins
    // memcpy: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- copy candidates from device to host \n");
    copy_candidates_from_device_to_host();
    if( use_timing )
      elapsed_candidates_copy_host += stop_cuda_clock();



    ///////////////////////////////////////
    // choose candidates above threshold //
    ///////////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- choose candidates above threshold \n");
    choose_candidates_above_threshold();
    if( use_timing )
      choose_candidates = stop_cuda_clock();



    ///////////////////////
    // coalesce triggers //
    ///////////////////////
    if( use_timing )
      start_cuda_clock();
    coalesce_triggers();
    if( use_timing )
      elapsed_coalesce += stop_cuda_clock();




    //////////////////////////////////
    // separate triggers into gates //
    //////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    separate_triggers_into_gates();
    if( use_timing )
      elapsed_gates += stop_cuda_clock();



    //////////////////
    // write output //
    //////////////////
    if( use_timing )
      start_cuda_clock();
    write_output();
    if( use_timing )
      elapsed_write_output += stop_cuda_clock();

    /////////////////////////////
    // deallocate event memory //
    /////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- deallocate memory \n");
    free_event_memories();
    if( use_timing )
      elapsed_free += stop_cuda_clock();

    n_events ++;

  }

  elapsed_total += stop_total_cuda_clock();


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

int GPU_daq::test_vertices_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts){

  start_total_cuda_clock();

  n_events = 0;

  while( 1 ){

    printf(" [2] ------ analyzing event %d \n", n_events+1);

    ////////////////
    // read input //
    ////////////////
    // set: n_hits, host_ids, host_times, time_offset, n_time_bins
    // use: time_offset, n_test_vertices
    // memcpy: constant_n_time_bins, constant_n_hits
    int earliest_time = 0;
    if( use_timing )
      start_c_clock();
    //    if( !read_the_input() ){
    if( !read_the_input_ToolDAQ(PMTid, time, &earliest_time) ){
      if( use_timing )
	elapsed_input += stop_c_clock();
      write_output();
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


    if( correct_mode != 8 && correct_mode != 10 ){
      ////////////////////////////////////////////////
      // set number of blocks and threads per block //
      ////////////////////////////////////////////////
      // set: number_of_kernel_blocks, number_of_threads_per_block
      // use: n_test_vertices, n_hits
      if( use_timing )
	start_c_clock();
      if( !setup_threads_for_tof_2d(n_test_vertices, n_hits) ) return 0;
      if( use_timing )
	elapsed_threads += stop_c_clock();
    }


    ///////////////////////////////////////
    // allocate correct memory on device //
    ///////////////////////////////////////
    // use: n_test_vertices, n_hits, n_time_bins
    // cudamalloc: device_ids, device_times, device_n_pmts_per_time_bin
    if( use_timing )
      start_cuda_clock();
    allocate_correct_memory_on_device();
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
    if( correct_mode == 0 ){
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_n_pmts_per_time_bin<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
    }else if( correct_mode == 1 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      setup_threads_for_histo(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_one_thread_one_vertex<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 2 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      checkCudaErrors(cudaMemcpy(host_time_bin_of_hit,
				 device_time_bin_of_hit,
				 n_hits*n_test_vertices*sizeof(unsigned int),
				 cudaMemcpyDeviceToHost));

      for( unsigned int u=0; u<n_hits*n_test_vertices; u++){
	unsigned int bin = host_time_bin_of_hit[u];
	if( bin < n_time_bins*n_test_vertices )
	  host_n_pmts_per_time_bin[ bin ] ++;
      }

      checkCudaErrors(cudaMemcpy(device_n_pmts_per_time_bin,
				 host_n_pmts_per_time_bin,
				 n_time_bins*n_test_vertices*sizeof(unsigned int),
				 cudaMemcpyHostToDevice));
      

    }else if( correct_mode == 3 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo();
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_stride<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 4 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      unsigned int njobs = n_time_bins*n_test_vertices/max_n_threads_per_block + 1;
      printf(" [2] executing %d njobs to get n pmts per time bin \n", njobs); 
      for( unsigned int iter=0; iter<njobs; iter++){

	setup_threads_for_histo_iterated((bool)(iter + 1 == njobs));

	kernel_histo_iterated<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*n_test_vertices*sizeof(unsigned int) >>>(device_time_bin_of_hit, device_n_pmts_per_time_bin, iter*max_n_threads_per_block);
	cudaThreadSynchronize();
	getLastCudaError("kernel_histo execution failed\n");
      }

    }else if( correct_mode == 5 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");

      if( !setup_threads_for_tof_2d(n_test_vertices, n_time_bins) ) return 0;

      printf(" [2] executing kernel to get n pmts per time bin \n"); 

      kernel_histo_stride_2d<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d >>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo execution failed\n");

    }else if( correct_mode == 6 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_per_vertex<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
    }else if( correct_mode == 7 ){
      printf(" [2] --- execute kernel to correct times \n");
      kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_time_bin_of_hit);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
      
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to get n pmts per time bin \n");
      kernel_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_time_bin_of_hit, device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_histo_one_thread_one_vertex execution failed\n");
      }else if( correct_mode == 8 ){
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_n_pmts_per_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_correct_times_and_get_histo_per_vertex_shared execution failed\n");
    }else if( correct_mode == 9 ){
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_and_get_n_pmts_per_time_bin_and_direction_bin<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_per_time_bin_and_direction_bin, device_directions_for_vertex_and_pmt);
      cudaThreadSynchronize();
      getLastCudaError("correct_kernel execution failed\n");
    }else if( correct_mode == 10 ){
      setup_threads_for_histo_per(n_test_vertices);
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      kernel_correct_times_calculate_averages_and_get_histo_per_vertex_shared<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d,n_time_bins*sizeof(unsigned int)>>>(device_n_pmts_per_time_bin,device_dx_per_time_bin,device_dy_per_time_bin,device_dz_per_time_bin,device_number_of_pmts_in_cone_in_time_bin);
      cudaThreadSynchronize();
      getLastCudaError("kernel_correct_times_calculate_averages_and_get_histo_per_vertex_shared execution failed\n");


      }
    if( use_timing )
      elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin += stop_cuda_clock();



    //////////////////////////////////
    // setup threads for candidates //
    //////////////////////////////////
    // set: number_of_kernel_blocks, number_of_threads_per_block
    // use: n_time_bins
    if( use_timing )
      start_c_clock();
    if( !setup_threads_to_find_candidates() ) return 0;
    if( use_timing )
      elapsed_threads_candidates += stop_c_clock();



    //////////////////////////////////////////
    // allocate candidates memory on device //
    //////////////////////////////////////////
    // use: n_time_bins
    // cudamalloc: device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    allocate_candidates_memory_on_device();
    if( use_timing )
      elapsed_candidates_memory_dev += stop_cuda_clock();



    /////////////////////////////////////
    // find candidates above threshold //
    /////////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- execute candidates kernel \n");
    if( correct_mode == 9 ){
      kernel_find_vertex_with_max_npmts_in_timebin_and_directionbin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin_and_direction_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts);
    }else if( correct_mode == 10 ){
      kernel_find_vertex_with_max_npmts_and_center_of_mass_in_timebin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts,device_number_of_pmts_in_cone_in_time_bin,device_max_number_of_pmts_in_cone_in_time_bin);

    }else{
      kernel_find_vertex_with_max_npmts_in_timebin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts);
    }
    getLastCudaError("candidates_kernel execution failed\n");
    if( use_timing )
      elapsed_candidates_kernel += stop_cuda_clock();



    /////////////////////////////////////////
    // copy candidates from device to host //
    /////////////////////////////////////////
    // use: n_time_bins
    // memcpy: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- copy candidates from device to host \n");
    copy_candidates_from_device_to_host();
    if( use_timing )
      elapsed_candidates_copy_host += stop_cuda_clock();



    ///////////////////////////////////////
    // choose candidates above threshold //
    ///////////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- choose candidates above threshold \n");
    choose_candidates_above_threshold();
    if( use_timing )
      choose_candidates = stop_cuda_clock();



    ///////////////////////
    // coalesce triggers //
    ///////////////////////
    if( use_timing )
      start_cuda_clock();
    coalesce_triggers();
    if( use_timing )
      elapsed_coalesce += stop_cuda_clock();




    //////////////////////////////////
    // separate triggers into gates //
    //////////////////////////////////
    if( use_timing )
      start_cuda_clock();
    separate_triggers_into_gates(trigger_ns, trigger_ts);
    if( use_timing )
      elapsed_gates += stop_cuda_clock();



    //////////////////
    // write output //
    //////////////////
    if( use_timing )
      start_cuda_clock();
    write_output();
    if( use_timing )
      elapsed_write_output += stop_cuda_clock();

    /////////////////////////////
    // deallocate event memory //
    /////////////////////////////
    if( use_timing )
      start_cuda_clock();
    if( use_verbose )
      printf(" [2] --- deallocate memory \n");
    free_event_memories();
    if( use_timing )
      elapsed_free += stop_cuda_clock();

    n_events ++;

    break;
  }

  elapsed_total += stop_total_cuda_clock();


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


int GPU_daq::test_vertices_finalize(){


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
    printf(" [2] make table of directions execution time : %f ms \n", elapsed_directions);
    printf(" [2] allocate tofs memory on device execution time : %f ms \n", elapsed_memory_tofs_dev);
    printf(" [2] allocate directions memory on device execution time : %f ms \n", elapsed_memory_directions_dev);
    printf(" [2] fill tofs memory on device execution time : %f ms \n", elapsed_tofs_copy_dev);
    printf(" [2] fill directions memory on device execution time : %f ms \n", elapsed_directions_copy_dev);
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

