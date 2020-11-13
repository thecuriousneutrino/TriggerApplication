#include "test_vertices.h"

test_vertices::test_vertices():Tool(){}


bool test_vertices::Initialise(std::string configfile, DataModel &data){


  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);
  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("test_vertices") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  m_data->triggeroutput=false;



  
  std::string DetectorFile;
  std::string ParameterFile;
  

  m_variables.Get("DetectorFile",DetectorFile);
  m_variables.Get("ParameterFile",ParameterFile);
  
  m_variables.Get("distance_between_vertices",        m_distance_between_vertices);
  m_variables.Get("wall_like_distance",   m_wall_like_distance);
  m_variables.Get("water_like_threshold_number_of_pmts",   m_water_like_threshold_number_of_pmts);
  m_variables.Get("wall_like_threshold_number_of_pmts",   m_wall_like_threshold_number_of_pmts);
  m_variables.Get("coalesce_time",   m_coalesce_time);
  m_variables.Get("trigger_gate_up",   m_trigger_gate_up);
  m_variables.Get("trigger_gate_down",   m_trigger_gate_down);
  m_variables.Get("output_txt",   m_output_txt);
  m_variables.Get("correct_mode",   m_correct_mode);
  m_variables.Get("n_direction_bins_theta",   m_n_direction_bins_theta);
  m_variables.Get("cylindrical_grid",   m_cylindrical_grid);
  m_variables.Get("costheta_cone_cut",   m_costheta_cone_cut);
  m_variables.Get("select_based_on_cone",   m_select_based_on_cone);
  m_variables.Get("write_output_mode",   m_write_output_mode);
  m_variables.Get("trigger_threshold_adjust_for_noise",                   m_trigger_threshold_adjust_for_noise);
  m_variables.Get("max_n_hits_per_job",   m_max_n_hits_per_job);
  m_variables.Get("num_blocks_y",   m_num_blocks_y);
  m_variables.Get("num_threads_per_block_y",   m_num_threads_per_block_y);
  m_variables.Get("num_threads_per_block_x",   m_num_threads_per_block_x);

  m_ss << " DetectorFile " << DetectorFile.c_str(); StreamToLog(INFO);
  m_ss << " ParameterFile " << ParameterFile.c_str() ; StreamToLog(INFO);
  m_ss << " m_distance_between_vertices " << m_distance_between_vertices; StreamToLog(INFO);
  m_ss << " m_wall_like_distance " << m_wall_like_distance; StreamToLog(INFO);
  m_ss << " m_water_like_threshold_number_of_pmts " << m_water_like_threshold_number_of_pmts; StreamToLog(INFO);
  m_ss << " m_wall_like_threshold_number_of_pmts " << m_wall_like_threshold_number_of_pmts; StreamToLog(INFO);
  m_ss << " m_coalesce_time " << m_coalesce_time; StreamToLog(INFO);
  m_ss << " m_trigger_gate_up " << m_trigger_gate_up; StreamToLog(INFO);
  m_ss << " m_trigger_gate_down " << m_trigger_gate_down; StreamToLog(INFO);
  m_ss << " m_output_txt " << m_output_txt; StreamToLog(INFO);
  m_ss << " m_correct_mode " << m_correct_mode; StreamToLog(INFO);
  m_ss << " m_n_direction_bins_theta " <<   m_n_direction_bins_theta; StreamToLog(INFO);
  m_ss << " m_cylindrical_grid " <<   m_cylindrical_grid; StreamToLog(INFO);
  m_ss << " m_costheta_cone_cut " <<   m_costheta_cone_cut; StreamToLog(INFO);
  m_ss << " m_select_based_on_cone " <<   m_select_based_on_cone; StreamToLog(INFO);
  m_ss << " m_write_output_mode " << m_write_output_mode; StreamToLog(INFO);
  m_ss << " m_max_n_hits_per_job " << m_max_n_hits_per_job; StreamToLog(INFO);
  m_ss << " m_trigger_threshold_adjust_for_noise " << m_trigger_threshold_adjust_for_noise; StreamToLog(INFO);
  m_ss << " m_num_blocks_y " <<   m_num_blocks_y; StreamToLog(INFO);
  m_ss << " m_num_threads_per_block_y " <<   m_num_threads_per_block_y; StreamToLog(INFO);
  m_ss << " m_num_threads_per_block_x " <<   m_num_threads_per_block_x; StreamToLog(INFO);


  //  gpu_daq_initialize(PMTFile,DetectorFile,ParameterFile);

#ifdef GPU
  //  GPU_daq::test_vertices_initialize();


  std::vector<int> tube_no;
  std::vector<float> tube_x;
  std::vector<float> tube_y;
  std::vector<float> tube_z;
  for( std::vector<PMTInfo>::const_iterator ip=m_data->IDGeom.begin(); ip!=m_data->IDGeom.end(); ip++){
    tube_no.push_back(ip->m_tubeno);
    tube_x.push_back(ip->m_x);
    tube_y.push_back(ip->m_y);
    tube_z.push_back(ip->m_z);
  }

  

  GPU_daq::test_vertices_initialize_ToolDAQ(m_data->detector_length, m_data->detector_radius, m_data->pmt_radius, ParameterFile, tube_no, tube_x, tube_y, tube_z, m_data->IDPMTDarkRate*1000,
 m_distance_between_vertices,
 m_wall_like_distance,
 m_water_like_threshold_number_of_pmts,
 m_wall_like_threshold_number_of_pmts,
 m_coalesce_time,
 m_trigger_gate_up,
 m_trigger_gate_down,
 m_output_txt,
 m_correct_mode,
 m_n_direction_bins_theta,
 m_cylindrical_grid,
 m_costheta_cone_cut,
 m_select_based_on_cone,
 m_trigger_threshold_adjust_for_noise,
 m_max_n_hits_per_job,
 m_num_blocks_y,
 m_num_threads_per_block_y,
 m_num_threads_per_block_x,
 m_write_output_mode
);

#else

  CPU_test_vertices_initialize();

#endif

  int npmts = m_data->IDNPMTs;
  double dark_rate_kHZ = m_data->IDPMTDarkRate;
  double dark_rate_Hz = dark_rate_kHZ * 1000;
  double average_occupancy = dark_rate_Hz * m_coalesce_time * npmts;
  m_time_int.reserve(2*(int)average_occupancy);

  // can acess variables directly like this and would be good if you could impliment in your code

  float dark_rate;
  m_variables.Get("dark_rate",dark_rate);

  //to do this in your code instead of passing the three strings you could just do

  //gpu_daq_initialize(m_variables);

  // then in your code you can include 
  //#include "Store.h"
  //  gpu_daq_initialize(Store variables);

  //and pull them out with the get function in the same way 

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool test_vertices::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  std::vector<SubSample> & samples = m_data->IDSamples;

  for( std::vector<SubSample>::const_iterator is=samples.begin(); is!=samples.end(); ++is){

    std::vector<int> trigger_ns;
    std::vector<int> trigger_ts;
    //Copy the times from the `float` format in the DataModel to `int` format
    //This is not strictly required for the CPU version of the algorithm, but is done for consistency of results
    m_time_int.clear();
    for(unsigned int i = 0; i < is->m_time.size(); i++) {
      m_time_int.push_back(is->m_time[i]);
    }

#ifdef GPU   
    GPU_daq::test_vertices_execute(is->m_PMTid, m_time_int, &trigger_ns, &trigger_ts);
#else
    CPU_test_vertices_execute(is->m_PMTid, m_time_int, &trigger_ns, &trigger_ts);
#endif
    for(int i=0; i<trigger_ns.size(); i++){
      m_data->IDTriggers.AddTrigger(kTriggerUndefined,
				    TimeDelta(trigger_ts[i] + m_trigger_gate_down) + is->m_timestamp, 
				    TimeDelta(trigger_ts[i] + m_trigger_gate_up) + is->m_timestamp,
				    TimeDelta(trigger_ts[i] + m_trigger_gate_down) + is->m_timestamp, 
				    TimeDelta(trigger_ts[i] + m_trigger_gate_up) + is->m_timestamp,
				    TimeDelta(trigger_ts[i]) + is->m_timestamp,
				    std::vector<float>(1, trigger_ns[i]));

      m_ss << " trigger! time "<< trigger_ts[i] << " -> " << TimeDelta(trigger_ts[i] ) + is->m_timestamp << " nhits " <<  trigger_ns[i]; StreamToLog(INFO);
    }
  }


  if(m_stopwatch) m_stopwatch->Stop();

  return true;

}


bool test_vertices::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

#ifdef GPU
  GPU_daq::test_vertices_finalize();
#else
  CPU_test_vertices_finalize();
#endif

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}


int test_vertices::CPU_test_vertices_initialize(){

  use_verbose = false;

  n_PMTs = m_data->IDNPMTs;
  if( !n_PMTs ) return 0;
  PMT_x = (double *)malloc(n_PMTs*sizeof(double));
  PMT_y = (double *)malloc(n_PMTs*sizeof(double));
  PMT_z = (double *)malloc(n_PMTs*sizeof(double));
  int n_PMTs_counter = 0;
  for( std::vector<PMTInfo>::const_iterator ip=m_data->IDGeom.begin(); ip!=m_data->IDGeom.end(); ip++){
    PMT_x[n_PMTs_counter] = ip->m_x;
    PMT_y[n_PMTs_counter] = ip->m_y;
    PMT_z[n_PMTs_counter] = ip->m_z;
    n_PMTs_counter++;
  }
  printf(" [2] detector contains %d PMTs \n", n_PMTs);
  
  //read_user_parameters();
  {
    twopi = 2.*acos(-1.);
    speed_light_water = 29.9792/1.3330; // speed of light in water, cm/ns
  //double speed_light_water = 22.490023;

    cerenkov_costheta =1./1.3330;
    cerenkov_angle_water = acos(cerenkov_costheta);
    costheta_cone_cut = m_costheta_cone_cut;
    select_based_on_cone = m_select_based_on_cone;

    dark_rate = m_data->IDPMTDarkRate*1000; // Hz
    cylindrical_grid = m_cylindrical_grid;
    distance_between_vertices = m_distance_between_vertices; // cm
    wall_like_distance = m_wall_like_distance; // units of distance between vertices
    time_step_size = (unsigned int)(sqrt(3.)*distance_between_vertices/(4.*speed_light_water)); // ns
    int extra_threshold = 0;
    if( m_trigger_threshold_adjust_for_noise ){
      extra_threshold = (int)(dark_rate*n_PMTs*2.*time_step_size*1.e-9); // to account for dark current occupancy
    }
    water_like_threshold_number_of_pmts = m_water_like_threshold_number_of_pmts + extra_threshold;
    wall_like_threshold_number_of_pmts = m_wall_like_threshold_number_of_pmts + extra_threshold;
    coalesce_time = m_coalesce_time; // ns
    trigger_gate_up = m_trigger_gate_up; // ns
    trigger_gate_down = m_trigger_gate_down; // ns
    max_n_hits_per_job = m_max_n_hits_per_job;
    output_txt = m_output_txt;
    correct_mode = m_correct_mode;
    write_output_mode = m_write_output_mode;
    
    n_direction_bins_theta = m_n_direction_bins_theta;
    n_direction_bins_phi = 2*(n_direction_bins_theta - 1);
    n_direction_bins = n_direction_bins_phi*n_direction_bins_theta - 2*(n_direction_bins_phi - 1);
    
  }

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
  printf(" [2] cylindrical_grid %d \n", cylindrical_grid);
  printf(" [2] time step size = %d ns \n", time_step_size);
  printf(" [2] write_output_mode %d \n", write_output_mode);
  if( correct_mode == 9 ){
    printf(" [2] n_direction_bins_theta %d, n_direction_bins_phi %d, n_direction_bins %d \n",
	   n_direction_bins_theta, n_direction_bins_phi, n_direction_bins);
  }




  /////////////////////
  // read detector ////
  /////////////////////
  // set: detector_height, detector_radius, pmt_radius
  detector_height = m_data->detector_length;
  detector_radius = m_data->detector_radius;
  printf(" [2] detector height %f cm, radius %f cm \n", detector_height, detector_radius);




  ////////////////////////
  // make test vertices //
  ////////////////////////
  // set: n_test_vertices, n_water_like_test_vertices, vertex_x, vertex_y, vertex_z
  // use: detector_height, detector_radius
  make_test_vertices();



  //////////////////////////////
  // table of times_of_flight //
  //////////////////////////////
  // set: host_times_of_flight, time_offset
  // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
  // malloc: host_times_of_flight
  make_table_of_tofs();

  if( correct_mode == 9 ){
    //////////////////////////////
    // table of directions //
    //////////////////////////////
    // set: host_directions_phi, host_directions_cos_theta
    // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
    // malloc: host_directions_phi, host_directions_cos_theta
    make_table_of_directions();
  }


  return 1;

}

void test_vertices::make_test_vertices(){

  printf(" [2] --- make test vertices \n");
  float semiheight = detector_height/2.;
  n_test_vertices = 0;


  if( !cylindrical_grid ){

    // 1: count number of test vertices
    for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
      for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
	for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {
	  if(pow(j,2)+pow(k,2) > pow(detector_radius,2))
	    continue;
	  n_test_vertices++;
	}
      }
    }
    vertex_x = (double *)malloc(n_test_vertices*sizeof(double));
    vertex_y = (double *)malloc(n_test_vertices*sizeof(double));
    vertex_z = (double *)malloc(n_test_vertices*sizeof(double));

    // 2: assign coordinates to test vertices
    // water-like events
    n_test_vertices = 0;
    for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
      for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
	for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {

	
	  if( 
	     // skip endcap region
	     abs(i) > semiheight - wall_like_distance*distance_between_vertices ||
	     // skip sidewall region
	     pow(j,2)+pow(k,2) > pow(detector_radius - wall_like_distance*distance_between_vertices,2)
	      ) continue;
	
	  vertex_x[n_test_vertices] = j*1.;
	  vertex_y[n_test_vertices] = k*1.;
	  vertex_z[n_test_vertices] = i*1.;
	  n_test_vertices++;
	}
      }
    }
    n_water_like_test_vertices = n_test_vertices;

    // wall-like events
    for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
      for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
	for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {

	  if( 
	     abs(i) > semiheight - wall_like_distance*distance_between_vertices ||
	     pow(j,2)+pow(k,2) > pow(detector_radius - wall_like_distance*distance_between_vertices,2)
	      ){

	    if(pow(j,2)+pow(k,2) > pow(detector_radius,2)) continue;
	  
	    vertex_x[n_test_vertices] = j*1.;
	    vertex_y[n_test_vertices] = k*1.;
	    vertex_z[n_test_vertices] = i*1.;
	    n_test_vertices++;
	  }
	}
      }
    }

  }else{ // cylindrical grid
  
    int n_vertical = detector_height/distance_between_vertices;
    double distance_vertical = detector_height/n_vertical;
    int n_radial = 2.*detector_radius/distance_between_vertices;
    double distance_radial = 2.*detector_radius/n_radial;
    int n_angular;
    double distance_angular;
    
    printf(" [2] distance_between_vertices %f, distance_vertical %f, distance_radial %f \n",
	   distance_between_vertices, distance_vertical, distance_radial);
    
    double the_r, the_z, the_phi;
    bool first = false; // true: add extra layer near wall
                       // false: regular spacing

    bool add_extra_layer = first;
    
    // 1: count number of test vertices
    the_r = detector_radius;
    while( the_r >= 0. ){
      n_angular = twopi*the_r / distance_between_vertices;
      distance_angular = twopi/n_angular;
      
      the_z = -semiheight;
      
      while( the_z <= semiheight){
	
	the_phi = 0.;
	while( the_phi < twopi - distance_angular/2. ){
	  
	  n_test_vertices ++;
	  
	  if( the_r == 0. ) break;
	  the_phi += distance_angular;
	}


	if( add_extra_layer ){
	  if( the_z + semiheight < 0.3*distance_vertical ) // only true at bottom endcap
	    the_z += distance_vertical/2.;
	  else if( semiheight - the_z < 0.7*distance_vertical ) // only true near top endcap
	    the_z += distance_vertical/2.;
	  else
	    the_z += distance_vertical;
	}else{
	  the_z += distance_vertical;
	}
      }
      if( first ){
	the_r -= distance_radial/2.;
	first = false;
      }
      else{
	the_r -= distance_radial;
      }
    }

    vertex_x = (double *)malloc(n_test_vertices*sizeof(double));
    vertex_y = (double *)malloc(n_test_vertices*sizeof(double));
    vertex_z = (double *)malloc(n_test_vertices*sizeof(double));

    first = add_extra_layer;
    // 2: assign coordinates to test vertices
    // water-like events
    n_test_vertices = 0;

    the_r = detector_radius;
    while( the_r >= 0. ){

      // skip sidewall region
      if(the_r <= detector_radius - wall_like_distance*distance_between_vertices ){

	n_angular = twopi*the_r / distance_between_vertices;
	distance_angular = twopi/n_angular;
	
	the_z = -semiheight;
	
	while( the_z <= semiheight){
	  
	  // skip endcap region
	  if( fabs(the_z) <= semiheight - wall_like_distance*distance_between_vertices ){

	    the_phi = 0.;
	    while( the_phi < twopi - distance_angular/2. ){
	      
	      vertex_x[n_test_vertices] = the_r*cos(the_phi);
	      vertex_y[n_test_vertices] = the_r*sin(the_phi);
	      vertex_z[n_test_vertices] = the_z;
	      n_test_vertices ++;
	      
	      if( the_r == 0. ) break;
	      the_phi += distance_angular;
	    }
	  }

	  if( add_extra_layer ){
	    if( the_z + semiheight < 0.3*distance_vertical ) // only true at bottom endcap
	      the_z += distance_vertical/2.;
	    else if( semiheight - the_z < 0.7*distance_vertical ) // only true near top endcap
	      the_z += distance_vertical/2.;
	    else
	      the_z += distance_vertical;
	  }else{
	    the_z += distance_vertical;
	  }
	}

      }
      if( first ){
	the_r -= distance_radial/2.;
	first = false;
      }
      else{
	the_r -= distance_radial;
      }
    }


    n_water_like_test_vertices = n_test_vertices;

    first = add_extra_layer;
    // wall-like events
    the_r = detector_radius;
    while( the_r >= 0. ){

      n_angular = twopi*the_r / distance_between_vertices;
      distance_angular = twopi/n_angular;
      
      the_z = -semiheight;
      
      while( the_z <= semiheight){
	
	if( fabs(the_z) > semiheight - wall_like_distance*distance_between_vertices ||
	    the_r > detector_radius - wall_like_distance*distance_between_vertices ){
	  
	  the_phi = 0.;
	  while( the_phi < twopi - distance_angular/2. ){
	    
	    vertex_x[n_test_vertices] = the_r*cos(the_phi);
	    vertex_y[n_test_vertices] = the_r*sin(the_phi);
	    vertex_z[n_test_vertices] = the_z;
	    n_test_vertices ++;
	    
	    if( the_r == 0. ) break;
	    the_phi += distance_angular;
	  }
	}
	if( add_extra_layer ){
	  if( the_z + semiheight < 0.3*distance_vertical ) // only true at bottom endcap
	    the_z += distance_vertical/2.;
	  else if( semiheight - the_z < 0.7*distance_vertical ) // only true near top endcap
	    the_z += distance_vertical/2.;
	  else
	    the_z += distance_vertical;
	}else{
	  the_z += distance_vertical;
	}
      }
      if( first ){
	the_r -= distance_radial/2.;
	first = false;
      }
      else{
	the_r -= distance_radial;
      }
    }
    

  }

  printf(" [2] made %d test vertices \n", n_test_vertices);

  return;

}


void test_vertices::make_table_of_tofs(){

  printf(" [2] --- fill times_of_flight \n");
  host_times_of_flight = (time_of_flight_t*)malloc(n_test_vertices*n_PMTs * sizeof(time_of_flight_t));
  printf(" [2] speed_light_water %f \n", speed_light_water);
  if( correct_mode == 10 ){
    host_light_dx = (float*)malloc(n_test_vertices*n_PMTs * sizeof(double));
    host_light_dy = (float*)malloc(n_test_vertices*n_PMTs * sizeof(double));
    host_light_dz = (float*)malloc(n_test_vertices*n_PMTs * sizeof(double));
    host_light_dr = (float*)malloc(n_test_vertices*n_PMTs * sizeof(double));
  }
  unsigned int distance_index;
  time_offset = 0.;
  for(unsigned int ip=0; ip<n_PMTs; ip++){
    for(unsigned int iv=0; iv<n_test_vertices; iv++){
      distance_index = get_distance_index(ip + 1, n_PMTs*iv);
      host_times_of_flight[distance_index] = sqrt(pow(vertex_x[iv] - PMT_x[ip],2) + pow(vertex_y[iv] - PMT_y[ip],2) + pow(vertex_z[iv] - PMT_z[ip],2))/speed_light_water;
      if( correct_mode == 10 ){
	host_light_dx[distance_index] = PMT_x[ip] - vertex_x[iv];
	host_light_dy[distance_index] = PMT_y[ip] - vertex_y[iv];
	host_light_dz[distance_index] = PMT_z[ip] - vertex_z[iv];
	host_light_dr[distance_index] = sqrt(pow(host_light_dx[distance_index],2) + pow(host_light_dy[distance_index],2) + pow(host_light_dz[distance_index],2));
      }
      if( host_times_of_flight[distance_index] > time_offset )
	time_offset = host_times_of_flight[distance_index];

    }
  }
  //print_times_of_flight();

  return;
}


void test_vertices::make_table_of_directions(){

  printf(" [2] --- fill directions \n");
  printf(" [2] cerenkov_angle_water %f \n", cerenkov_angle_water);
  host_directions_for_vertex_and_pmt = (bool*)malloc(n_test_vertices*n_PMTs*n_direction_bins * sizeof(bool));
  float dx, dy, dz, dr, phi, cos_theta, sin_theta;
  float phi2, cos_theta2, angle;
  unsigned int dir_index_at_angles;
  unsigned int dir_index_at_pmt;
  for(unsigned int ip=0; ip<n_PMTs; ip++){
    for(unsigned int iv=0; iv<n_test_vertices; iv++){
      dx = PMT_x[ip] - vertex_x[iv];
      dy = PMT_y[ip] - vertex_y[iv];
      dz = PMT_z[ip] - vertex_z[iv];
      dr = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
      phi = atan2(dy,dx);
      // light direction
      cos_theta = dz/dr;
      sin_theta = sqrt(1. - pow(cos_theta,2));
      // particle direction
      for(unsigned int itheta = 0; itheta < n_direction_bins_theta; itheta++){
	cos_theta2 = -1. + 2.*itheta/(n_direction_bins_theta - 1);
	for(unsigned int iphi = 0; iphi < n_direction_bins_phi; iphi++){
	  phi2 = 0. + twopi*iphi/n_direction_bins_phi;

	  if( (itheta == 0 || itheta + 1 == n_direction_bins_theta ) && iphi != 0 ) break;

	  // angle between light direction and particle direction
	  angle = acos( sin_theta*sqrt(1 - pow(cos_theta2,2)) * cos(phi - phi2) + cos_theta*cos_theta2 );

	  dir_index_at_angles = get_direction_index_at_angles(iphi, itheta);
	  dir_index_at_pmt = get_direction_index_at_pmt(ip, iv, dir_index_at_angles);

	  //printf(" [2] phi %f ctheta %f phi' %f ctheta' %f angle %f dir_index_at_angles %d dir_index_at_pmt %d \n", 
	  //	 phi, cos_theta, phi2, cos_theta2, angle, dir_index_at_angles, dir_index_at_pmt);

	  host_directions_for_vertex_and_pmt[dir_index_at_pmt] 
	    = (bool)(fabs(angle - cerenkov_angle_water) < twopi/(2.*n_direction_bins_phi));
	}
      }
    }
  }
  //print_directions();

  return;
}




unsigned int test_vertices::get_distance_index(unsigned int pmt_id, unsigned int vertex_block){
  // block = (npmts) * (vertex index)

  return pmt_id - 1 + vertex_block;

}

unsigned int test_vertices::get_time_index(unsigned int hit_index, unsigned int vertex_block){
  // block = (n time bins) * (vertex index)

  return hit_index + vertex_block;

}

 unsigned int test_vertices::get_direction_index_at_angles(unsigned int iphi, unsigned int itheta){

   if( itheta == 0 ) return 0;
   if( itheta + 1 == n_direction_bins_theta ) return n_direction_bins - 1;

   return 1 + (itheta - 1) * n_direction_bins_phi + iphi;

}

unsigned int test_vertices::get_direction_index_at_pmt(unsigned int pmt_id, unsigned int vertex_index, unsigned int direction_index){

  //                                                     pmt id 1                        ...        pmt id p
  // [                      vertex 1                              vertex 2 ... vertex m] ... [vertex 1 ... vertex m]
  // [(dir 1 ... dir n) (dir 1 ... dir n) ... (dir 1 ... dir n)] ...

  return n_direction_bins * (pmt_id * n_test_vertices  + vertex_index) + direction_index ;

}

unsigned int test_vertices::get_direction_index_at_time(unsigned int time_bin, unsigned int vertex_index, unsigned int direction_index){

  //                                                     time 1                        ...        time p
  // [                      vertex 1                              vertex 2 ... vertex m] ... [vertex 1 ... vertex m]
  // [(dir 1 ... dir n) (dir 1 ... dir n) ... (dir 1 ... dir n)] ...

  return n_direction_bins* (time_bin * n_test_vertices  + vertex_index ) + direction_index ;

}


int test_vertices::CPU_test_vertices_finalize(){


  //////////////////////////////
  // deallocate global memory //
  //////////////////////////////
  if( use_verbose )
    printf(" [2] --- deallocate tofs memory \n");
  free_global_memories();



  return 1;

}

void test_vertices::free_global_memories(){

  if( correct_mode == 9 ){
    free(host_directions_for_vertex_and_pmt);
  }

  free(PMT_x);
  free(PMT_y);
  free(PMT_z);
  free(vertex_x);
  free(vertex_y);
  free(vertex_z);
  free(host_times_of_flight);
  if( correct_mode == 10 ){
    free(host_light_dx);
    free(host_light_dy);
    free(host_light_dz);
    free(host_light_dr);
  }

  return;
}



int test_vertices::CPU_test_vertices_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts){

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
    //    if( !read_the_input() ){
    if( !read_the_input_ToolDAQ(PMTid, time, &earliest_time) ){
      write_output();
      n_events ++;
      continue;
    }
  


    ////////////////////////////////////////
    // allocate candidates memory on host //
    ////////////////////////////////////////
    // use: n_time_bins
    // malloc: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
    allocate_candidates_memory_on_host();



    ////////////////////
    // execute kernel //
    ////////////////////
    if( correct_mode == 8 ){
      printf(" [2] --- execute kernel to correct times and get n pmts per time bin \n");
      correct_times_and_get_histo_per_vertex_shared(host_n_pmts_per_time_bin);
    }





    /////////////////////////////////////
    // find candidates above threshold //
    /////////////////////////////////////
    if( use_verbose )
      printf(" [2] --- execute candidates kernel \n");
    find_vertex_with_max_npmts_in_timebin(host_n_pmts_per_time_bin, host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts);



    ///////////////////////////////////////
    // choose candidates above threshold //
    ///////////////////////////////////////
    if( use_verbose )
      printf(" [2] --- choose candidates above threshold \n");
    choose_candidates_above_threshold();



    ///////////////////////
    // coalesce triggers //
    ///////////////////////
    coalesce_triggers();




    //////////////////////////////////
    // separate triggers into gates //
    //////////////////////////////////
    separate_triggers_into_gates(trigger_ns, trigger_ts);



    //////////////////
    // write output //
    //////////////////
    write_output();

    /////////////////////////////
    // deallocate event memory //
    /////////////////////////////
    if( use_verbose )
      printf(" [2] --- deallocate memory \n");
    free_event_memories();

    n_events ++;

    break;
  }

  printf(" [2] ------ analyzed %d events \n", n_events);


  return 1;
}


bool test_vertices::read_the_input_ToolDAQ(std::vector<int> PMTids, std::vector<int> times, int * earliest_time){

  printf(" [2] --- read input \n");
  n_hits = PMTids.size();
  if( !n_hits ) return false;
  if( n_hits != times.size() ){
    printf(" [2] n PMT ids %d but n times %d \n", n_hits, times.size());
    return false;
  }
  printf(" [2] input contains %d hits \n", n_hits);
  host_ids = (unsigned int *)malloc(n_hits*sizeof(unsigned int));
  host_times = (unsigned int *)malloc(n_hits*sizeof(unsigned int));

  //copy all hit times from `int` to `unsigned int` arrays
  // `unsigned int` operations are faster on GPU (done here for consistency)
  //  if( !read_input() ) return false;
  // read_input()
  {
    int min = INT_MAX;
    int max = INT_MIN;
    int time;
    for(int i=0; i<PMTids.size(); i++){
      time = int(floor(times[i]));
      host_times[i] = time;
      host_ids[i] = PMTids[i];
      //      printf(" [2] input %d PMT %d time %d \n", i, host_ids[i], host_times[i]);
      if( time > max ) max = time;
      if( time < min ) min = time;
    }
    //ensure there are no negatively underflowed `unsigned int` times
    if( min < 0 ){
      for(int i=0; i<PMTids.size(); i++){
	host_times[i] -= min;
      }
      max -= min;
      min -= min;
    }
    the_max_time = max;
    *earliest_time = min - min % time_step_size;
  }


  //time_offset = 600.; // set to constant to match trevor running
  n_time_bins = int(floor((the_max_time + time_offset)/time_step_size))+1; // floor returns the integer below
  printf(" [2] input max_time %d, n_time_bins %d \n", the_max_time, n_time_bins);
  printf(" [2] time_offset = %d ns \n", time_offset);
  //print_input();

  return true;
}

void test_vertices::write_output(){

  if( output_txt ){
    FILE *of=fopen(output_file.c_str(), "a");

    int trigger;
    if( write_output_mode == 0 ){
      // output 1 if there is a trigger, 0 otherwise
      trigger = (trigger_pair_vertex_time.size() > 0 ? 1 : 0);
    }

    if( write_output_mode == 1 ){
      // output the n of triggers
      trigger = trigger_pair_vertex_time.size();
    }

    if( write_output_mode == 2 ){
      // output the n of water-like triggers
      int trigger = 0;
      for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger){
        if( itrigger->first  < n_water_like_test_vertices )
      	trigger ++;
      }
    }

    if( write_output_mode == 0 || write_output_mode == 1 || write_output_mode == 2 ){
      fprintf(of, " %d \n", trigger);
    }

    if( write_output_mode == 3 ){
      unsigned int triggertime, trigger_index;
      // output reconstructed vertices
      for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger){
	triggertime = itrigger->second*time_step_size - time_offset;
	if( correct_mode == 10 ){
	  trigger_index = itrigger - trigger_pair_vertex_time.begin();
	  fprintf(of, " %d %f %f %f %d %d %d \n", n_events, vertex_x[itrigger->first], vertex_y[itrigger->first], vertex_z[itrigger->first], triggertime, trigger_npmts_in_time_bin.at(trigger_index), trigger_npmts_in_cone_in_time_bin.at(trigger_index));
	}else{
	  fprintf(of, " %d %f %f %f %d \n", n_events, vertex_x[itrigger->first], vertex_y[itrigger->first], vertex_z[itrigger->first], triggertime);
	}
      }
    }

    if( write_output_mode == 4 ){
      // output non-corrected and corrected times for best vertex
      int max_n_pmts = 0;
      unsigned int best_vertex;
      for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger){
	unsigned int vertex_index = itrigger->first;
	unsigned int time_index = itrigger->second;
	unsigned int local_n_pmts = host_max_number_of_pmts_in_time_bin[itrigger->second];
	if( local_n_pmts > max_n_pmts ){
	  max_n_pmts = local_n_pmts;
	  best_vertex = vertex_index;
	}
      }
      unsigned int distance_index;
      double tof;
      double corrected_time;
      
      for(unsigned int i=0; i<n_hits; i++){
	
	distance_index = get_distance_index(host_ids[i], n_PMTs*best_vertex);
	tof = host_times_of_flight[distance_index];
	corrected_time = host_times[i]-tof;
	
	fprintf(of, " %d %d %f \n", host_ids[i], host_times[i], corrected_time);
	//fprintf(of, " %d %f \n", host_ids[i], corrected_time);
      }
    }
    
    fclose(of);
  }


}

void test_vertices::allocate_candidates_memory_on_host(){

  printf(" [2] --- allocate candidates memory on host \n");

  host_max_number_of_pmts_in_time_bin = (histogram_t *)malloc(n_time_bins*sizeof(histogram_t));
  host_vertex_with_max_n_pmts = (unsigned int *)malloc(n_time_bins*sizeof(unsigned int));
    host_n_pmts_per_time_bin = (unsigned int *)malloc(n_time_bins*n_test_vertices*sizeof(unsigned int));

  if( correct_mode == 10 ){
    host_max_number_of_pmts_in_cone_in_time_bin = (unsigned int *)malloc(n_time_bins*sizeof(unsigned int));
  }

  return;

}


void test_vertices::correct_times_and_get_histo_per_vertex_shared(histogram_t *ct){

  unsigned int distance_index;
  double tof;
  double corrected_time;
  unsigned int bin;
  
  for(unsigned int i=0; i<n_hits; i++){
    for(unsigned int iv=0; iv<n_test_vertices; iv++){
    
      distance_index = get_distance_index(host_ids[i], n_PMTs*iv);
      tof = host_times_of_flight[distance_index];
      corrected_time = host_times[i]-tof + time_offset;
    
      bin = corrected_time/time_step_size + n_time_bins*iv;

      ct[bin] ++;
    }
  }
  

}


void test_vertices::find_vertex_with_max_npmts_in_timebin(histogram_t * np, histogram_t * mnp, unsigned int * vmnp){

  for(unsigned int time_bin_index=0; time_bin_index<n_time_bins; time_bin_index++){

    unsigned int number_of_pmts_in_time_bin = 0;
    unsigned int time_index;
    histogram_t max_number_of_pmts_in_time_bin=0;
    unsigned int vertex_with_max_n_pmts = 0;
    
    for(unsigned int iv=0;iv<n_test_vertices;iv++) { // loop over test vertices
      // sum the number of hit PMTs in this time window and the next
    
      time_index = time_bin_index + n_time_bins*iv;
      if( time_index >= n_time_bins*n_test_vertices - 1 ) continue;
      number_of_pmts_in_time_bin = np[time_index] + np[time_index+1];
      if( number_of_pmts_in_time_bin >= max_number_of_pmts_in_time_bin ){
	max_number_of_pmts_in_time_bin = number_of_pmts_in_time_bin;
	vertex_with_max_n_pmts = iv;
      }
    }

    mnp[time_bin_index] = max_number_of_pmts_in_time_bin;
    vmnp[time_bin_index] = vertex_with_max_n_pmts;
  }

  return;

}

void test_vertices::choose_candidates_above_threshold(){

  candidate_trigger_pair_vertex_time.clear();
  candidate_trigger_npmts_in_time_bin.clear();
  if( correct_mode == 10 ){
    candidate_trigger_npmts_in_cone_in_time_bin.clear();
  }

  unsigned int the_threshold;
  unsigned int number_of_pmts_to_cut_on;

  for(unsigned int time_bin = 0; time_bin<n_time_bins - 1; time_bin++){ // loop over time bins
    // n_time_bins - 1 as we are checking the i and i+1 at the same time
    
    if(host_vertex_with_max_n_pmts[time_bin] < n_water_like_test_vertices )
      the_threshold = water_like_threshold_number_of_pmts;
    else
      the_threshold = wall_like_threshold_number_of_pmts;

    number_of_pmts_to_cut_on = host_max_number_of_pmts_in_time_bin[time_bin];
    if( correct_mode == 10 ){
      if( select_based_on_cone ){
	number_of_pmts_to_cut_on = host_max_number_of_pmts_in_cone_in_time_bin[time_bin];
      }
    }

    if(number_of_pmts_to_cut_on > the_threshold) {

      if( use_verbose ){
	printf(" [2] time %f vertex (%f, %f, %f) npmts %d \n", (time_bin + 2)*time_step_size - time_offset, vertex_x[host_vertex_with_max_n_pmts[time_bin]], vertex_y[host_vertex_with_max_n_pmts[time_bin]], vertex_z[host_vertex_with_max_n_pmts[time_bin]], number_of_pmts_to_cut_on);
      }

      candidate_trigger_pair_vertex_time.push_back(std::make_pair(host_vertex_with_max_n_pmts[time_bin],time_bin+1));
      candidate_trigger_npmts_in_time_bin.push_back(host_max_number_of_pmts_in_time_bin[time_bin]);
      if( correct_mode == 10 ){
	candidate_trigger_npmts_in_cone_in_time_bin.push_back(host_max_number_of_pmts_in_cone_in_time_bin[time_bin]);
      }
    }

  }

  if( use_verbose )
    printf(" [2] n candidates: %d \n", candidate_trigger_pair_vertex_time.size());
}



void test_vertices::coalesce_triggers(){

  trigger_pair_vertex_time.clear();
  trigger_npmts_in_time_bin.clear();
  if( correct_mode == 10 ){
    trigger_npmts_in_cone_in_time_bin.clear();
  }

  unsigned int vertex_index, time_upper, number_of_pmts_in_time_bin, number_of_pmts_in_cone_in_time_bin;
  unsigned int max_pmt=0,max_vertex_index=0,max_time=0,max_pmt_in_cone=0;
  bool first_trigger, last_trigger, coalesce_triggers;
  unsigned int trigger_index;
  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=candidate_trigger_pair_vertex_time.begin(); itrigger != candidate_trigger_pair_vertex_time.end(); ++itrigger){

    vertex_index =      itrigger->first;
    time_upper = itrigger->second;
    trigger_index = itrigger - candidate_trigger_pair_vertex_time.begin();
    number_of_pmts_in_time_bin = candidate_trigger_npmts_in_time_bin.at(trigger_index);
    if( correct_mode == 10 ){
      number_of_pmts_in_cone_in_time_bin = candidate_trigger_npmts_in_cone_in_time_bin.at(trigger_index);
    }

    first_trigger = (trigger_index == 0);
    last_trigger = (trigger_index == candidate_trigger_pair_vertex_time.size()-1);
       
    if( first_trigger ){
      max_pmt = number_of_pmts_in_time_bin;
      max_vertex_index = vertex_index;
      max_time = time_upper;
      if( correct_mode == 10 ){
	max_pmt_in_cone = number_of_pmts_in_cone_in_time_bin;
      }
    }
    else{
      coalesce_triggers = (std::abs((int)(max_time - time_upper)) < coalesce_time/time_step_size);

      if( coalesce_triggers ){
	if( number_of_pmts_in_time_bin >= max_pmt) {
	  max_pmt = number_of_pmts_in_time_bin;
	  max_vertex_index = vertex_index;
	  max_time = time_upper;
	  if( correct_mode == 10 ){
	    max_pmt_in_cone = number_of_pmts_in_cone_in_time_bin;
	  }
	}
      }else{
	trigger_pair_vertex_time.push_back(std::make_pair(max_vertex_index,max_time));
	trigger_npmts_in_time_bin.push_back(max_pmt);
	max_pmt = number_of_pmts_in_time_bin;
	max_vertex_index = vertex_index;
	max_time = time_upper;     
	if( correct_mode == 10 ){
	  trigger_npmts_in_cone_in_time_bin.push_back(max_pmt_in_cone);
	  max_pmt_in_cone = number_of_pmts_in_cone_in_time_bin;
	}
      }
    }

    if(last_trigger){
      trigger_pair_vertex_time.push_back(std::make_pair(max_vertex_index,max_time));
      trigger_npmts_in_time_bin.push_back(max_pmt);
      if( correct_mode == 10 ){
	trigger_npmts_in_cone_in_time_bin.push_back(max_pmt_in_cone);
      }
    }
     
  }

  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger)
    printf(" [2] coalesced trigger timebin %d vertex index %d \n", itrigger->first, itrigger->second);

  return;

}


void test_vertices::separate_triggers_into_gates(std::vector<int> * trigger_ns, std::vector<int> * trigger_ts){

  final_trigger_pair_vertex_time.clear();
  unsigned int trigger_index;

  unsigned int time_start=0;
  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger){
    //once a trigger is found, we must jump in the future before searching for the next
    if(itrigger->second > time_start) {
      unsigned int triggertime = itrigger->second*time_step_size - time_offset;
      final_trigger_pair_vertex_time.push_back(std::make_pair(itrigger->first,triggertime));
      time_start = triggertime + trigger_gate_up;
      trigger_index = itrigger - trigger_pair_vertex_time.begin();
      output_trigger_information.clear();
      output_trigger_information.push_back(vertex_x[itrigger->first]);
      output_trigger_information.push_back(vertex_y[itrigger->first]);
      output_trigger_information.push_back(vertex_z[itrigger->first]);
      output_trigger_information.push_back(trigger_npmts_in_time_bin.at(trigger_index));
      output_trigger_information.push_back(triggertime);

      trigger_ns->push_back(trigger_npmts_in_time_bin.at(trigger_index));
      trigger_ts->push_back(triggertime);

      printf(" [2] triggertime: %d, npmts: %d, x: %f, y: %f, z: %f \n", triggertime, trigger_npmts_in_time_bin.at(trigger_index), vertex_x[itrigger->first], vertex_y[itrigger->first], vertex_z[itrigger->first]);

      /* if( output_txt ){ */
      /* 	FILE *of=fopen(output_file.c_str(), "w"); */

      /* 	unsigned int distance_index; */
      /* 	double tof; */
      /* 	double corrected_time; */

      /* 	for(unsigned int i=0; i<n_hits; i++){ */

      /* 	  distance_index = get_distance_index(host_ids[i], n_PMTs*(itrigger->first)); */
      /* 	  tof = host_times_of_flight[distance_index]; */

      /* 	  corrected_time = host_times[i]-tof; */

      /* 	  //fprintf(of, " %d %d %f \n", host_ids[i], host_times[i], corrected_time); */
      /* 	  fprintf(of, " %d %f \n", host_ids[i], corrected_time); */
      /* 	} */

      /* 	fclose(of); */
      /* } */

    }
  }


  return;
}


void test_vertices::free_event_memories(){

  free(host_ids);
  free(host_times);
  free(host_max_number_of_pmts_in_time_bin);
  free(host_vertex_with_max_n_pmts);
  if( correct_mode == 10 ){
    free(host_max_number_of_pmts_in_cone_in_time_bin);
  }

  return;
}
