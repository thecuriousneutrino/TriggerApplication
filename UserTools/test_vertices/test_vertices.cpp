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

  int npmts = m_data->IDNPMTs;
  double dark_rate_kHZ = m_data->IDPMTDarkRate;
  double dark_rate_Hz = dark_rate_kHZ * 1000;
  double average_occupancy = dark_rate_Hz * m_coalesce_time * npmts;
  m_time_int.reserve(2*(int)average_occupancy);

#endif

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
#ifdef GPU   

    std::vector<int> trigger_ns;
    std::vector<int> trigger_ts;
    m_time_int.clear();
    for(unsigned int i = 0; i < is->m_time.size(); i++) {
      m_time_int.push_back(is->m_time[i]);
    }
    GPU_daq::test_vertices_execute(is->m_PMTid, m_time_int, &trigger_ns, &trigger_ts);
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
#else
    ;
#endif
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
#endif

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
