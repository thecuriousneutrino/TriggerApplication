#include "test_vertices.h"

test_vertices::test_vertices():Tool(){}


bool test_vertices::Initialise(std::string configfile, DataModel &data){


  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("test_vertices") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  m_data->triggeroutput=false;



  
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;
  

  m_variables.Get("PMTFile",PMTFile);
  m_variables.Get("DetectorFile",DetectorFile);
  m_variables.Get("ParameterFile",ParameterFile);
  
  //  gpu_daq_initialize(PMTFile,DetectorFile,ParameterFile);

#ifdef GPU
  GPU_daq::test_vertices_initialize();
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

  //do stuff with m_data->Samples
  /*
  for( std::vector<SubSample>::const_iterator is=m_data->Samples.begin(); is!=m_data->Samples.end(); ++is){
    PMTids.push_back(is->m_PMTid);
    times.push_back(is->m_time);
  }
  */
  int the_output;
  //  the_output = CUDAFunction(m_data->Samples.at(0).m_PMTid, m_data->Samples.at(0).m_time);
#ifdef GPU   
  the_output =   GPU_daq::test_vertices_execute();
#endif
  m_data->triggeroutput=(bool)the_output;

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
