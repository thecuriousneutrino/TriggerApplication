#include "nhits.h"

nhits::nhits():Tool(){}


bool nhits::Initialise(std::string configfile, DataModel &data){



  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

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
  //  GPU_daq::nhits_initialize();
  GPU_daq::nhits_initialize_ToolDAQ(PMTFile,DetectorFile,ParameterFile);
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



  return true;
}


bool nhits::Execute(){


  int the_output;

  //do stuff with m_data->Samples

  printf(" qqq data samples size %d \n", m_data->Samples.size());

  for( std::vector<SubSample>::const_iterator is=m_data->Samples.begin(); is!=m_data->Samples.end(); ++is){
#ifdef GPU   
  //  the_output =   GPU_daq::nhits_execute();
  the_output =   GPU_daq::nhits_execute(is->m_PMTid, is->m_time);
  printf(" qqq qqq look at %d of size %d \n", is - m_data->Samples.begin(), m_data->Samples.size());
#endif
  }

  //  the_output = CUDAFunction(m_data->Samples.at(0).m_PMTid, m_data->Samples.at(0).m_time);
  m_data->triggeroutput=(bool)the_output;


  return true;
}


bool nhits::Finalise(){


#ifdef GPU
  GPU_daq::nhits_finalize();
#endif


  return true;
}
