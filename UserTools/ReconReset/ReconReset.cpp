#include "ReconReset.h"

ReconReset::ReconReset():Tool(){}


bool ReconReset::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("ReconReset") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool ReconReset::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  //reset the main holder of reconstructed information
  m_data->RecoInfo.Reset();

  //and all the filtered versions
  for (std::map<std::string, ReconInfo *>::iterator it=m_data->RecoInfoMap.begin();
       it!=m_data->RecoInfoMap.end();
       ++it) {
    ss << "DEBUG: Resetting RecoInfoMap name " << it->first;
    StreamToLog(DEBUG1);
    it->second->Reset();
  }

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool ReconReset::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
