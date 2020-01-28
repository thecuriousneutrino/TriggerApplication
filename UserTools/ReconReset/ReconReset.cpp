#include "ReconReset.h"

ReconReset::ReconReset():Tool(){}


bool ReconReset::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  return true;
}


bool ReconReset::Execute(){

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

  return true;
}


bool ReconReset::Finalise(){

  return true;
}
