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

  return true;
}


bool ReconReset::Finalise(){

  return true;
}
