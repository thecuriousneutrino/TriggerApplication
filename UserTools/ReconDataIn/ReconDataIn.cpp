#include "ReconDataIn.h"

ReconDataIn::ReconDataIn():Tool(){}


bool ReconDataIn::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  return true;
}


bool ReconDataIn::Execute(){

  return true;
}


bool ReconDataIn::Finalise(){

  return true;
}
