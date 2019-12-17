#include "ReconFilter.h"

ReconFilter::ReconFilter():Tool(){}


bool ReconFilter::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  return true;
}


bool ReconFilter::Execute(){

  return true;
}


bool ReconFilter::Finalise(){

  return true;
}
