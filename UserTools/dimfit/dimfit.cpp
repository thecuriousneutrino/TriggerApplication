#include "dimfit.h"

dimfit::dimfit():Tool(){}


bool dimfit::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  return true;
}


bool dimfit::Execute(){

  return true;
}


bool dimfit::Finalise(){

  return true;
}
