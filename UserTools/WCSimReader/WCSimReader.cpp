#include "WCSimReader.h"

WCSimReader::WCSimReader():Tool(){}


bool WCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  return true;
}


bool WCSimReader::Execute(){

  return true;
}


bool WCSimReader::Finalise(){

  return true;
}
