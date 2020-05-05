#include "pass_all.h"

pass_all::pass_all():Tool(){}


bool pass_all::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  return true;
}


bool pass_all::Execute(){
  int n_digits = m_data->IDSamples.at(0).m_time.size();
  m_data->IDTriggers.AddTrigger(kTriggerNoTrig,
				std::numeric_limits<float>::min(), //readout
				std::numeric_limits<float>::max(),
				std::numeric_limits<float>::min(), //mask
				std::numeric_limits<float>::max(),
				0, //triger time
				std::vector<float>(1, n_digits));

  return true;
}


bool pass_all::Finalise(){

  return true;
}
