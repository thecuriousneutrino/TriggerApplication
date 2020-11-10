#include "PWCSimReader.h"

PWCSimReader_args::PWCSimReader_args():Thread_args(){}

PWCSimReader_args::~PWCSimReader_args(){}


PWCSimReader::PWCSimReader():Tool(){}


bool PWCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  int threadcount=0;

  if(!m_variables.Get("Verbosity",m_verbose)) m_verbose=0;
  if(!m_variables.Get("Threads",threadcount)) threadcount=1;
  if(!m_variables.Get("Concurrent_Slices",m_slicecount)) m_slicecount=2;
  if(!m_variables.Get("Input_File",inputfile)){
   
    Log("Error: No Inputfile given",0,m_verbose);    
    return false;
  }
  
  m_event_num=0;

  //Tom: i assume open the Wcsim input file here

  m_util=new Utilities(m_data->context);

  for(int i=0;i<threadcount;i++){
    PWCSimReader_args* tmparg=new PWCSimReader_args();
    tmparg->busy=0;
    tmparg->time_slice=0;
    tmparg->event_num=0;
    args.push_back(tmparg);
    std::stringstream tmp;
    tmp<<"T"<<i; 
    m_util->CreateThread(tmp.str(), &Thread, args.at(i));
  }
  
  
  
  return true;
}


bool PWCSimReader::Execute(){

  
  for(int i=0; i<args.size(); i++){
    if(args.at(i)->busy==0){
      
      if(args.at(i)->time_slice!=0){
	
	m_data->time_slices.push_back(args.at(i)->time_slice);
	args.at(i)->time_slice=0;
      
      }
      
      if(m_data->time_slices.size()<m_slicecount){
	TimeSlice* tmp=new TimeSlice;
	args.at(i)->time_slice=tmp;
	args.at(i)->event_num=m_event_num;
	m_event_num++;
	args.at(i)->busy=1;
      }
      
    }
  }
  
  usleep(500);
  
  return true;
}


bool PWCSimReader::Finalise(){

  for(int i=0;i<args.size();i++) m_util->KillThread(args.at(i));
    
  args.clear();
  
  delete m_util;
  m_util=0;

  return true;
}

void PWCSimReader::Thread(Thread_args* arg){

  PWCSimReader_args* args=reinterpret_cast<PWCSimReader_args*>(arg);

  if(!args->busy) usleep(500);
  else{ 

    ///Tom: load single WCSim event given by args->event_num into args->time_slice

    std::cout<<"loading event "<<args->event_num<<std::endl;
    sleep(10); //Tom: please remove was jsut for testing to simulate work load

    args->busy=0;
  }

}
