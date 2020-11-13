#include "Pnhits.h"

Pnhits_args::Pnhits_args():Thread_args(){}

Pnhits_args::~Pnhits_args(){}


Pnhits::Pnhits():Tool(){}


bool Pnhits::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  int threadcount=0;
  if(!m_variables.Get("Threads",threadcount)) threadcount=4;
  if(!m_variables.Get("Verbosity",m_verbose)) m_verbose=0;


  m_util=new Utilities(m_data->context);

  for(int i=0;i<threadcount;i++){
    Pnhits_args* tmparg=new Pnhits_args();   
    tmparg->busy=0;
    tmparg->numchunks=0;
    args.push_back(tmparg);
    std::stringstream tmp;
    tmp<<"T"<<i; 
    m_util->CreateThread(tmp.str(), &Thread, args.at(i));
  }

  m_current_slice=0;  
  m_freethreads=threadcount;
  
    
  
  return true;
}


bool Pnhits::Execute(){

  if(m_data->time_slices.size()>0 && m_current_slice!=m_data->time_slices.at(m_data->time_slices.size()-1)){

    
    if(m_current_slice==0) m_current_slice=m_data->time_slices.at(0);
    
    else {

      for(int i=0;i<m_data->time_slices.size()-1;i++){

	if(m_current_slice==m_data->time_slices.at(i)){

	  m_current_slice=m_data->time_slices.at(i+1);
	  break;  

	}

      }

    }

    if(m_current_slice->chunks!=0){
      int numchunks =m_current_slice->chunks->size()/args.size();

      
      for(int i=0; i<(args.size()-1); i++){
	job tmp;

	tmp.chunkit= m_current_slice->chunks->begin();

	for(int j=0;j<(i*numchunks);j++) tmp.chunkit++;

	tmp.numchunks=numchunks-1;
	jobs.push_back(tmp);

      }

      job tmp;
      tmp.chunkit= m_current_slice->chunks->begin();
      for(int j=0;j<(numchunks*(args.size()-1));j++) tmp.chunkit++;
      tmp.numchunks=m_current_slice->chunks->size() - (numchunks*(args.size()-1));
      
      jobs.push_back(tmp);
      
      
      
    }
  }
  
  
  for(int i=0; i<args.size(); i++){
    if(args.at(i)->busy==0 && jobs.size()>0){
      
      args.at(i)->chunkit=jobs.at(0).chunkit;
      args.at(i)->numchunks=jobs.at(0).numchunks;
      args.at(i)->busy=1;
      jobs.pop_front();
      
    }
    
  }
  
  /*
    m_freethreads=0;
    for(int i=0; i<args.size(); i++){
    if(args.at(i)->busy==0) m_freethreads++;
    }
    
    std::cout<<"free threads="<<m_freethreads<<":"<<args.size()<<std::endl;
    
    sleep(1);
  */
  
  return true;
}


bool Pnhits::Finalise(){

  for(int i=0;i<args.size();i++) m_util->KillThread(args.at(i));
  
  args.clear();
  
  delete m_util;
  m_util=0;
  
  jobs.clear();
  m_current_slice=0;
  
  return true;
}

void Pnhits::Thread(Thread_args* arg){
  
  Pnhits_args* args=reinterpret_cast<Pnhits_args*>(arg);
  
  if(!args->busy) usleep(100);
  else{ 
    
    for(int i=0;i<<args->numchunks;i++){
 
      int sum=args->chunkit->second->hits.size();
      BoostStore* tmp= &args->chunkit->second->tool_output;
      args->chunkit++;
      sum+=args->chunkit->second->hits.size();
      tmp->Set("nhits",sum);

    }
    
    args->busy=0;
  }
  
}
