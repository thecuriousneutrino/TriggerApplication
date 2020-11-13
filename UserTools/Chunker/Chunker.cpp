#include "Chunker.h"

Chunker_args::Chunker_args():Thread_args(){}

Chunker_args::~Chunker_args(){}


Chunker::Chunker():Tool(){}


bool Chunker::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  int threadcount=0;

  if(!m_variables.Get("Threads",threadcount)) threadcount=4;
  if(!m_variables.Get("Verbosity",m_verbose)) m_verbose=0;
  if(!m_variables.Get("Chunk_size",m_chunk_size)) m_chunk_size=200;

  m_util=new Utilities(m_data->context);

  for(int i=0;i<threadcount;i++){
    Chunker_args* tmparg=new Chunker_args();   
    tmparg->busy=0;
    tmparg->time_slice=0;
    tmparg->chunks=0;
    args.push_back(tmparg);
    std::stringstream tmp;
    tmp<<"T"<<i; 
    m_util->CreateThread(tmp.str(), &Thread, args.at(i));
  }

  m_current_slice=0;  
  m_freethreads=threadcount;
  
    
  
  return true;
}


bool Chunker::Execute(){

  for(int i=0; i<args.size(); i++){

    if(args.at(i)->busy==0){

      if(args.at(i)->chunks!=0 &&  args.at(i)->time_slice!=0){

	args.at(i)->time_slice->chunks=args.at(i)->chunks;
	args.at(i)->chunks=0;
	args.at(i)->time_slice=0;

      }

      if(m_current_slice==0 && m_data->time_slices.size()>0){

	m_current_slice=m_data->time_slices.at(0);

	args.at(i)->time_slice=m_current_slice;
	args.at(i)->busy=1;

      }

      else if(m_data->time_slices.size()>0 && m_current_slice!=m_data->time_slices.at(m_data->time_slices.size()-1)){

	for(int i=0;i<m_data->time_slices.size()-1;i++){

	  if(m_current_slice==m_data->time_slices.at(i)){

	    m_current_slice=m_data->time_slices.at(i+1);

	    args.at(i)->time_slice=m_current_slice;
	    args.at(i)->busy=1;
	    break;

	  }
	  
	}

      }
      
    }

  }
  
  //std::cout<<"d18"<<std::endl;
  /*m_freethreads=0;
  for(int i=0; i<args.size(); i++){
    if(args.at(i)->busy==0) m_freethreads++;
  }

  std::cout<<"free threads="<<m_freethreads<<":"<<args.size()<<std::endl;
  */
  
    
  return true;
}


bool Chunker::Finalise(){

  for(int i=0;i<args.size();i++) m_util->KillThread(args.at(i));
    
  args.clear();
  
  delete m_util;
  m_util=0;

  m_current_slice=0;

  return true;
}

void Chunker::Thread(Thread_args* arg){

  Chunker_args* args=reinterpret_cast<Chunker_args*>(arg);

  if(!args->busy) usleep(100);
  else{ 
    
    args->chunks= new std::map<float,Chunk*>;

    /*
    for( hits in args->time_slice){
      
      float rounded_time=hit.time-(hit.time % m_chunk_size);
      
      if(args->chunks.count(rounded_time)==0) args->chunks[rounded_time]=new Chunk;

      Hit tmphit;
      tmphit.time=hit.time;
      tmphit.charge=hit.charge;
      tmphit.PMTID=hit.PMTID;

      args->chunks[rounded_time].hits.push_back(tmphit);

    }

    */



    args->busy=0;
  }

}
