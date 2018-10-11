#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

//#include "TTree.h"

#include "Store.h"
#include "BoostStore.h"
#include "Logging.h"

#include <zmq.hpp>

class SubSample{

 public:

  SubSample();
  SubSample(std::vector<int> PMTid,std::vector<int> time){
    m_PMTid=PMTid;
    m_time=time;
  }

  std::vector<int> m_PMTid;
  std::vector<int> m_time;

};

class DataModel {


 public:
  
  DataModel();

  //TTree* GetTTree(std::string name);
  //void AddTTree(std::string name,TTree *tree);
  //void DeleteTTree(std::string name);

  Store vars;
  BoostStore CStore;
  std::map<std::string,BoostStore*> Stores;
  
  Logging *Log;

  zmq::context_t* context;

  std::vector<SubSample> Samples;
  bool triggeroutput;
  


 private:


  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
