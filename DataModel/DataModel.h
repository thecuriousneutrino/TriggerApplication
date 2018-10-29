#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

//#include "TTree.h"

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

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
  SubSample(std::vector<int> PMTid, std::vector<int> time, std::vector<int> charge) {
    m_PMTid  = PMTid;
    m_time   = time;
    m_charge = charge;
  }

  std::vector<int> m_PMTid;
  std::vector<int> m_time;
  std::vector<int> m_charge;
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
  
  WCSimRootOptions WCSimOpt;
  WCSimRootEvent   WCSimEvt;
  WCSimRootGeom    WCSimGeo;


 private:


  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
