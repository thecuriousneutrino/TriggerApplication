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

class PMTInfo{
 public:
  PMTInfo(int tubeno, float x, float y, float z) {
    m_tubeno = tubeno;
    m_x = x;
    m_y = y;
    m_z = z;
  }
  int m_tubeno;
  float m_x, m_y, m_z;
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

  std::vector<SubSample> IDSamples;
  std::vector<SubSample> ODSamples;

  std::vector<PMTInfo> IDGeom;
  std::vector<PMTInfo> ODGeom;

  bool triggeroutput;

  double IDPMTDarkRate;
  double ODPMTDarkRate;
  double IDNPMTs;
  double ODNPMTs;
  
  WCSimRootOptions WCSimOpt;
  WCSimRootEvent   WCSimEvt;
  WCSimRootGeom    WCSimGeo;


 private:


  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
