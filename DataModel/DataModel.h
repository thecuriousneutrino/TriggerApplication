#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

#include "TChain.h"

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

#include "Store.h"
#include "BoostStore.h"
#include "Logging.h"

#include <zmq.hpp>

#include <SubSample.h>
#include <PMTInfo.h>

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
  int IDNPMTs;
  int ODNPMTs;
  
  TChain * WCSimGeomTree;
  TChain * WCSimOptionsTree;
  TChain * WCSimEventTree;
  std::vector<int> CurrentWCSimEventNums;
  TObjArray * CurrentWCSimFiles;
  WCSimRootEvent * WCSimEventID;
  WCSimRootEvent * WCSimEventOD;

  bool HasOD;
  bool IsMC;

 private:


  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
