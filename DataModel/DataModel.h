#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

#include "TChain.h"

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"
#include "WCSimEnumerations.hh"

#include "Store.h"
#include "BoostStore.h"
#include "Logging.h"

#include <zmq.hpp>

#include <SubSample.h>
#include <PMTInfo.h>
#include <TriggerInfo.h>
#include <ReconInfo.h>

class DataModel {


 public:
  
  DataModel();

  //TTree* GetTTree(std::string name);
  //void AddTTree(std::string name,TTree *tree);
  //void DeleteTTree(std::string name);

  ReconInfo * GetFilter(std::string name, bool can_create);

  Store vars;
  BoostStore CStore;
  std::map<std::string,BoostStore*> Stores;
  
  Logging *Log;

  zmq::context_t* context;

  std::vector<SubSample> IDSamples;
  std::vector<SubSample> ODSamples;

  std::vector<PMTInfo> IDGeom;
  std::vector<PMTInfo> ODGeom;

  TriggerInfo IDTriggers;
  TriggerInfo ODTriggers;

  ReconInfo RecoInfo;
  std::map<std::string, ReconInfo*> RecoInfoMap;

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
  WCSimRootEvent * IDWCSimEvent_Raw;
  WCSimRootEvent * ODWCSimEvent_Raw;
  WCSimRootEvent * IDWCSimEvent_Triggered;
  WCSimRootEvent * ODWCSimEvent_Triggered;

  std::vector<SNWarningParams> SupernovaWarningParameters;

  bool HasOD;
  bool IsMC;

 private:


  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
