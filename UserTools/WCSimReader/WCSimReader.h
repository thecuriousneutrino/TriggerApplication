#ifndef WCSimReader_H
#define WCSimReader_H

#include <string>
#include <iostream>
#include <sstream>

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

#include "TChain.h"

#include "Tool.h"

class WCSimReader: public Tool {


 public:

  WCSimReader();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:
  //methods used in Initialise
  bool ReadTree(TChain * chain);
  bool CompareTree(TChain * chain);

  //methods used in Execute
  SubSample GetDigits();

  TChain * fChainOpt;
  TChain * fChainEvent;
  TChain * fChainGeom;

  WCSimRootOptions * fWCOpt;
  WCSimRootEvent   * fWCEvtID;
  WCSimRootEvent   * fWCEvtOD;
  WCSimRootGeom    * fWCGeo;
  WCSimRootTrigger * fEvt;

  long int fCurrEvent;
  long int fNEvents;

  std::string fInFile;
  std::string fFileList;
};


#endif
