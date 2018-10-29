#ifndef WCSimReader_H
#define WCSimReader_H

#include <string>
#include <iostream>

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
  bool ReadTree(TChain * chain);
  bool CompareTree(TChain * chain);

  TChain * fChainOpt;
  TChain * fChainEvent;
  TChain * fChainGeom;

  long int fCurrEvent;
  long int fNEvents;

  std::string fInFile;
  std::string fFileList;

};


#endif
