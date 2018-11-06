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
  bool AddTreeToChain(const char * fname, TChain * chain);
  bool CompareTree(TChain * chain, int mode);
  template <typename T> bool CompareVariable(T v1, T v2, const char * tag);

  //methods used in Execute
  SubSample GetDigits();

  TChain * fChainOpt;
  TChain * fChainEvent;
  TChain * fChainGeom;

  WCSimRootOptions * fWCOpt;
  WCSimRootOptions * fWCOpt_Store;
  WCSimRootEvent   * fWCEvtID;
  WCSimRootEvent   * fWCEvtOD;
  WCSimRootGeom    * fWCGeo;
  WCSimRootGeom    * fWCGeo_Store;
  WCSimRootTrigger * fEvt;

  long int fCurrEvent;
  long int fNEvents;

  std::string fInFile;
  std::string fFileList;
  std::vector<std::string> fWCSimFiles;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
