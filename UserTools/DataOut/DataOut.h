#ifndef DataOut_H
#define DataOut_H

#include <string>
#include <iostream>

#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"
#include "WCSimRootOptions.hh"

#include "TFile.h"
#include "TTree.h"

#include "Tool.h"

class DataOut: public Tool {


 public:

  DataOut();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:
  void RemoveDigits(WCSimRootEvent * WCSimEvent);
  bool TimeInRange(double time);

  std::string fOutFilename;
  TFile fOutFile;
  TTree * fTreeEvent;
  TTree * fTreeGeom;
  TTree * fTreeOptions;
  WCSimRootEvent * fWCSimEventID;
  WCSimRootEvent * fWCSimEventOD;
  TString * fWCSimFilename;

  std::vector<std::pair<double, double> > fTriggerIntervals;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
