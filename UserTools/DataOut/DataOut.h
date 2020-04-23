#ifndef DataOut_H
#define DataOut_H

#include <string>
#include <iostream>
#include <map>

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
  void CreateSubEvents(WCSimRootEvent * WCSimEvent);
  void FinaliseSubEvents(WCSimRootEvent * WCSimEvent);
  void RemoveDigits(WCSimRootEvent * WCSimEvent,
		    std::map<int, std::map<int, bool> > & NDigitPerPMTPerTriggerMap);
  void MoveTracks(WCSimRootEvent * WCSimEvent);
  int  TimeInTriggerWindow(TimeDelta time);
  unsigned int TimeInTriggerWindowNoDelete(TimeDelta time);

  std::string fOutFilename;
  TFile fOutFile;
  TTree * fTreeEvent;
  TTree * fTreeGeom;
  TTree * fTreeOptions;
  TString * fWCSimFilename;

  TriggerInfo * fTriggers;
  double fTriggerOffset;

  int fEvtNum;

  bool fSaveOnlyFailedDigits;
  bool fSaveMultiDigiPerTrigger;
  std::map<int, std::map<int, bool> > fIDNDigitPerPMTPerTriggerMap;
  std::map<int, std::map<int, bool> > fODNDigitPerPMTPerTriggerMap;

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  int m_verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, m_verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
