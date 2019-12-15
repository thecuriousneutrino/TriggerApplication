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
  void CreateSubEvents(WCSimRootEvent * WCSimEvent);
  void FinaliseSubEvents(WCSimRootEvent * WCSimEvent);
  void RemoveDigits(WCSimRootEvent * WCSimEvent);
  int  TimeInTriggerWindow(double time);

  std::string fOutFilename;
  TFile fOutFile;
  TTree * fTreeEvent;
  TTree * fTreeGeom;
  TTree * fTreeOptions;
  TTree * fTreeRecon;
  TString * fWCSimFilename;

  std::vector<std::pair<double, double> > fTriggerIntervals;
  TriggerInfo * fTriggers;

  int    fRTTriggerNum;
  int    fRTNHits;
  Reconstructer_t fRTReconstructer;
  double fRTTime;
  double fRTVertex[4];  //x,y,z
  bool   fRTHasDirection;
  double fRTDirectionEuler[3]; // theta (zenith), phi (azimuth), alpha
  double fRTCherenkovCone[2];  // cos(Cherenkov angle), ellipticity
  double fRTDirectionLikelihood;
  double fRTGoodnessOfFit;
  double fRTGoodnessOfTimeFit;

  int fEvtNum;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
