#ifndef BONSAI_H
#define BONSAI_H

#include <string>
#include <iostream>

#include "Tool.h"

#include <vector>

#include "WCSimRootEvent.hh"
#include "WCSimBonsai.hh"

class BONSAI: public Tool {


 public:

  BONSAI();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  bool FileExists(const char * filename);

  WCSimBonsai * _bonsai;
  WCSimRootTrigger * _trigger;
  int _in_nhits;
  std::vector<int>   * _in_PMTIDs;
  std::vector<float> * _in_Ts;
  std::vector<float> * _in_Qs;

  std::string fOutFilename;
  TFile fOutFile;
  TTree * fTVertexInfo;
  long   fEventNum;
  long   fTriggerNum;
  double fVertex[4];  //x,y,z,t
  double fDirectionEuler[3]; // theta (zenith), phi (azimuth), alpha
  double fCherenkovCone[2];  // cos(Cherenkov angle), ellipticity
  double fDirectionLikelihood;
  double fGoodnessOfFit;
  double fGoodnessOfTimeFit;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};



};


#endif
