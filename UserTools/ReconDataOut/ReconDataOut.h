#ifndef ReconDataOut_H
#define ReconDataOut_H

#include <string>
#include <iostream>

#include "Tool.h"

class ReconDataOut: public Tool {


 public:

  ReconDataOut();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  ReconInfo * fInFilter;
  std::string fInputFilterName;

  std::string fOutFilename;
  TFile fOutFile;
  TTree * fTreeRecon;

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
