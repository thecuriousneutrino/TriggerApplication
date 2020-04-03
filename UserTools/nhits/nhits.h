#ifndef nhits_H
#define nhits_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "GPUFunctions.h"
#include "Stopwatch.h"

class nhits: public Tool {


 public:

  nhits();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:
  float fTriggerSearchWindow;
  float fTriggerSearchWindowStep;
  float fTriggerThreshold;
  float fTriggerSaveWindowPre;
  float fTriggerSaveWindowPost;
  bool  fTriggerOD;

  void AlgNDigits(const SubSample * samples); ///< Modified from WCSim v1.7.0
 
  static const int kALongTime;      ///< An arbitrary long time to use in loops (ns)

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
