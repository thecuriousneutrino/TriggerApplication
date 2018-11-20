#ifndef nhits_H
#define nhits_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "GPUFunctions.h"

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

};


#endif
