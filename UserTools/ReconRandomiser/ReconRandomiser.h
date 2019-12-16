#ifndef ReconRandomiser_H
#define ReconRandomiser_H

#include <string>
#include <iostream>

#include "TRandom3.h"

#include "Tool.h"

class ReconRandomiser: public Tool {


 public:

  ReconRandomiser();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  void CreateVertex(double * pos);

  int fNVerticesMean;

  //vertex distribution
  double fXMean;
  double fXWidth;
  double fYMean;
  double fYWidth;
  double fZMean;
  double fZWidth;
  double fMaxZPos;
  double fMaxRPos;
  bool   fFlatR;
  bool   fUniformX;
  bool   fUniformY;
  bool   fUniformZ;

  //time distribution
  double fTMin;
  double fTMax;

  //direction distribution
  bool   fRandomDirection;

  TRandom3 * fRand;

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};


};


#endif
