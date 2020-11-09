#ifndef ReconRandomiser_H
#define ReconRandomiser_H

#include <string>
#include <iostream>

#include "TRandom3.h"

#include "Tool.h"
#include "TMath.h"
#include "Stopwatch.h"

typedef enum EDistribution {kUniform, kGauss, kFixed} Distribution_t;

static std::string EnumAsString(Distribution_t dist) {
  switch(dist) {
  case (kUniform):
    return "Uniform";
    break;
  case (kGauss):
    return "Gauss";
    break;
  case (kFixed):
    return "Fixed";
    break;
  default:
    return "";
  }
  return "";
}

class ReconRandomiser: public Tool {


 public:

  ReconRandomiser();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  void CreateVertex(double * pos);
  double GetRandomNumber(Distribution_t dist, double max, double mean, double width, const int maxcount);
  Distribution_t GetDistributionType(double width, const char * axis);

  long int fCurrEvent;
  long int fNEvents;

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
  Distribution_t fXDistribution;
  Distribution_t fYDistribution;
  Distribution_t fZDistribution;

  //time distribution
  double fTMin;
  double fTMax;

  //direction distribution
  bool   fRandomDirection;

  TRandom3 * fRand;

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
