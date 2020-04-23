#ifndef PrepareSubSamples_H
#define PrepareSubSamples_H

#include <string>
#include <iostream>

#include "Tool.h"
#include "TimeDelta.h"

class PrepareSubSamples: public Tool {


 public:

  PrepareSubSamples();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  /// The desired maximum SubSample length
  TimeDelta m_sample_width;
  /// The desired SubSample overlap time
  TimeDelta m_sample_overlap;

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  int m_verbose;

  std::stringstream m_ss;

  void StreamToLog(int level) {
    Log(m_ss.str(), level, m_verbose);
    m_ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};

  /// Sort the digits in all SubSamples by time
  void SortSubSampleVector(std::vector<SubSample> &samples);
  /// Check whether any of the SubSamples needs to be split
  bool CheckSubSampleVectorNeedsSplitting(const std::vector<SubSample> &samples);
  /// Check whether the SubSample needs to be split
  bool CheckSubSampleNeedsSplitting(const SubSample &sample);
  /// Split all SubSamples
  std::vector<SubSample> SplitSubSampleVector(std::vector<SubSample> &samples);

};


#endif
