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

  bool FileExists(std::string pathname, std::string filename);

  WCSimBonsai * m_bonsai;
  WCSimRootTrigger * m_trigger;
  int m_in_nhits;
  std::vector<int>   * m_in_PMTIDs;
  std::vector<float> * m_in_Ts;
  std::vector<float> * m_in_Qs;

  unsigned int m_nhits_min;
  unsigned int m_nhits_max;

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



};


#endif
