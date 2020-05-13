#ifndef nhits_H
#define nhits_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "GPUFunctions.h"
#include "Stopwatch.h"

class NHits: public Tool {

 public:

  NHits();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();

 private:
 
  /// Width of the sliding window
  TimeDelta m_trigger_search_window;
  /// Trigger threshold - number of digits must be above this value (equal to does not fire the trigger)
  unsigned int m_trigger_threshold;
  /// Pre-trigger time for saving digits
  TimeDelta m_trigger_save_window_pre;
  /// Post-trigger time for saving digits
  TimeDelta m_trigger_save_window_post;
  /// Pre-trigger time for masking digits from future tools
  TimeDelta m_trigger_mask_window_pre;
  /// Post-trigger time for masking digits from future tools
  TimeDelta m_trigger_mask_window_post;
  /// Trigger on OD digits, rather than ID digits?
  bool m_trigger_OD;
  /// degrade data type from float to int
  bool m_degrade_CPU;

  /// CPU version of the NDigits algorithm
  void AlgNDigits(const SubSample * samples);

#ifdef GPU   
  /// integer times to run over GPU card
  std::vector<int> m_time_int;
#endif

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
