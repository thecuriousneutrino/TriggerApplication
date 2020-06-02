#ifndef SupernovaDirectionCalculator_H
#define SupernovaDirectionCalculator_H

#include <string>
#include <iostream>

#include "Tool.h"
#include "Stopwatch.h"

class SupernovaDirectionCalculator: public Tool {


 public:

  SupernovaDirectionCalculator();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();

  /// Return the weight for the event
  double GetEventWeight(double log10_energy);


 private:

  ReconInfo * m_in_filter;
  std::string m_input_filter_name;

  /// Sum of event directions
  double m_direction[3];

  /// Enable weighting of events
  bool m_weight_events;

  /// Vector of log10(energy) for weight interpolation
  std::vector<double> m_log10_energy;
  /// Vector of weights for interpolation
  std::vector<double> m_weight;

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  /// Verbosity level, as defined in tool parameter file
  int m_verbose;

  /// For easy formatting of Log messages
  std::stringstream m_ss;

  /// Print the current value of the streamer at the set log level,
  ///  then clear the streamer
  void StreamToLog(int level) {
    Log(m_ss.str(), level, m_verbose);
    m_ss.str("");
  }

  /// Log level enumerations
  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};


};


#endif
