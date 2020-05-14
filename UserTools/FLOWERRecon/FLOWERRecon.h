#ifndef FLOWERRecon_H
#define FLOWERRecon_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "WCSimFLOWER.h"

class FLOWERRecon: public Tool {


 public:

  FLOWERRecon();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  /// Instance of FLOWER
  WCSimFLOWER * m_flower;
  /// Read in the total number of hits in the given trigger
  int m_in_nhits;
  /// Read in the PMT IDs of all hits in the given trigger
  std::vector<int>   * m_in_PMTIDs;
  /// Read in the times of all hits in the given trigger
  std::vector<float> * m_in_Ts;
  /// x,y,z of input reconstructed vertex
  float m_vertex[3];

  /// Number of hits must be greater than this, else FLOWER won't be run on this trigger
  unsigned int m_nhits_min;
  /// Number of hits must be less than this, else FLOWER won't be run on this trigger
  unsigned int m_nhits_max;

  /// Holds reconstructed vertex information
  ReconInfo * m_input_filter;
  /// Which named filter to use? For preselecting which reconstructed vertices will be used by FLOWER
  std::string m_input_filter_name;

  /// Number of working PMTs, taken from config file (defaults to NPMTs in geometry)
  int         m_n_working_pmts;
  /// Name of the detector, used to set default FLOWER parameters
  std::string m_detector_name;
  /// Overwrite the precalculated nearest neighbours ROOT file that FLOWER uses?
  bool        m_overwrite_nearest;

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
