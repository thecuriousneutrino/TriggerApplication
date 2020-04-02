#ifndef EnergeticBONSAI_H
#define EnergeticBONSAI_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "WCSimEBonsai.h"

class EnergeticBONSAI: public Tool {


 public:

  EnergeticBONSAI();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  /// Instance of energetic-BONSAI
  WCSimEBonsai * m_ebonsai;
  /// Read in the digits in each trigger period to here
  WCSimRootTrigger * m_trigger;
  /// Read in the total number of hits in the given trigger
  int m_in_nhits;
  /// Read in the PMT IDs of all hits in the given trigger
  std::vector<int>   * m_in_PMTIDs;
  /// Read in the times of all hits in the given trigger
  std::vector<float> * m_in_Ts;
  /// x,y,z of input reconstructed vertex
  float m_vertex[3];

  /// Number of hits must be greater than this, else energetic-BONSAI won't be run on this trigger
  unsigned int m_nhits_min;
  /// Number of hits must be less than this, else energetic-BONSAI won't be run on this trigger
  unsigned int m_nhits_max;

  /// Holds reconstructed vertex information
  ReconInfo * m_input_filter;
  /// Which named filter to use? For preselecting which reconstructed vertices will be used by energetic-BONSAI
  std::string m_input_filter_name;

  /// Number of working PMTs, taken from config file (defaults to NPMTs in geometry)
  int         m_n_working_pmts;
  /// Name of the detector, used to set default energetic-BONSAI parameters
  std::string m_detector_name;
  /// Overwrite the precalculated nearest neighbours ROOT file that energetic-BONSAI uses?
  bool        m_overwrite_nearest;

  int m_verbose;

  std::stringstream m_ss;

  void StreamToLog(int level) {
    Log(m_ss.str(), level, m_verbose);
    m_ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};


};


#endif
