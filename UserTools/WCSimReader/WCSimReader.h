#ifndef WCSimReader_H
#define WCSimReader_H

#include <string>
#include <iostream>
#include <sstream>

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

#include "TChain.h"

#include "Tool.h"

class WCSimReader: public Tool {


 public:

  WCSimReader();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:
  //methods used in Initialise

  /// Calls AddTreeToChain for  m_input_filename, or for every file within the m_input_filelist
  bool ReadTree(TChain * chain);
  /// Adds a file (or files - wildcards allowed) to the chain
  bool AddTreeToChain(const char * fname, TChain * chain);
  /// Checks for equality between entry 0 and other entries in the chain
  /// Mode 0: compares WCSimRootOptions
  /// Mode 1: compares WCSimRootGeom
  bool CompareTree(TChain * chain, int mode);
  /// Checks for equality between 2 variables
  template <typename T> bool CompareVariable(T v1, T v2, const char * tag);

  //methods used in Execute

  /// Creates a SubSample containing the digits from the current m_wcsim_trigger
  SubSample GetDigits();

  /// Input wcsimRootOptionsT chain - holds WCSim run options
  TChain * m_chain_opt;
  /// Input wcsimT chain - holds WCSim events 
  TChain * m_chain_event;
  /// Input wcsimGeoT chain - holds WCSim geometry
  TChain * m_chain_geom;

  /// Holds WCSim running options - trigger thresholds, geometry names, input .kin filename, etc.
  WCSimRootOptions * m_wcsim_opt;
  /// Holds event information for the ID - tracks, hits, digits
  WCSimRootEvent   * m_wcsim_event_ID;
  /// Holds event information for the OD - hits, digits
  WCSimRootEvent   * m_wcsim_event_OD;
  /// Holds geometry information - tank size, PMT size, PMT positions, etc.
  WCSimRootGeom    * m_wcsim_geom;
  /// Holds trigger information - trigger time, digits, etc.
  WCSimRootTrigger * m_wcsim_trigger;

  /// The current WCSim event number
  long int m_current_event_num;
  /// The first WCSim event number to read
  long int m_first_event_num;
  /// The total number of events in m_chain_event
  long int m_n_events;

  /// The input WCSim filename from config file (wildcards allowed)
  std::string m_input_filename;
  /// The input WCSim filelist filename from config file
  std::string m_input_filelist;

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  /// Verbosity level
  int m_verbose;

  /// Streamer for easy formatting of log messages
  std::stringstream m_ss;

  /// Helper function to print streamer at specified level, and clear streamer
  void StreamToLog(int level) {
    Log(m_ss.str(), level, m_verbose);
    m_ss.str("");
  }

  /// enumeration of the log levels
  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};
};


#endif
