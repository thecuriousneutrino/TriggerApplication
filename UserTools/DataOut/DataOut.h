#ifndef DataOut_H
#define DataOut_H

#include <string>
#include <iostream>
#include <map>

#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"
#include "WCSimRootOptions.hh"

#include "TFile.h"
#include "TTree.h"

#include "Tool.h"

class DataOut: public Tool {


 public:

  DataOut();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:
  /// Runs other methods to take information from the DataModel and create/populate the WCSimRootEvent
  void ExecuteSubDet(WCSimRootEvent * wcsim_event, std::vector<SubSample> & samples, WCSimRootEvent * original_wcsim_event = 0);
  /// If there are multiple triggers in the event,
  ///  create subevents (i.e. new WCSimRootTrigger's) in the WCSimRootEvent
  /// Also sets the time correctly
  void CreateSubEvents(WCSimRootEvent * wcsim_event);
  //Get the WCSim "date", used later to give the hits the correct absolute time.
  // Also add the trigger offset from the config file
  TimeDelta GetOffset(WCSimRootEvent * original_wcsim_event = 0);
  /// For every hit, if it's in a trigger window,
  ///  add it to the appropriate WCSimRootTrigger in the WCSimRootEvent
  void FillHits(WCSimRootEvent * wcsim_event, std::vector<SubSample> & samples);
  /// If this is an MC file, we also need to add
  /// - true tracks
  /// - true hits
  void AddTruthInfo(WCSimRootEvent * wcsim_event, WCSimRootEvent * original_wcsim_event, const TimeDelta & time_shift);
  //Set some trigger header infromation that requires all the hits to be 
  // present to calculate e.g. sumq
  void FinaliseSubEvents(WCSimRootEvent * wcsim_event);

  /// Output ROOT filename that this tool RECREATE's
  std::string m_output_filename;
  /// Output ROOT file
  TFile * m_output_file;
  /// Tree contain WCSimRootEvent(s), and the original WCSim filename / event number
  TTree * m_event_tree;

  /// Combined list of triggers from all sources (ID+OD)
  TriggerInfo * m_all_triggers;
  /// A time used to offset all hit times. Set by config file
  TimeDelta m_trigger_offset;

  /// Current event number
  int m_event_num;

  /// If true, saves hits that failed the trigger, rather those that passed
  bool m_save_only_failed_hits;
  /// If false, only 1 hit is allowed to be saved per trigger, rather than all hits from that trigger
  bool m_save_multiple_hits_per_trigger;

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
