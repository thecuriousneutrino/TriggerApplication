#ifndef TRIGGERINFO_H
#define TRIGGERINFO_H

#include <iostream>
#include <vector>

#include "WCSimEnumerations.hh"
#include "TimeDelta.h"

class TriggerInfo{

 public:

  TriggerInfo();

  /// Add a trigger, all times in ns
  __attribute__((deprecated))
  void AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, std::vector<float> info);

  /// Add a trigger
  void AddTrigger(TriggerType_t type, TimeDelta starttime, TimeDelta endtime, TimeDelta triggertime, std::vector<float> info);

  /// Add all triggers from another TriggerInfo object
  void AddTriggers(TriggerInfo * in);

  /// Clear all triggers
  void Clear();

  /// Sort triggers by their starting time
  void SortByStartTime();

  /// The number of triggers
  unsigned int m_N;
  /// The type of Trigger
  std::vector<TriggerType_t> m_type;
  /// The starting time of the trigger window
  std::vector<TimeDelta>     m_starttime;
  /// The ending time of the trigger window
  std::vector<TimeDelta>     m_endtime;
  /// The actual time of the trigger
  std::vector<TimeDelta>     m_triggertime;
  /// Additional information, specific to the trigger
  std::vector<std::vector<float> > m_info;
};





#endif
