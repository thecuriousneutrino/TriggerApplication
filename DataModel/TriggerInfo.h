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
  void AddTrigger(TriggerType_t type, TimeDelta readout_start_time, TimeDelta readout_end_time, TimeDelta mask_start_time, TimeDelta mask_end_time, TimeDelta trigger_time, std::vector<float> info);

  /// Add all triggers from another TriggerInfo object
  void AddTriggers(TriggerInfo * in);

  /// Clear all triggers
  void Clear();

  /// The number of triggers
  unsigned int m_num_triggers;
  /// The type of Trigger
  std::vector<TriggerType_t> m_type;
  /// The starting time of the trigger window
  std::vector<TimeDelta> m_readout_start_time;
  /// The ending time of the trigger window
  std::vector<TimeDelta> m_readout_end_time;
  /// The starting time of the hit mask
  std::vector<TimeDelta> m_mask_start_time;
  /// The ending time of the hit mask
  std::vector<TimeDelta> m_mask_end_time;
  /// The actual time of the trigger
  std::vector<TimeDelta> m_trigger_time;
  /// Additional information, specific to the trigger
  std::vector<std::vector<float> > m_info;
};

std::ostream& operator<<(std::ostream& outs, const TriggerInfo& trig);

#endif
