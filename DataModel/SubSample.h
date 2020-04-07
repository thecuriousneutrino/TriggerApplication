#ifndef SUBSAMPLE_H
#define SUBSAMPLE_H

#include <iostream>
#include <vector>
#include <stdint.h>

#include "TimeDelta.h"

class SubSample{

 public:

  SubSample() {};

  /// Deprecated constructor, use empty constructor and Append instead.
  __attribute__((deprecated))
  SubSample(
    std::vector<int> PMTid,
    std::vector<TimeDelta::short_time_t> time,
    std::vector<float> charge = std::vector<float>(),
    TimeDelta timestamp = 0
  );

  /// Timestamp of the whole SubSample
  TimeDelta m_timestamp;

  /// Append the hits in a different SubSample to this one
  ///
  /// Returns `true` if successful.
  bool Append(const SubSample& sub);

  /// Append the hits in a different SubSample to this one
  ///
  /// Returns `true` if successful.
  bool Append(const std::vector<int> PMTid, const std::vector<TimeDelta::short_time_t> time, const std::vector<float> charge, const TimeDelta timestamp);

  /// Vector of PMT IDs for all hits in SubSample
  std::vector<int> m_PMTid;
  /// Vector of hit times relative to timestamp for all hits in SubSample. Unit: ns
  std::vector<TimeDelta::short_time_t> m_time;
  /// Vector of charges for all hits in SubSample. Unit: ?
  std::vector<float> m_charge;

  /// Return the absolute time (timestamp + digit time) of a digit
  TimeDelta AbsoluteDigitTime(int index) const;

  /// Sort all digits in the SubSample by their time
  void SortByTime();
  /// Check whether all hits are in time order
  bool IsSortedByTime() const;

  /// Split SubSample into multiple overlapping ones
  ///
  /// The SubSample needs to be sorted by time for this to work!
  /// Otherwise it will return an empty vector.
  std::vector<SubSample> Split(TimeDelta target_width, TimeDelta target_overlap) const;
};

#endif
