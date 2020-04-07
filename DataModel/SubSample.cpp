#include "SubSample.h"

SubSample::SubSample(std::vector<int> PMTid, std::vector<TimeDelta::short_time_t> time, std::vector<float> charge, TimeDelta timestamp){
  // If charge vector is empty, fill with 0s
  if (charge.size() == 0){
    charge =  std::vector<float>(PMTid.size(), 0.);
  }
  // Assign values
  m_PMTid  = PMTid;
  m_time   = time;
  m_charge = charge;
  m_timestamp = timestamp;
}

void SubSample::SortByTime(){
  //algorithm borrowed from WCSimWCDigi::SortArrayByHitTime()
  int i, j;
  TimeDelta::short_time_t save_time;
  int save_PMTid;
  double save_charge;

  for (i = 1; i < m_PMTid.size(); ++i) {
    save_time       = m_time[i];
    save_PMTid      = m_PMTid[i];
    save_charge     = m_charge[i];
    for (j = i; j > 0 && m_time[j-1] > save_time; j--) {
      m_time[j]     = m_time[j-1];
      m_PMTid[j]    = m_PMTid[j-1];
      m_charge[j]   = m_charge[j-1];
    }//j
    m_time[j]     = save_time;
    m_PMTid[j]    = save_PMTid;
    m_charge[j]   = save_charge;
  }//i
}

bool SubSample::IsSortedByTime() const {
  int size = m_time.size();
  for (int i=0; i<size-1; ++i){
    if (m_time[i] > m_time[i+1]){
      return false;
    }
  }
  return true;
}

std::vector<SubSample> SubSample::Split(TimeDelta target_width, TimeDelta target_overlap) const {

  // If the sample is empty, just return a copy of self
  if (m_time.size() == 0){
    return std::vector<SubSample>(1, *this);
  }

  // Ensure everything is sorted
  if (not IsSortedByTime()){
    // Otherwise return empty vector
    return std::vector<SubSample>();
  }

  // Distance between SubSamples
  TimeDelta target_stride = target_width - target_overlap;

  // The vector of samples to be returned
  std::vector<SubSample> split_samples;

  // Temporary information for storing digits that will be added to the samples
  std::vector<float> temp_charge;
  std::vector<int> temp_PMTid;
  std::vector<TimeDelta::short_time_t> temp_time;
  TimeDelta temp_timestamp = m_timestamp;

  // Set first SubSample timestamp according to first digit time
  // Make sure hit times are not negative:
  while ( m_timestamp + TimeDelta(m_time.at(0)) - temp_timestamp < TimeDelta(0.) ){
    temp_timestamp -= target_stride;
  }
  // Make sure first SubSample is not empty
  while ( m_timestamp + TimeDelta(m_time.at(0)) - temp_timestamp > target_stride ){
    temp_timestamp += target_stride;
  }

  // Add digits to new SubSamples
  for (int i = 0; i < m_time.size(); ++i){
    TimeDelta time_in_window = m_timestamp + TimeDelta(m_time.at(i)) - temp_timestamp;
    if (time_in_window < target_width){
      // Add digit to thin time window to current SubSample
      temp_time.push_back(time_in_window / TimeDelta::ns);
      temp_charge.push_back(m_charge[i]);
      temp_PMTid.push_back(m_PMTid[i]);
    } else {
      // Digit outside target window
      // Save current SubSample and rewind to prepare a new one at the overlap position
      SubSample new_sample;
      new_sample.Append(temp_PMTid, temp_time, temp_charge, temp_timestamp);
      split_samples.push_back(new_sample);
      // Reset temporary vectors
      temp_PMTid.clear();
      temp_time.clear();
      temp_charge.clear();
      // Update timestamp
      while ( not (m_timestamp + TimeDelta(m_time.at(i)) - temp_timestamp < target_width) ){
        temp_timestamp += target_stride;
      }
      // Rewind index to cover overlap
      while ( m_timestamp + TimeDelta(m_time.at(i)) - temp_timestamp > TimeDelta(0.) ){
        --i;
        // This will stop when `i` is just outside the new time window
        // Then `i` will get increased by one at the end of the loop
      }
    }
  }
  // Add final SubSample
  SubSample new_sample;
  new_sample.Append(temp_PMTid, temp_time, temp_charge, temp_timestamp);
  split_samples.push_back(new_sample);

  return split_samples;
}

TimeDelta SubSample::AbsoluteDigitTime(int index) const{
  return m_timestamp + TimeDelta(m_time.at(index));
}

bool SubSample::Append(const SubSample& sub)
{
  return Append(sub.m_PMTid, sub.m_time, sub.m_charge, sub.m_timestamp);
}

bool SubSample::Append(const std::vector<int> PMTid, const std::vector<TimeDelta::short_time_t> time, const std::vector<float> charge, const TimeDelta timestamp){
  if (not (PMTid.size() == time.size() && PMTid.size() == charge.size())){
    return false;
  }
  m_PMTid.insert (m_PMTid.end(),  PMTid.begin(),  PMTid.end());
  m_charge.insert(m_charge.end(), charge.begin(), charge.end());

  // If these are the first hits to be added, just use their timestamp
  if (m_time.size() == 0){
    m_timestamp = timestamp;
    m_time.insert(m_time.end(), time.begin(), time.end());
  } else {
    // Need to shift the hit times by the difference of timestamp offsets
    TimeDelta::short_time_t time_shift = (timestamp - m_timestamp) / TimeDelta::ns;
    for (int i=0; i<time.size(); ++i){
        m_time.push_back(time[i] + time_shift);
    }
  }
  return true;
}
