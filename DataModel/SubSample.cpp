#include "SubSample.h"
#include "Utilities.h"

#include <algorithm>

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
  m_first_unique = 0;
  //set the trigger info
  const std::vector<int> empty;
  m_trigger_readout_windows.assign(m_time.size(), empty);
  m_masked.assign(m_time.size(), false);
  m_start_trigger = 0;
}

void SubSample::SortByTime(){
  //algorithm borrowed from WCSimWCDigi::SortArrayByHitTime()
  int i, j;
  TimeDelta::short_time_t save_time;
  int save_PMTid;
  double save_charge;
  std::vector<int> save_triggers;
  bool save_masked;


  for (i = 1; i < m_PMTid.size(); ++i) {
    save_time       = m_time[i];
    save_PMTid      = m_PMTid[i];
    save_charge     = m_charge[i];
    save_triggers   = m_trigger_readout_windows[i];
    save_masked     = m_masked[i];
    for (j = i; j > 0 && m_time[j-1] > save_time; j--) {
      m_time[j]     = m_time[j-1];
      m_PMTid[j]    = m_PMTid[j-1];
      m_charge[j]   = m_charge[j-1];
      m_trigger_readout_windows[j] = m_trigger_readout_windows[j-1];
      m_masked[j]   = m_masked[j-1];
    }//j
    m_time[j]     = save_time;
    m_PMTid[j]    = save_PMTid;
    m_charge[j]   = save_charge;
    m_trigger_readout_windows[j] = save_triggers;
    m_masked[j]   = save_masked;
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
  int ihit_first_unique = 0, ihit_first = 0;
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
      new_sample.m_first_unique = ihit_first_unique - ihit_first;
      split_samples.push_back(new_sample);
      ihit_first_unique = i;
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
      ihit_first = i + 1;
    }
  }//i (loop over m_time)
  // Add final SubSample
  SubSample new_sample;
  new_sample.Append(temp_PMTid, temp_time, temp_charge, temp_timestamp);
  new_sample.m_first_unique = ihit_first_unique - ihit_first;
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

  //set the trigger info
  const std::vector<int> empty;
  m_trigger_readout_windows.insert(m_trigger_readout_windows.end(),
				   m_time.size() - m_trigger_readout_windows.size()
				   , empty);
  m_masked.insert(m_masked.end(),
		  m_time.size() - m_masked.size(),
		  false);
  m_start_trigger = 0;

  return true;
}

void SubSample::TellMeAboutTheTriggers(const TriggerInfo & triggers, const int verbose) {

  //loop over triggers to get the start/end of the readout & mask windows
  const unsigned int num_triggers = triggers.m_num_triggers;
  std::vector<util::Window> readout_windows(num_triggers - m_start_trigger);
  std::vector<util::Window> mask_windows(num_triggers - m_start_trigger);
  if(util::DEBUG2 <= verbose)
    util::Log("DEBUG: Trigger times before sorting:", util::DEBUG1);
  std::stringstream ss;
  for(unsigned int itrigger = m_start_trigger; itrigger < num_triggers; itrigger++) {
    util::Window readout(itrigger,
			 triggers.m_readout_start_time[itrigger],
			 triggers.m_readout_end_time[itrigger]);
    readout_windows[itrigger - m_start_trigger] = readout;
    util::Window mask(itrigger,
		      triggers.m_mask_start_time[itrigger],
		      triggers.m_mask_end_time[itrigger]);
    mask_windows[itrigger - m_start_trigger] = mask;
    if(util::DEBUG2 <= verbose) {
      ss << "DEBUG: Trigger: " << itrigger
	 << "\tReadout windows: " << readout_windows[itrigger].m_start
	 << "\t" << readout_windows[itrigger].m_end
	 << "\tMask windows: " << mask_windows[itrigger].m_start
	 << "\t" << mask_windows[itrigger].m_end;
      util::Log(ss, util::DEBUG2);
    }//DEBUG1
  }//itrigger

  //iterate m_start_trigger, so we don't try masking hits from the same triggers next time this is called
  m_start_trigger += num_triggers;

  //sort the readout windows.
  // Done in reverse order, so we can pop them from the
  //  end of the vector they've gone out of range in the
  //  loop over hits (hits are also time sorted)
  std::sort(readout_windows.rbegin(), readout_windows.rend(), util::WindowSorter);
  std::sort(mask_windows.rbegin(), mask_windows.rend(), util::WindowSorter);

  if(util::DEBUG1 <= verbose) {
    util::Log("DEBUG: Trigger times after sorting:", util::DEBUG1);
    for(unsigned int itrigger = 0; itrigger < readout_windows.size(); itrigger++) {
      ss << "DEBUG: Trigger: " << itrigger
	 << "\tReadout windows: " << readout_windows[itrigger].m_start
	 << "\t" << readout_windows[itrigger].m_end
	 << "\tMask windows: " << mask_windows[itrigger].m_start
	 << "\t" << mask_windows[itrigger].m_end;
      util::Log(ss, util::DEBUG1);
    }//itrigger
  }//DEBUG1

  //ensure the hits are sorted in time
  if(!IsSortedByTime())
    SortByTime();

  //loop over hits
  size_t n_hits = m_time.size();
  TimeDelta hit_time;
  for(size_t ihit = 0; ihit < n_hits; ihit++) {
    hit_time = AbsoluteDigitTime(ihit);
    //Is the hit in this readout window?
    for(std::vector<util::Window>::reverse_iterator it = readout_windows.rbegin();
	it != readout_windows.rend(); ++it) {
      if(util::DEBUG3 <= verbose) {
	ss << "DEBUG: READOUT " << (*it).m_start << "\t" << (*it).m_end << "\t" << (*it).m_trigger_num;
	util::Log(ss, util::DEBUG3);
      }//DEBUG3
      //the trigger time is later than this hit
      if(hit_time < (*it).m_start) {
	if(util::DEBUG2 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " earlier than all subsequent trigger readouts";
	  util::Log(ss, util::DEBUG2);
	}//DEBUG2
	break;
      }
      //the trigger time is earlier than this hit
      else if (hit_time > (*it).m_end) {
	if(util::DEBUG2 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " later than last trigger window. Removing last window from readout vector";
	  util::Log(ss, util::DEBUG2);
	}//DEBUG2
	readout_windows.pop_back();
      }
      //the hit is in this trigger
      else {
	if(util::DEBUG3 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " in trigger readout window " << (*it).m_trigger_num;
	  util::Log(ss, util::DEBUG3);
	}//DEBUG3
	m_trigger_readout_windows[ihit].push_back((*it).m_trigger_num);
      }
    }//readout_windows
    //Is the hit in this mask window?
    for(std::vector<util::Window>::reverse_iterator it = mask_windows.rbegin();
	it != mask_windows.rend(); ++it) {
      if(util::DEBUG3 <= verbose) {
	ss << "DEBUG: MASK " << (*it).m_start << "\t" << (*it).m_end << "\t" << (*it).m_trigger_num;
	util::Log(ss, util::DEBUG3);
      }//DEBUG3
      //the trigger time is later than this hit
      if(hit_time < (*it).m_start) {
	if(util::DEBUG2 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " earlier than all subsequent trigger masks";
	  util::Log(ss, util::DEBUG2);
	}//DEBUG2
	break;
      }
      //the trigger time is earlier than this hit
      else if (hit_time > (*it).m_end) {
	if(util::DEBUG2 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " later than last trigger window. Removing last window from mask vector";
	  util::Log(ss, util::DEBUG2);
	}//DEBUG2
	mask_windows.pop_back();
      }
      //the hit is in this trigger
      else {
	if(util::DEBUG3 <= verbose) {
	  ss << "DEBUG: Hit time " << hit_time << " in trigger mask window " << (*it).m_trigger_num;
	  util::Log(ss, util::DEBUG3);
	}//DEBUG3
      }
    }//mask_windows
  }//ihit
}
