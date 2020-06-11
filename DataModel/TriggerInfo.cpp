#include "TriggerInfo.h"

TriggerInfo::TriggerInfo(){
  Clear();
}

void TriggerInfo::AddTrigger(TriggerType_t type,
			     TimeDelta readout_start_time, TimeDelta readout_end_time,
			     TimeDelta mask_start_time, TimeDelta mask_end_time,
			     TimeDelta trigger_time, std::vector<float> info) {
  m_type              .push_back(type);
  m_readout_start_time.push_back(readout_start_time);
  m_readout_end_time  .push_back(readout_end_time);
  m_mask_start_time   .push_back(mask_start_time);
  m_mask_end_time     .push_back(mask_end_time);
  m_trigger_time      .push_back(trigger_time);
  m_info              .push_back(info);
  m_num_triggers++;
}

void TriggerInfo::AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, std::vector<float> info) {
  AddTrigger(type, TimeDelta(starttime), TimeDelta(endtime), TimeDelta(starttime), TimeDelta(endtime), TimeDelta(triggertime),  info);
}

void TriggerInfo::AddTriggers(TriggerInfo * in) {
  for(int i = 0; i < in->m_num_triggers; i++) {
    AddTrigger(in->m_type.at(i),
	       in->m_readout_start_time.at(i), in->m_readout_end_time.at(i),
	       in->m_mask_start_time.at(i), in->m_mask_end_time.at(i),
	       in->m_trigger_time.at(i), in->m_info.at(i));
  }
}

void TriggerInfo::Clear() {
  m_num_triggers = 0;
  m_type.clear();
  m_readout_start_time.clear();
  m_readout_end_time.clear();
  m_mask_start_time.clear();
  m_mask_end_time.clear();
  m_trigger_time.clear();
  m_info.clear();
}

std::ostream& operator<<(std::ostream& outs, const TriggerInfo& trig){
  outs << trig.m_num_triggers << " triggers stored:" << std::endl;
  for(unsigned int itrig = 0; itrig < trig.m_num_triggers; itrig++) {
    std::cout << "\t" << itrig << "\t"
	      << WCSimEnumerations::EnumAsString(trig.m_type[itrig])
	      << "\tReadout: " << trig.m_readout_start_time[itrig]
	      << "\t to " << trig.m_readout_end_time[itrig]
	      << std::endl;
  }//itrig
  return outs;
}
