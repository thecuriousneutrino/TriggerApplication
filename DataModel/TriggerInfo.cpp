#include "TriggerInfo.h"

TriggerInfo::TriggerInfo()
{
  Clear();
}

void TriggerInfo::AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, std::vector<float> info){
  AddTrigger(type, TimeDelta(starttime), TimeDelta(endtime), TimeDelta(triggertime),  info);
}

void TriggerInfo::AddTrigger(TriggerType_t type, TimeDelta starttime, TimeDelta endtime, TimeDelta triggertime, std::vector<float> info)
{
  m_type        .push_back(type);
  m_starttime   .push_back(starttime);
  m_endtime     .push_back(endtime);
  m_triggertime .push_back(triggertime);
  m_info        .push_back(info);
  m_N++;
}

void TriggerInfo::AddTriggers(TriggerInfo * in)
{
  for(int i = 0; i < in->m_N; i++) {
    AddTrigger(in->m_type.at(i), in->m_starttime.at(i), in->m_endtime.at(i), in->m_triggertime.at(i), in->m_info.at(i));
  }
}

void TriggerInfo::Clear()
{
  m_N = 0;
  m_type.clear();
  m_starttime.clear();
  m_endtime.clear();
  m_triggertime.clear();
  m_info.clear();
}

void TriggerInfo::SortByStartTime()
{ 
  //algorithm borrowed from WCSimWCDigi::SortArrayByHitTime()
  int i, j;
  TimeDelta save_starttime, save_endtime, save_triggertime;
  TriggerType_t save_type;
  std::vector<float> save_info;
  for (i = 1; i < (int) m_N; ++i) {
    save_type        = m_type[i];
    save_starttime   = m_starttime[i];
    save_endtime     = m_endtime[i];
    save_triggertime = m_triggertime[i];
    save_info        = m_info[i];
    for (j = i; j > 0 && m_starttime[j-1] > save_starttime; j--) {
      m_type       [j] = m_type[j-1];
      m_starttime  [j] = m_starttime[j-1];
      m_endtime    [j] = m_endtime[j-1];
      m_triggertime[j] = m_triggertime[j-1];
      m_info       [j] = m_info[j-1];
    }//j
    m_type       [j] = save_type;
    m_starttime  [j] = save_starttime;
    m_endtime    [j] = save_endtime;
    m_triggertime[j] = save_triggertime;
    m_info       [j] = save_info;
  }//i
}
