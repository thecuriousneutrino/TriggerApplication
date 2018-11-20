#ifndef TRIGGERINFO_H
#define TRIGGERINFO_H

#include <iostream>
#include <vector>

class TriggerInfo{

 public:

  TriggerInfo()
    {
      Clear();
    }

  void AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, double info)
  {
    m_type        .push_back(type);
    m_starttime   .push_back(starttime);
    m_endtime     .push_back(endtime);
    m_triggertime .push_back(triggertime);
    m_info        .push_back(info);
    m_N++;
  }

  void Clear() 
  {
    m_N = 0;
    m_type.clear();
    m_starttime.clear();
    m_endtime.clear();
    m_triggertime.clear();
    m_info.clear();
  }

  unsigned int m_N;
  std::vector<TriggerType_t> m_type;
  std::vector<double>        m_starttime;
  std::vector<double>        m_endtime;
  std::vector<double>        m_triggertime;
  std::vector<double>        m_info;
};





#endif
