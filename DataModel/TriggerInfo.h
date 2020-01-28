#ifndef TRIGGERINFO_H
#define TRIGGERINFO_H

#include <iostream>
#include <vector>

#include "WCSimEnumerations.hh"

class TriggerInfo{

 public:

  TriggerInfo();

  void AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, std::vector<float> info);

  void AddTriggers(TriggerInfo * in);

  void Clear() ;

  void SortByStartTime();

  unsigned int m_N;
  std::vector<TriggerType_t> m_type;
  std::vector<double>        m_starttime;
  std::vector<double>        m_endtime;
  std::vector<double>        m_triggertime;
  std::vector<std::vector<float> > m_info;
};





#endif
