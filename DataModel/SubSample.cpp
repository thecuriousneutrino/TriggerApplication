#include "SubSample.h"

#include <cassert>

SubSample::SubSample(std::vector<int> PMTid,std::vector<float> time)
{
  assert(PMTid.size() == time.size());
  m_PMTid=PMTid;
  m_time=time;
  //fill the int versions
  m_time_int.resize(m_time.size());
  for(unsigned int i = 0; i < m_time.size(); i++) {
    m_time_int[i] = m_time[i];
  }
}

SubSample::SubSample(std::vector<int> PMTid, std::vector<float> time, std::vector<float> charge)
{
  assert(PMTid.size() == time.size() && PMTid.size() == charge.size());
  m_PMTid  = PMTid;
  m_time   = time;
  m_charge = charge;
  //fill the int versions
  m_time_int.resize(m_time.size());
  m_charge_int.resize(m_time.size());
  for(unsigned int i = 0; i < m_time.size(); i++) {
    m_time_int[i] = m_time[i];
    m_charge_int[i] = m_charge[i];
  }
}
