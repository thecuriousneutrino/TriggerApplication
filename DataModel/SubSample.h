#ifndef SUBSAMPLE_H
#define SUBSAMPLE_H

#include <iostream>
#include <vector>

class SubSample{

 public:

  SubSample();
  SubSample(std::vector<int> PMTid,std::vector<float> time)
    {
      assert(m_PMTid.size() == m_time.size());
      m_PMTid=PMTid;
      m_time=time;
      for(unsigned int i = 0; i < m_time.size(); i++) {
	m_time_int[i] = m_time[i];
      }
    }

  SubSample(std::vector<int> PMTid, std::vector<float> time, std::vector<float> charge)
    {
      assert(m_PMTid.size() == m_time.size() && m_PMTid.size() == m_charge.size());
      m_PMTid  = PMTid;
      m_time   = time;
      m_charge = charge;
      for(unsigned int i = 0; i < m_time.size(); i++) {
	m_time_int[i] = m_time[i];
	m_charge_int[i] = m_charge[i];
      }
    }

  std::vector<int> m_PMTid;
  std::vector<float> m_time;
  std::vector<float> m_charge;
  std::vector<int> m_time_int;
  std::vector<int> m_charge_int;

};





#endif
