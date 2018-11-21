#ifndef SUBSAMPLE_H
#define SUBSAMPLE_H

#include <iostream>
#include <vector>

class SubSample{

 public:

  SubSample();
  SubSample(std::vector<int> PMTid,std::vector<float> time)
    {
      m_PMTid=PMTid;
      m_time=time;
    }

  SubSample(std::vector<int> PMTid, std::vector<float> time, std::vector<float> charge)
    {
      m_PMTid  = PMTid;
      m_time   = time;
      m_charge = charge;
    }

  std::vector<int> m_PMTid;
  std::vector<float> m_time;
  std::vector<float> m_charge;

};





#endif
