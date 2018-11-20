#ifndef SUBSAMPLE_H
#define SUBSAMPLE_H

#include <iostream>
#include <vector>

class SubSample{

 public:

  SubSample();
  SubSample(std::vector<int> PMTid,std::vector<int> time){
    m_PMTid=PMTid;
    m_time=time;
  }
  SubSample(std::vector<int> PMTid, std::vector<int> time, std::vector<int> charge) {
    m_PMTid  = PMTid;
    m_time   = time;
    m_charge = charge;
  }

  std::vector<int> m_PMTid;
  std::vector<int> m_time;
  std::vector<int> m_charge;

};





#endif
