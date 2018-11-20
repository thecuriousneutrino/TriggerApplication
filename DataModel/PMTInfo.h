#ifndef PMTINFO_H
#define PMTINFO_H

#include <iostream>
#include <vector>

class PMTInfo{
 
 public:
  
  PMTInfo(int tubeno, float x, float y, float z) {
    m_tubeno = tubeno;
    m_x = x;
    m_y = y;
    m_z = z;
  }
 
  int m_tubeno;
  float m_x, m_y, m_z;
  
};




#endif
