#ifndef TIMESLICE_H
#define TIMESLICE_H

#include <map>
#include <Chunk.h>

class TimeSlice{

 public:

  TimeSlice();

  //Tom please add what ever WCSim ID OD mPMT structures etc needed to load in a single WCSim event

  std::map<float,Chunk*> chunks;



};


#endif
