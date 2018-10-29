#ifndef WCSimReader_H
#define WCSimReader_H

#include <string>
#include <iostream>

#include "Tool.h"

class WCSimReader: public Tool {


 public:

  WCSimReader();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
