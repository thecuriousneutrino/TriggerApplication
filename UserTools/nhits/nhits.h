#ifndef nhits_H
#define nhits_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "GPUFunctions.h"

class nhits: public Tool {


 public:

  nhits();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
