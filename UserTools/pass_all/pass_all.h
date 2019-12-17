#ifndef pass_all_H
#define pass_all_H

#include <string>
#include <iostream>

#include "Tool.h"

class pass_all: public Tool {


 public:

  pass_all();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
