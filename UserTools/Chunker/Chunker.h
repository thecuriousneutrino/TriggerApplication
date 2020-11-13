#ifndef Chunker_H
#define Chunker_H

#include <string>
#include <iostream>

#include "Tool.h"

struct Chunker_args:Thread_args{

  Chunker_args();
  ~Chunker_args();
  bool busy;
  TimeSlice* time_slice;
  std::map<float,Chunk*>* chunks;
  
};

/**
 * \class Chunker
 *
 * This is a balnk template for a Tool used by the script to generate a new custom tool. Please fill out the descripton and author information.
 *
 * $Author: B.Richards $
 * $Date: 2019/05/28 10:44:00 $
 * Contact: b.richards@qmul.ac.uk
 */
class Chunker: public Tool {


 public:

  Chunker(); ///< Simple constructor
  bool Initialise(std::string configfile,DataModel &data); ///< Initialise Function for setting up Tool resorces. @param configfile The path and name of the dynamic configuration file to read in. @param data A reference to the transient data class used to pass information between Tools.
  bool Execute(); ///< Executre function used to perform Tool perpose. 
  bool Finalise(); ///< Finalise funciton used to clean up resorces.


 private:

  static void Thread(Thread_args* arg);
  Utilities* m_util;
  std::vector<Chunker_args*> args;

  TimeSlice* m_current_slice;
  int m_freethreads;
  int m_chunk_size;
  int m_verbose;

};


#endif
