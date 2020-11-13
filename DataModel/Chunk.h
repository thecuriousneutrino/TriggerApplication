#ifndef CHUNK_H
#define CHUNK_H

#include <Hit.h>
#include <vector>
#include <BoostStore.h>

class Chunk{

 public:

  Chunk();
  ~Chunk();

  std::vector<Hit> hits;
  BoostStore tool_output;

};


#endif
