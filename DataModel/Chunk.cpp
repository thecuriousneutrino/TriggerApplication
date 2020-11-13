#include <Chunk.h>

Chunk::Chunk() :  tool_output(false,0) {}

Chunk::~Chunk(){

  hits.clear();

}
