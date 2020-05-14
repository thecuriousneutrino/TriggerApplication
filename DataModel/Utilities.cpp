#include "Utilities.h"

#include <unistd.h>

//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
bool util::FileExists(std::string pathname, std::string filename) {
  std::string filepath = pathname + "/" + filename;
  bool exists = access(filepath.c_str(), F_OK) != -1;
  if(!exists) {
    std::stringstream ss;
    ss << "FATAL: " << filepath << " not found or inaccessible";
    util::Log(ss.str(), util::FATAL);
    return false;
  }
  return true;
}
//////////////////////////////////////////////////////////////////
void util::Log(const std::string & message, const int message_level) {
  std::stringstream tmp;
  tmp << "[" << message_level << "] " << message;
  std::cout << tmp.str() << std::endl;
}
//////////////////////////////////////////////////////////////////
void util::Log(std::stringstream & message, const int message_level) {
  Log(message.str(), message_level);
  message.str("");
}
//////////////////////////////////////////////////////////////////
