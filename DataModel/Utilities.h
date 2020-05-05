#ifndef UTILITIES_H
#define UTILITIES_H

#include "TimeDelta.h"
#include <sstream>

namespace util {
  ///
  struct Window {
    int m_trigger_num;
    TimeDelta m_start;
    TimeDelta m_end;
    Window() {}
    Window(int trigger_num, TimeDelta start, TimeDelta end) :
      m_trigger_num(trigger_num), m_start(start), m_end(end) {}
  };

  ///
  static bool WindowSorter(const Window & lhs,
			   const Window & rhs) {
    return lhs.m_start < rhs.m_start;
  }

  /// Format messages in the same way as for tools
  void Log(const std::string & message, const int message_level) {
    std::stringstream tmp;
    tmp << "[" << message_level << "] " << message;
    std::cout << tmp.str() << std::endl;
  }

  /// Format messages in the same way as for tools
  void Log(std::stringstream & message, const int message_level) {
    Log(message.str(), message_level);
    message.str("");
  }

  /// Log level enumerations
  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};

} //namespace util

#endif
