#ifndef UTILITIES_H
#define UTILITIES_H

#include "TimeDelta.h"
#include <sstream>

namespace util {
  ///Contains the start and end of a time window, along with an ID (nominally trigger number)
  struct Window {
    int m_trigger_num;
    TimeDelta m_start;
    TimeDelta m_end;
    Window() {}
    Window(int trigger_num, TimeDelta start, TimeDelta end) :
      m_trigger_num(trigger_num), m_start(start), m_end(end) {}
  };

  ///When sorting Window structs, sort by the start time
  static bool WindowSorter(const Window & lhs,
			   const Window & rhs) {
    return lhs.m_start < rhs.m_start;
  }

  /// Check if a file exists
  bool FileExists(std::string pathname, std::string filename);

  /// Format messages in the same way as for tools
  void Log(const std::string & message, const int message_level);

  /// Format messages in the same way as for tools
  void Log(std::stringstream & message, const int message_level);

  /// Log level enumerations
  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};

} //namespace util

#endif
