#include "Stopwatch.h"

#include <sstream>
#include <iostream>
#include <algorithm>

#include "TH1D.h"
#include "TLegend.h"
#include "TCanvas.h"

///////////////////////////////////////////////////
util::Stopwatch::Stopwatch(const char * tool_name)
  : m_tool_name(tool_name)
{
}

///////////////////////////////////////////////////
void util::Stopwatch::Start() {
  m_running = true;
  m_sw.Start();
}

///////////////////////////////////////////////////
util::StopwatchTimes util::Stopwatch::Stop() {
  m_sw.Stop();
  m_running = false;
  StopwatchTimes times;
  times.cpu_time  = m_sw.CpuTime();
  times.real_time = m_sw.RealTime();
  
  //append to the results
  m_results.push_back(times);

  return times;
}

///////////////////////////////////////////////////
std::string util::Stopwatch::Result(std::string method_name, std::string output_file) {
  //If it's running, stop it
  if(m_running)
    this->Stop();

  //get the stats
  double min_cpu  = 9999, max_cpu  = -9999, tot_cpu  = 0;
  double min_real = 9999, max_real = -9999, tot_real = 0;
  double cpu, real;
  const int n = m_results.size();
  for(size_t i = 0; i < n; i++) {
    cpu  = m_results[i].cpu_time;
    real = m_results[i].real_time;
    //increment totals
    tot_cpu  += cpu;
    tot_real += real;
    //find mins
    min_cpu  = cpu  < min_cpu  ? cpu  : min_cpu;
    min_real = real < min_real ? real : min_real;
    //and maxes
    max_cpu  = cpu  > max_cpu  ? cpu  : max_cpu;
    max_real = real > max_real ? real : max_real;
  }//i
  //Create, fill, and save a histogram
  if(output_file.length()) {
    double min = std::min(min_cpu, min_real);
    double max = std::max(max_cpu, max_real);
    TH1D hcpu ("hcpu",  ";Time (s);Runs", sqrt(n), min, max);
    TH1D hreal("hreal", ";Time (s);Runs", sqrt(n), min, max);
    for(size_t i = 0; i < n; i++) {
      hcpu .Fill(m_results[i].cpu_time);
      hreal.Fill(m_results[i].real_time);
    }//i
    TCanvas c;
    hcpu.GetYaxis()->SetRangeUser(0, 1.05 * std::max(hcpu.GetMaximum(), hreal.GetMaximum()));
    hcpu.Draw();
    hreal.SetLineColor(kRed);
    hreal.Draw("SAME");
    TLegend leg(0.8, 0.9, 1, 1);
    leg.AddEntry(&hcpu,   "CPU time", "L");
    leg.AddEntry(&hreal, "Real time", "L");
    leg.Draw();
    c.SaveAs(output_file.c_str());
  }

  //format the output
  std::stringstream ss;
  ss << m_tool_name << "::" << method_name << "() run timing stats:" << std::endl
     << " CPU Time (s): Total = " << tot_cpu << std::endl
     << "             Average = " << tot_cpu / (double)n
     << " over " << n << " runs" << std::endl
     << "  Range of " << min_cpu << " to " << max_cpu << std::endl
     << " Real Time (s): Total = " << tot_real << std::endl
     << "              Average = " << tot_real / (double)n
     << " over " << n << " runs" << std::endl
     << "  Range of " << min_real << " to " << max_real << std::endl;

  //reset
  this->Reset();

  return ss.str();
}

///////////////////////////////////////////////////
void util::Stopwatch::Reset() {
  if(m_running)
    m_sw.Stop();
  m_results.clear();
}

///////////////////////////////////////////////////
