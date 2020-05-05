#include "nhits.h"

#include <deque>

NHits::NHits():Tool(){}

bool NHits::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("nhits") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

#ifdef GPU
  // TODO: The geometry should be taken from the DataModel, not from a separate file
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;

  m_variables.Get("PMTFile",PMTFile);
  m_variables.Get("DetectorFile",DetectorFile);
  m_variables.Get("ParameterFile",ParameterFile);

  GPU_daq::nhits_initialize_ToolDAQ(PMTFile,DetectorFile,ParameterFile);
#endif

  double temp_m_trigger_search_window;
  double temp_m_trigger_save_window_pre;
  double temp_m_trigger_save_window_post;
  double temp_m_trigger_mask_window_pre;
  double temp_m_trigger_mask_window_post;

  m_variables.Get("trigger_search_window",   temp_m_trigger_search_window);
  m_variables.Get("trigger_threshold",            m_trigger_threshold);
  m_variables.Get("pretrigger_save_window",  temp_m_trigger_save_window_pre);
  m_variables.Get("posttrigger_save_window", temp_m_trigger_save_window_post);
  m_variables.Get("trigger_od",                   m_trigger_OD);

  m_trigger_search_window    = TimeDelta(temp_m_trigger_search_window);
  m_trigger_save_window_pre  = TimeDelta(temp_m_trigger_save_window_pre);
  m_trigger_save_window_post = TimeDelta(temp_m_trigger_save_window_post);

  //Set the masks to sensible values
  //pretrigger
  if(!m_variables.Get("pretrigger_mask_window",  temp_m_trigger_mask_window_pre)) {
    m_ss << "WARN: pretrigger_mask_window parameter not given. Setting it to pretrigger_save_window value: " << temp_m_trigger_save_window_pre;
    StreamToLog(WARN);
    m_trigger_mask_window_pre = TimeDelta(temp_m_trigger_save_window_pre);
  } else if(temp_m_trigger_mask_window_pre > temp_m_trigger_save_window_pre) {
    m_ss << "WARN: pretrigger_mask_window parameter value: " << temp_m_trigger_mask_window_pre 
	 << " larger than pretrigger_save_window value: " << temp_m_trigger_save_window_pre
	 << " Setting it to pretrigger_save_window value";
    StreamToLog(WARN);
    m_trigger_mask_window_pre = TimeDelta(temp_m_trigger_save_window_pre);
  } else
    m_trigger_mask_window_pre  = TimeDelta(temp_m_trigger_mask_window_pre);
  //posttrigger
  if(!m_variables.Get("posttrigger_mask_window", temp_m_trigger_mask_window_post)) {
    m_ss << "WARN: posttrigger_mask_window parameter not given. Setting it to posttrigger_save_window value: " << temp_m_trigger_save_window_post;
    StreamToLog(WARN);
    m_trigger_mask_window_post = TimeDelta(temp_m_trigger_save_window_post);
  } else if(temp_m_trigger_mask_window_post > temp_m_trigger_save_window_post) {
    m_ss << "WARN: posttrigger_mask_window parameter value: " << temp_m_trigger_mask_window_post 
	 << " larger than posttrigger_save_window value: " << temp_m_trigger_save_window_post
	 << " Setting it to posttrigger_save_window value";
    StreamToLog(WARN);
    m_trigger_mask_window_post = TimeDelta(temp_m_trigger_save_window_post);
  } else
    m_trigger_mask_window_post  = TimeDelta(temp_m_trigger_mask_window_post);

  bool adjust_for_noise;
  m_variables.Get("trigger_threshold_adjust_for_noise", adjust_for_noise);
  if(adjust_for_noise) {
    int npmts = m_trigger_OD ? m_data->ODNPMTs : m_data->IDNPMTs;
    double dark_rate_kHZ = m_trigger_OD ? m_data->ODPMTDarkRate : m_data->IDPMTDarkRate;
    double trigger_window_seconds = m_trigger_search_window / TimeDelta::s;
    double dark_rate_Hz = dark_rate_kHZ * 1000;
    double average_occupancy = dark_rate_Hz * trigger_window_seconds * npmts;

    m_ss << "INFO: Average number of PMTs in detector active in a " << m_trigger_search_window
       << "ns window with a dark noise rate of " << dark_rate_kHZ
       << "kHz is " << average_occupancy
       << " (" << npmts << " total PMTs)";
    StreamToLog(INFO);
    m_ss << "INFO: Updating the NDigits threshold, from " << m_trigger_threshold
       << " to " << m_trigger_threshold + round(average_occupancy) << std::endl;
    StreamToLog(INFO);
    m_trigger_threshold += round(average_occupancy);
  }

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}

bool NHits::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  //do stuff with m_data->Samples

  std::vector<SubSample> & samples = m_trigger_OD ? (m_data->ODSamples) : (m_data->IDSamples);

  m_ss << " qqq Number of data samples " << samples.size();
  StreamToLog(DEBUG1);

  for( std::vector<SubSample>::iterator is=samples.begin(); is!=samples.end(); ++is){
#ifdef GPU
    GPU_daq::nhits_execute(is->m_PMTid, is->m_time);
    m_ss << " qqq qqq Look at " << is - samples.begin();
    StreamToLog(DEBUG1);
#else
    // Make sure digit times are ordered in time
    is->SortByTime();
    AlgNDigits(&(*is));
#endif
  }//loop over SubSamples

  //Now we have all the triggers, get the SubSample to determine
  // - which trigger readout windows each hit is associated with
  // - which hits should be masked from future triggers
  for( std::vector<SubSample>::iterator is=samples.begin(); is!=samples.end(); ++is) {
    (*is).TellMeAboutTheTriggers(m_data->IDTriggers, m_verbose);
  }//loop over SubSamples

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}

void NHits::AlgNDigits(const SubSample * sample)
{
  //we will try to find triggers
  //loop over PMTs, and Digits in each PMT.  If ndigits > Threshhold in a time window, then we have a trigger

  const unsigned int n_hits = sample->m_time.size();
  m_ss << "DEBUG: NHits::AlgNDigits(). Number of entries in input digit collection: " << n_hits;
  StreamToLog(DEBUG1);

  // Where to store the triggers we find
  TriggerInfo * triggers = m_trigger_OD ? &(m_data->ODTriggers) : &(m_data->IDTriggers);

  // Loop over all digits
  std::deque<TimeDelta> times;
  TimeDelta hit_time;
  for(int idigit = 0; idigit < n_hits; idigit++) {
    // Skip if the current digit should be ignored
    if(sample->m_masked[idigit]) continue;

    // Add the current digit to the back of the queue
    hit_time = sample->m_time[idigit];
    times.push_back(hit_time);

    // Remove any digits that are at the start of the queue,
    //  and no longer in the trigger search window
    for(std::deque<TimeDelta>::iterator it = times.begin();
	  it != times.end(); ++it) {
      if(*it < hit_time - m_trigger_search_window) {
	times.pop_front();
	m_ss << "DEBUG: Removing hit with time " << *it
	     << " from times deque. Search window starts at "
	     << hit_time - m_trigger_search_window;
	StreamToLog(DEBUG3);
      }
      break;
    }

    // if # of digits in window over threshold, issue trigger
    const int n_digits_in_search_window = times.size();
    if(n_digits_in_search_window > m_trigger_threshold) {
      const TimeDelta triggertime = sample->AbsoluteDigitTime(idigit);
      m_ss << "DEBUG: Found NHits trigger in SubSample at " << triggertime;
      StreamToLog(DEBUG2);
      m_ss << "DEBUG: Advancing search by posttrigger_save_window " << m_trigger_save_window_post;
      StreamToLog(DEBUG2);
      while(sample->AbsoluteDigitTime(idigit) <
	    triggertime + m_trigger_save_window_post){
        ++idigit;
        if (idigit >= n_hits){
          // Break if we run out of digits
          break;
        }
      }//advance to end of post trigger window
      idigit--; // We want the last digit *within* post-trigger-window (because we're looping over this in the for loop, so the first digit into the deque will be the first *after* post-trigger-window)
      m_ss << "DEBUG: Number of digits in trigger search window: " << n_digits_in_search_window;
      StreamToLog(DEBUG2);

      triggers->AddTrigger(kTriggerNDigits,
                           triggertime - m_trigger_save_window_pre,
                           triggertime + m_trigger_save_window_post,
			   triggertime - m_trigger_mask_window_pre,
			   triggertime + m_trigger_mask_window_post,
                           triggertime,
                           std::vector<float>(1, n_digits_in_search_window));

      //clear the deque
      times.clear();
    }//trigger found
  }//loop over Digits
  
  m_ss << "INFO: Found " << triggers->m_num_triggers << " NDigit trigger(s) from " << (m_trigger_OD ? "OD" : "ID");
  StreamToLog(INFO);
}

bool NHits::Finalise(){

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

#ifdef GPU
  GPU_daq::nhits_finalize();
#endif

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
