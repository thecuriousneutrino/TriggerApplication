#include "nhits.h"

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

  double temp_trigger_search_window;
  double temp_trigger_search_window_step;
  double temp_trigger_save_window_pre;
  double temp_trigger_save_window_post;

  m_variables.Get("trigger_search_window",   temp_trigger_search_window);
  m_variables.Get("trigger_search_window_step",   temp_trigger_search_window_step);
  m_variables.Get("trigger_threshold",            m_trigger_threshold);
  m_variables.Get("pretrigger_save_window",  temp_trigger_save_window_pre);
  m_variables.Get("posttrigger_save_window", temp_trigger_save_window_post);
  m_variables.Get("trigger_od",                   m_trigger_OD);
  m_variables.Get("degrade_CPU",                   m_degrade_CPU);

  m_trigger_search_window = TimeDelta(temp_trigger_search_window);
  m_trigger_save_window_pre = TimeDelta(temp_trigger_save_window_pre);
  m_trigger_save_window_post = TimeDelta(temp_trigger_save_window_post);

  bool adjust_for_noise;
  m_variables.Get("trigger_threshold_adjust_for_noise", adjust_for_noise);
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

  m_ss << "INFO: m_degrade_CPU " << m_degrade_CPU; StreamToLog(INFO);

  if(adjust_for_noise) {
    m_ss << "INFO: Updating the NDigits threshold, from " << m_trigger_threshold
       << " to " << m_trigger_threshold + round(average_occupancy) << std::endl;
    StreamToLog(INFO);
    m_trigger_threshold += round(average_occupancy);
  }

#ifdef GPU
  std::string ParameterFile;

  m_variables.Get("ParameterFile",ParameterFile);

  GPU_daq::nhits_initialize_ToolDAQ(ParameterFile,m_data->IDGeom.size(),temp_trigger_search_window, temp_trigger_search_window_step, m_trigger_threshold, temp_trigger_save_window_pre, temp_trigger_save_window_post);


  m_time_int.reserve(2*(int)average_occupancy);

#endif

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}

bool NHits::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  //do stuff with m_data->Samples

  std::vector<SubSample> & samples = m_trigger_OD ? (m_data->ODSamples) : (m_data->IDSamples);

  m_ss << " Number of data samples " << samples.size();
  StreamToLog(DEBUG1);

  for( std::vector<SubSample>::iterator is=samples.begin(); is!=samples.end(); ++is){
#ifdef GPU

    std::vector<int> trigger_ns;
    std::vector<int> trigger_ts;
    m_time_int.clear();
    for(unsigned int i = 0; i < is->m_time.size(); i++) {
      m_time_int.push_back(is->m_time[i]);
    }
    GPU_daq::nhits_execute(is->m_PMTid, m_time_int, &trigger_ns, &trigger_ts);
    for(int i=0; i<trigger_ns.size(); i++){
      m_data->IDTriggers.AddTrigger(kTriggerNDigits,
                                    TimeDelta(trigger_ts[i]) - m_trigger_save_window_pre + is->m_timestamp,
                                    TimeDelta(trigger_ts[i]) + m_trigger_save_window_post + is->m_timestamp,
                                    TimeDelta(trigger_ts[i]) - m_trigger_save_window_pre + is->m_timestamp,
                                    TimeDelta(trigger_ts[i]) + m_trigger_save_window_post + is->m_timestamp,
                                    TimeDelta(trigger_ts[i]) + is->m_timestamp,
                                    std::vector<float>(1, trigger_ns[i]));

      m_ss << "trigger! time  " << trigger_ts[i] << " nhits " <<  trigger_ns[i]; StreamToLog(INFO);
    }
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

  const unsigned int ndigits = sample->m_time.size();
  m_ss << "DEBUG: NHits::AlgNDigits(). Number of entries in input digit collection: " << ndigits;
  StreamToLog(DEBUG1);

  // Where to store the triggers we find
  TriggerInfo * triggers = m_trigger_OD ? &(m_data->ODTriggers) : &(m_data->IDTriggers);

  // Loop over all digits
  // But we can start with an offset of at least the threhshold to save some time
  int current_digit = std::min(m_trigger_threshold, ndigits);
  int first_digit_in_window = 0;
  for(;current_digit < ndigits; ++current_digit) {
    // Update first digit in trigger window

    if( !m_degrade_CPU ){
      TimeDelta::short_time_t digit_time = sample->m_time.at(current_digit);
      while(TimeDelta(sample->m_time[first_digit_in_window]) < TimeDelta(digit_time) - m_trigger_search_window){
	++first_digit_in_window;
      }
    }else{
      //F. Nova degrade info from float to int to match GPU run
      int digit_time = (int)sample->m_time.at(current_digit);
      while(TimeDelta((int)sample->m_time[first_digit_in_window]) <= TimeDelta(digit_time) - m_trigger_search_window){
	++first_digit_in_window;
      }
    }
   

    // if # of digits in window over threshold, issue trigger
    int n_digits_in_window = current_digit - first_digit_in_window + 1; // +1 because difference is 0 when first digit is the only digit in window

    if( n_digits_in_window > m_trigger_threshold) {
      TimeDelta triggertime = sample->AbsoluteDigitTime(current_digit);
      
      if( m_degrade_CPU )	
	triggertime = ((uint64_t) (triggertime /TimeDelta::ns)) * TimeDelta::ns;
      
      m_ss << "DEBUG: Found NHits trigger in SubSample at " << triggertime;
      StreamToLog(DEBUG2);
      m_ss << "DEBUG: Advancing search by posttrigger_save_window " << m_trigger_save_window_post;
      StreamToLog(DEBUG2);
      while(sample->AbsoluteDigitTime(current_digit) < triggertime + m_trigger_save_window_post){
	++current_digit;
	if (current_digit >= ndigits){
	  // Break if we run out of digits
	  break;
	}
      }
      --current_digit; // We want the last digit *within* post-trigger-window
      int n_digits = current_digit - first_digit_in_window + 1;
      m_ss << "DEBUG: Number of digits between (trigger_time - trigger_search_window) and (trigger_time + posttrigger_save_window):" << n_digits;
      StreamToLog(DEBUG2);
      
      triggers->AddTrigger(kTriggerNDigits,
			   triggertime - m_trigger_save_window_pre + sample->m_timestamp,
			   triggertime + m_trigger_save_window_post + sample->m_timestamp,
			   triggertime - m_trigger_save_window_pre + sample->m_timestamp,
			   triggertime + m_trigger_save_window_post + sample->m_timestamp,
			   triggertime + sample->m_timestamp,
			   std::vector<float>(1, n_digits));
    }
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
