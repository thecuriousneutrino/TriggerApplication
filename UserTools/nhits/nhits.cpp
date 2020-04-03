#include "nhits.h"

const int nhits::kALongTime = 1E6; // ns = 1ms. event time

nhits::nhits():Tool(){}


bool nhits::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("nhits") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  m_data->triggeroutput=false;
  
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;
  

  m_variables.Get("PMTFile",PMTFile);
  m_variables.Get("DetectorFile",DetectorFile);
  m_variables.Get("ParameterFile",ParameterFile);
  
  //  gpu_daq_initialize(PMTFile,DetectorFile,ParameterFile);

#ifdef GPU
  //  GPU_daq::nhits_initialize();
  GPU_daq::nhits_initialize_ToolDAQ(PMTFile,DetectorFile,ParameterFile);
#endif

  // can acess variables directly like this and would be good if you could impliment in your code

  m_variables.Get("trigger_search_window",        fTriggerSearchWindow);
  m_variables.Get("trigger_search_window_step",   fTriggerSearchWindowStep);
  m_variables.Get("trigger_threshold",            fTriggerThreshold);
  m_variables.Get("pretrigger_save_window",       fTriggerSaveWindowPre);
  m_variables.Get("posttrigger_save_window",      fTriggerSaveWindowPost);
  m_variables.Get("trigger_od",                   fTriggerOD);

  bool adjust_for_noise;
  m_variables.Get("trigger_threshold_adjust_for_noise", adjust_for_noise);
  if(adjust_for_noise) {
    int npmts = fTriggerOD ? m_data->ODNPMTs : m_data->IDNPMTs;
    double dark_rate_kHZ = fTriggerOD ? m_data->ODPMTDarkRate : m_data->IDPMTDarkRate;
    double trigger_window_seconds = fTriggerSearchWindow * 1E-9;
    double dark_rate_Hz = dark_rate_kHZ * 1000;
    double average_occupancy = dark_rate_Hz * trigger_window_seconds * npmts;

    ss << "INFO: Average number of PMTs in detector active in a " << fTriggerSearchWindow
       << "ns window with a dark noise rate of " << dark_rate_kHZ
       << "kHz is " << average_occupancy
       << " (" << npmts << " total PMTs)";
    StreamToLog(INFO);
    ss << "INFO: Updating the NDigits threshold, from " << fTriggerThreshold
       << " to " << fTriggerThreshold + round(average_occupancy) << std::endl;
    StreamToLog(INFO);
    fTriggerThreshold += round(average_occupancy);    
  }

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, verbose);

  return true;
}


bool nhits::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  int the_output;

  //do stuff with m_data->Samples

  std::vector<SubSample> & samples = fTriggerOD ? (m_data->ODSamples) : (m_data->IDSamples);

  printf(" qqq data samples size %d \n", samples.size());

  for( std::vector<SubSample>::const_iterator is=samples.begin(); is!=samples.end(); ++is){
#ifdef GPU   
  //  the_output =   GPU_daq::nhits_execute();
  the_output =   GPU_daq::nhits_execute(is->m_PMTid, is->m_time);
  printf(" qqq qqq look at %d of size %d \n", is - samples.begin(), samples.size());
#else
  AlgNDigits(&(*is));
#endif
  }

  //  the_output = CUDAFunction(samples.at(0).m_PMTid, samples.at(0).m_time);
  m_data->triggeroutput=(bool)the_output;

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}

void nhits::AlgNDigits(const SubSample * sample)
{
  //we will try to find triggers
  //loop over PMTs, and Digits in each PMT.  If ndigits > Threshhold in a time window, then we have a trigger

  const unsigned int ndigits = sample->m_charge.size();
  ss << "DEBUG: nhits::AlgNDigits(). Number of entries in input digit collection: " << ndigits;
  StreamToLog(DEBUG1);
  
  TriggerInfo * Triggers = fTriggerOD ? &(m_data->ODTriggers) : &(m_data->IDTriggers);

  //Loop over each digit
  float firsthit = +nhits::kALongTime;
  float lasthit  = -nhits::kALongTime;
  for(unsigned int idigit = 0; idigit < ndigits; idigit++) {
    float digit_time = sample->m_time.at(idigit);
    //get the time of the last hit (to make the loop shorter)
    if(digit_time > lasthit)
      lasthit = digit_time;
    if(digit_time < firsthit)
      firsthit = digit_time;
  }//loop over Digits
  int window_start_time = firsthit;
  window_start_time -= window_start_time % 5;
  int window_end_time   = lasthit - fTriggerSearchWindow + fTriggerSearchWindowStep;
  window_end_time -= window_end_time % 5;
  ss << "DEBUG: Found first/last hits. Looping from " << window_start_time
     << " to " << window_end_time 
     << " in steps of " << fTriggerSearchWindowStep;
  StreamToLog(DEBUG1);

  std::vector<float> digit_times;

  // the upper time limit is set to the final possible full trigger window
  while(window_start_time <= window_end_time) {
    int n_digits = 0;
    float triggertime; //save each digit time, because the trigger time is the time of the first hit above threshold
    bool triggerfound = false;
    digit_times.clear();
    
    //Loop over each digit
    for(unsigned int idigit = 0; idigit < ndigits; idigit++) {
      //int tube   = sample->m_PMTid.at(idigit);
      //float charge = sample->m_charge.at(idigit);
      float digit_time = sample->m_time.at(idigit);
      //hit in trigger window?
      if(digit_time >= window_start_time && digit_time <= (window_start_time + fTriggerSearchWindow)) {
	n_digits++;
	digit_times.push_back(digit_time);
      }
    }//loop over Digits

    //if over threshold, issue trigger
    if(n_digits > fTriggerThreshold) {
      //The trigger time is the time of the first hit above threshold
      std::sort(digit_times.begin(), digit_times.end());
      triggertime = digit_times[fTriggerThreshold];
      triggertime -= (int)triggertime % 5;
      triggerfound = true;
      Triggers->AddTrigger(kTriggerNDigits,
			   triggertime - fTriggerSaveWindowPre, 
			   triggertime + fTriggerSaveWindowPost,
			   triggertime,
			   std::vector<float>(1, n_digits));
    }

    if(n_digits)
      ss << "DEBUG: " << n_digits << " digits found in 200nsec trigger window ["
	 << window_start_time << ", " << window_start_time + fTriggerSearchWindow
	 << "]. Threshold is: " << fTriggerThreshold;
    StreamToLog(DEBUG2);

    //move onto the next go through the timing loop
    if(triggerfound) {
      window_start_time = triggertime + fTriggerSaveWindowPost;
      ss << "INFO: nhits trigger found at time " << triggertime
	 << " with " << n_digits << " digits in the decision window";
      StreamToLog(INFO);
    }//triggerfound
    else {
      window_start_time += fTriggerSearchWindowStep;
    }

  }//sliding trigger window while loop
  
  ss << "INFO: Found " << Triggers->m_N << " NDigit trigger(s) from " << (fTriggerOD ? "OD" : "ID");
  StreamToLog(INFO);
}

bool nhits::Finalise(){

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, verbose);
    m_stopwatch->Start();
  }

#ifdef GPU
  GPU_daq::nhits_finalize();
#endif

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, verbose);
    delete m_stopwatch;
  }

  return true;
}
