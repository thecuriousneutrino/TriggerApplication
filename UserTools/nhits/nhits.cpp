#include "nhits.h"

const int nhits::kALongTime = 1E6; // ns = 1ms. event time

nhits::nhits():Tool(){}


bool nhits::Initialise(std::string configfile, DataModel &data){



  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

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

  float dark_rate = -99;
  m_variables.Get("dark_rate",dark_rate);
  if(!dark_rate)
    dark_rate = 0;
 
  m_variables.Get("trigger_search_window",        fTriggerSearchWindow);
  m_variables.Get("trigger_search_window_step",   fTriggerSearchWindowStep);
  m_variables.Get("trigger_threshold",            fTriggerThreshold);
  m_variables.Get("pretrigger_save_window",       fTriggerSaveWindowPre);
  m_variables.Get("posttrigger_save_window",      fTriggerSaveWindowPost);
  m_variables.Get("trigger_od",                   fTriggerOD);
  return true;
}


bool nhits::Execute(){
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

  return true;
}

void nhits::AlgNDigits(const SubSample * sample)
{
  //we will try to find triggers
  //loop over PMTs, and Digits in each PMT.  If ndigits > Threshhold in a time window, then we have a trigger

  const unsigned int ndigits = sample->m_charge.size();
  std::cout << "WCSimWCTriggerBase::AlgNDigits. Number of entries in input digit collection: " << ndigits << std::endl;
  
  //first thing: what to loop over
  int temp_total_pe = 0;
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
    temp_total_pe += sample->m_charge.at(idigit);
  }//loop over Digits
  int window_start_time = firsthit;
  int window_end_time   = lasthit - fTriggerSearchWindow + fTriggerSearchWindowStep;
  std::cout << "Found first/last hits. Looping from " << window_start_time
	    << " to " << window_end_time 
	    << " in steps of " << fTriggerSearchWindowStep
	    << std::endl;

  std::cout << "WCSimWCTriggerBase::AlgNDigits. " << temp_total_pe << " total p.e. input" << std::endl;
  
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
      //int charge = sample->m_charge.at(idigit);
      int digit_time = sample->m_time.at(idigit);
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
      m_data->IDTriggers.AddTrigger(kTriggerNDigits,
				    triggertime - fTriggerSaveWindowPre, 
				    triggertime + fTriggerSaveWindowPost,
				    triggertime,
				    n_digits);
    }

    if(n_digits)
      std::cout << n_digits << " digits found in 200nsec trigger window ["
		<< window_start_time << ", " << window_start_time + fTriggerSearchWindow
		<< "]. Threshold is: " << fTriggerThreshold << std::endl;

    //move onto the next go through the timing loop
    if(triggerfound) {
      window_start_time = triggertime + fTriggerSaveWindowPost;
    }//triggerfound
    else {
      window_start_time += fTriggerSearchWindowStep;
    }

  }//sliding trigger window while loop
  
  std::cout << "Found " << m_data->IDTriggers.m_N << " NDigit triggers" << std::endl;
}

bool nhits::Finalise(){


#ifdef GPU
  GPU_daq::nhits_finalize();
#endif


  return true;
}
