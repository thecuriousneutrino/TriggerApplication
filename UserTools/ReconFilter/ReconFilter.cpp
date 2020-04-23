#include "ReconFilter.h"

ReconFilter::ReconFilter():Tool(){}


bool ReconFilter::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("ReconFilter") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  //parameters determining what to name the (pre)filtered RecoInfo objects
  if(!m_variables.Get("input_filter_name", fInputFilterName)) {
    Log("INFO: input_filter_name not given. Using ALL", WARN, m_verbose);
    fInputFilterName = "ALL";
  }
  if(!m_variables.Get("output_filter_name", fOutputFilterName)) {
    Log("WARN: output_filter_name not given. Using TEMP", WARN, m_verbose);
    fOutputFilterName = "TEMP";
  }
  fInFilter  = m_data->GetFilter(fInputFilterName, false);
  if(!fInFilter) {
    ss << "FATAL: no filter named " << fInputFilterName << " found. Returning false";
    StreamToLog(FATAL);
    return false;
  }
  fOutFilter = m_data->GetFilter(fOutputFilterName, true);

  if(fOutFilter == &(m_data->RecoInfo)) {
    Log("ERROR: Cannot use the full RecoInfo object to store filtered events", ERROR, m_verbose);
    return false;
  }
  if(fInFilter == fOutFilter) {
    Log("ERROR: Can't use the same filter for input and output. TODO add ReconInfo::RemoveRecon() methods and change logic here in ReconFilter to deal with it", ERROR, m_verbose);
    return false;
  }
  if(fOutFilter->GetNRecons()) {
    Log("ERROR: output_filter_name must be unique (needs to be a blank canvas). TODO  add ReconInfo::RemoveRecon() methods and change logic here in ReconFilter to deal with it", ERROR, m_verbose);
    return false;
  }
    
  //parameters determining which reconstructed events to use
  std::string reconstruction_algorithm_str;
  fReconstructionAlgorithm = kReconUndefined;
  m_variables.Get("reconstruction_algorithm", reconstruction_algorithm_str);
  fReconstructionAlgorithm = ReconInfo::ReconstructerFromString(reconstruction_algorithm_str);
  if(fReconstructionAlgorithm == kReconUndefined) {
    Log("ERROR: The reconstruction_algorithm parameter you have chosen is not defined. Please choose a valid option", ERROR, m_verbose);
    return false;
  }
  if(!m_variables.Get("fMinReconLikelihood", fMinReconLikelihood)) {
    fMinReconLikelihood = 0;
    Log("WARN: No fMinReconLikelihood parameter found. Using a value of 0", WARN, m_verbose);
  }
  if(!m_variables.Get("fMinReconTimeLikelihood", fMinReconTimeLikelihood)) {
    fMinReconTimeLikelihood = 0;
    Log("WARN: No fMinReconTimeLikelihood parameter found. Using a value of 0", WARN, m_verbose);
  }
  if(!m_variables.Get("max_r_pos", fMaxRPos_cm)) {
    fMaxRPos_cm = 3500;
    Log("WARN: No max_r_pos parameter found. Using a value of 3500 (cm)", WARN, m_verbose);
  }
  if(!m_variables.Get("max_z_pos", fMaxZPos_cm)) {
    fMaxZPos_cm = 2700;
    Log("WARN: No max_z_pos parameter found. Using a value of 2700 (cm)", WARN, m_verbose);
  }

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}

bool ReconFilter::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  const int N = m_data->RecoInfo.GetNRecons();
  for(int irecon = 0; irecon < N; irecon++) {
    //skip events reconstructed with the wrong algorithm
    if(fInFilter->GetReconstructer(irecon) != fReconstructionAlgorithm)
      continue;

    //skip events with poor likelihoods
    if(fInFilter->GetGoodnessOfFit(irecon) < fMinReconLikelihood)
      continue;
    if(fInFilter->GetGoodnessOfTimeFit(irecon) < fMinReconTimeLikelihood)
      continue;
    
    //get the vertex position
    Pos3D pos = fInFilter->GetVertex(irecon);
    //skip events that are reconstructed too close to the wall
    if(pos.R() > fMaxRPos_cm)
      continue;
    if(abs(pos.z) > fMaxZPos_cm)
      continue;

    //the event has passed the filter. Save it in the output list
    fOutFilter->AddReconFrom(fInFilter, irecon);
  }//irecon 

  ss << "INFO: ReconFilter has reduced number of reconstructed events from "
     << fInFilter ->GetNRecons() << " (" << fInputFilterName  << ") to "
     << fOutFilter->GetNRecons() << " (" << fOutputFilterName << ")";
  StreamToLog(INFO);

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool ReconFilter::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
