#include "WCSimReader.h"

#include "TFile.h"
#include "TVectorT.h"

WCSimReader::WCSimReader():Tool(){}


bool WCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  m_data= &data;

  //config reading
  if(! m_variables.Get("nevents",  m_n_events) ) {
    Log("WARN: nevents configuration not found. Reading all events", WARN, m_verbose);
    m_n_events = -1;
  }
  if(! (m_variables.Get("infile",   m_input_filename) !=
	m_variables.Get("filelist", m_input_filelist))) {
    Log("ERROR: You must use exactly one of the following options: infile filelist", ERROR, m_verbose);
    return false;
  }

  m_ss << "INFO: m_n_events  \t" << m_n_events;
  StreamToLog(INFO);
  if(m_input_filename.size())
    m_ss << "INFO: m_input_filename   \t" << m_input_filename;
  else
    m_ss << "INFO: m_input_filelist \t" << m_input_filelist;
  StreamToLog(INFO);

  //open the trees
  m_chain_opt   = new TChain("wcsimRootOptionsT");
  m_chain_event = new TChain("wcsimT");
  m_chain_geom  = new TChain("wcsimGeoT");

  //add the files
  if(!ReadTree(m_chain_opt))
    return false;
  if(!ReadTree(m_chain_event))
    return false;
  if(!ReadTree(m_chain_geom))
    return false;

  //set branch addresses
  m_wcsim_opt = new WCSimRootOptions();
  m_chain_opt  ->SetBranchAddress("wcsimrootoptions", &m_wcsim_opt);
  m_chain_opt->GetEntry(0);
  m_wcsim_event_ID = new WCSimRootEvent();
  m_chain_event->SetBranchAddress("wcsimrootevent",   &m_wcsim_event_ID);
  if(m_wcsim_opt->GetGeomHasOD()) {
    Log("INFO: The geometry has an OD. Will add OD digits to m_data", INFO, m_verbose);
    m_wcsim_event_OD = new WCSimRootEvent();
    m_chain_event->SetBranchAddress("wcsimrootevent_OD",   &m_wcsim_event_OD);
    m_data->HasOD = true;
  }
  else {
    m_wcsim_event_OD = 0;
    m_data->HasOD = false;
  }
  m_wcsim_geom = new WCSimRootGeom();
  m_chain_geom ->SetBranchAddress("wcsimrootgeom",    &m_wcsim_geom);

  //set number of events
  if(m_n_events <= 0)
    m_n_events = m_chain_event->GetEntries();
  else if (m_n_events > m_chain_event->GetEntries())
    m_n_events = m_chain_event->GetEntries();
  m_current_event_num = 0;

  //ensure that the geometry & options are the same for each file
  if(!CompareTree(m_chain_opt, 0)) {
    m_n_events = 0;
    return false;
  }
  if(!CompareTree(m_chain_geom, 1)) {
    m_n_events = 0;
    return false;
  }

  //store the PMT locations
  std::cerr << "OD PMTs are not currently stored in WCSimRootGeom. When they are TODO fill IDGeom & ODGeom depending on where the PMT is" << std::endl;
  m_chain_geom->GetEntry(0);
  for(int ipmt = 0; ipmt < m_wcsim_geom->GetWCNumPMT(); ipmt++) {
    WCSimRootPMT pmt = m_wcsim_geom->GetPMT(ipmt);
    PMTInfo pmt_light(pmt.GetTubeNo(), pmt.GetPosition(0), pmt.GetPosition(1), pmt.GetPosition(2));
    m_data->IDGeom.push_back(pmt_light);
  }//ipmt

  //store the pass through information in the transient data model
  m_data->WCSimGeomTree = m_chain_geom;
  m_data->WCSimOptionsTree = m_chain_opt;
  m_data->WCSimEventTree = m_chain_event;
  m_data->IDWCSimEvent_Raw = m_wcsim_event_ID;
  m_data->ODWCSimEvent_Raw = m_wcsim_event_OD;

  //setup the TObjArray to store the filenames
  //int nfiles = m_chain_event->GetListOfFiles()->GetEntries();
  m_data->CurrentWCSimFiles = new TObjArray();
  m_data->CurrentWCSimFiles->SetOwner(true);

  //store the relevant options
  m_data->IsMC = true;
  //geometry
  m_data->IDPMTDarkRate = m_wcsim_opt->GetPMTDarkRate("tank");
  m_data->IDNPMTs = m_wcsim_geom->GetWCNumPMT();
  if(m_data->HasOD) {
    m_data->ODPMTDarkRate = m_wcsim_opt->GetPMTDarkRate("OD");
    m_data->ODNPMTs = m_wcsim_geom->GetODWCNumPMT();
  } else {
    m_data->ODPMTDarkRate = 0;
    m_data->ODNPMTs = 0;
  }

  return true;
}

bool WCSimReader::AddTreeToChain(const char * fname, TChain * chain) {
  if(! chain->Add(fname, -1)) {
    m_ss << "ERROR: Could not load tree: " << chain->GetName()
       << " in file(s): " << fname;
    StreamToLog(ERROR);
    return false;
  }
  m_ss << "INFO: Loaded tree: " << chain->GetName()
     << " from file(s): " << fname
     << " Chain: " << chain->GetName()
     << " now has " << chain->GetEntries()
     << " entries";
  StreamToLog(INFO);
  return true;
}

bool WCSimReader::ReadTree(TChain * chain) {
  //use InFile
  if(m_input_filename.size()) {
    return AddTreeToChain(m_input_filename.c_str(), chain);
  }
  //use FileList
  else if(m_input_filelist.size()) {
    std::ifstream infile(m_input_filelist.c_str());
    std::string fname;
    bool return_value = true;
    while(infile >> fname) {
      if(fname.size())
	if(!AddTreeToChain(fname.c_str(), chain))
	  return_value = false;
    }//read file list file
    return return_value;
  }
  else {
    Log("ERROR: Must use exactly one of the following options: infile filelist", ERROR, m_verbose);
    return false;
  }
}

bool WCSimReader::CompareTree(TChain * chain, int mode)
{
  int n = chain->GetEntries();
  //only 1 entry so nothing to compare
  if(n == 1)
    return true;
  //get the 1st entry
  chain->GetEntry(0);
  std::string modestr;
  WCSimRootOptions * wcsim_opt_0;
  WCSimRootGeom    * wcsim_geom_0;
  if(mode == 0) {
    wcsim_opt_0 = new WCSimRootOptions(*m_wcsim_opt);
    modestr = "WCSimRootOptions";
  }
  else if(mode == 1) {
    //wcsim_geom_0 = new WCSimRootGeom(*m_wcsim_geom); //this operation doesn't work in WCSim
    Log("WARN: geometry not being checked for equality between files. TODO - uncomment this line after WCSim PR 281", WARN, m_verbose);
    modestr = "WCSimRootGeom";
  }
  //loop over all the other entries
  bool diff = false;
  for(int i = 1; i < n; i++) {
    chain->GetEntry(i);
    bool diff_file = false;
    if(mode == 0) {
      //compare only the relevant options
      //WCSimDetector
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDetectorName(),
					     m_wcsim_opt->GetDetectorName(),
					     "DetectorName");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetGeomHasOD(),
					   m_wcsim_opt->GetGeomHasOD(),
					   "GeomHasOD");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetPMTQEMethod(),
					  m_wcsim_opt->GetPMTQEMethod(),
					  "PMTQEMethod");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetPMTCollEff(),
					  m_wcsim_opt->GetPMTCollEff(),
					  "PMTCollEff");
      //WCSimWCAddDarkNoise
      std::vector<string> pmtlocs;
      pmtlocs.push_back("tank");
      pmtlocs.push_back("OD");
      for(unsigned int i = 0; i < pmtlocs.size(); i++) {
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetPMTDarkRate(pmtlocs.at(i)),
						 m_wcsim_opt->GetPMTDarkRate(pmtlocs.at(i)),
						 "PMTDarkRate");
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetConvRate(pmtlocs.at(i)),
						 m_wcsim_opt->GetConvRate(pmtlocs.at(i)),
						 "ConvRate");
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDarkHigh(pmtlocs.at(i)),
						 m_wcsim_opt->GetDarkHigh(pmtlocs.at(i)),
						 "DarkHigh");
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDarkLow(pmtlocs.at(i)),
						 m_wcsim_opt->GetDarkLow(pmtlocs.at(i)),
						 "DarkLow");
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDarkWindow(pmtlocs.at(i)),
						 m_wcsim_opt->GetDarkWindow(pmtlocs.at(i)),
						 "DarkWindow");
	diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDarkMode(pmtlocs.at(i)),
						 m_wcsim_opt->GetDarkMode(pmtlocs.at(i)),
						 "DarkMode");
      }//i
      //WCSimWCDigitizer
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDigitizerClassName(),
					     m_wcsim_opt->GetDigitizerClassName(),
					     "DigitizerClassName");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDigitizerDeadTime(),
					  m_wcsim_opt->GetDigitizerDeadTime(),
					  "DigitizerDeadTime");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDigitizerIntegrationWindow(),
					  m_wcsim_opt->GetDigitizerIntegrationWindow(),
					  "DigitizerIntegrationWindow");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDigitizerTimingPrecision(),
					  m_wcsim_opt->GetDigitizerTimingPrecision(),
					  "DigitizerTimingPrecision");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetDigitizerPEPrecision(),
					  m_wcsim_opt->GetDigitizerPEPrecision(),
					  "DigitizerPEPrecision");
      //WCSimTuningParameters
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetRayff(),
					     m_wcsim_opt->GetRayff(),
					     "Rayff");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetBsrff(),
					     m_wcsim_opt->GetBsrff(),
					     "Bsrff");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetAbwff(),
					     m_wcsim_opt->GetAbwff(),
					     "Abwff");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetRgcff(),
					     m_wcsim_opt->GetRgcff(),
					     "Rgcff");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetMieff(),
					     m_wcsim_opt->GetMieff(),
					     "Mieff");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetTvspacing(),
					     m_wcsim_opt->GetTvspacing(),
					     "Tvspacing");
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetTopveto(),
					   m_wcsim_opt->GetTopveto(),
					   "Topveto");
      //WCSimPhysicsListFactory
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetPhysicsListName(),
					     m_wcsim_opt->GetPhysicsListName(),
					     "PhysicsListName");
      //WCSimRandomParameters
      diff_file = diff_file || CompareVariable(wcsim_opt_0->GetRandomGenerator(),
					  m_wcsim_opt->GetRandomGenerator(),
					  "RandomGenerator");

    }//mode == 0
    else if(mode == 1) {
      //diff_file = !wcsim_geom_0->CompareAllVariables(m_wcsim_geom);
      Log("WARN: geometry not being checked for equality between files. TODO - uncomment this line after WCSim PR 281", WARN, m_verbose);
    }//mode == 1
    if(diff_file) {
      m_ss << "ERROR: Difference between " << modestr << " tree between input file 0 and " << i;
      StreamToLog(ERROR);
      diff = true;
    }
  }//i
  if(mode == 0) {
    delete wcsim_opt_0;
  }
  else if(mode == 1) {
    //delete wcsim_geom_0;
    Log("WARN: geometry not being checked for equality between files. TODO - uncomment this line after WCSim PR 281", WARN, m_verbose);
  }
  if(diff) {
    m_ss << "ERROR: Difference between " << modestr << " trees";
    StreamToLog(ERROR);
    return false;
  }
  return true;
}

template <typename T> bool WCSimReader::CompareVariable(T v1, T v2, const char * tag)
{
  if(v1 == v2)
    return false;
  else {
    m_ss << "WARN: Difference between strings " << v1 << " and " << v2 << " for variable " << tag;
    StreamToLog(WARN);
    return true;
  }
}

bool WCSimReader::Execute(){
  m_data->IDSamples.clear();
  m_data->ODSamples.clear();

  if(m_n_events <= 0) {
    Log("WARN: Reading 0 events", WARN, m_verbose);
    m_data->vars.Set("StopLoop",1);
    return true;
  }

  if(m_current_event_num % 100 == 0) {
    m_ss << "INFO: Event " << m_current_event_num+1 << " of " << m_n_events;
    StreamToLog(INFO);
  }
  else if(m_verbose >= DEBUG1) {
    m_ss << "DEBUG: Event " << m_current_event_num+1 << " of " << m_n_events;
    StreamToLog(DEBUG1);
  }
  //get the digits
  if(!m_chain_event->GetEntry(m_current_event_num)) {
    m_ss << "WARN: Could not read event " << m_current_event_num << " in event TChain";
    StreamToLog(WARN);
    return false;
  }

  //store the WCSim filename(s) and event number(s) for the current event(s)
  m_data->CurrentWCSimFiles->Clear();
  m_data->CurrentWCSimEventNums.clear();
  TObjString * fname = new TObjString(m_chain_event->GetFile()->GetName());
  int event_in_wcsim_file = m_current_event_num - m_chain_event->GetTreeOffset()[m_chain_event->GetTreeNumber()];
  m_ss << "DEBUG: Current event is event " << event_in_wcsim_file << " from WCSim file " << fname->String() << " " << m_chain_event->GetTreeOffset()[m_chain_event->GetTreeNumber()];
  StreamToLog(DEBUG1);
  m_data->CurrentWCSimFiles->Add(fname);
  m_data->CurrentWCSimEventNums.push_back(event_in_wcsim_file);

  //store digit info in the transient data model
  //ID
  if(m_wcsim_event_ID->GetNumberOfEvents() == 1 && m_wcsim_event_ID->GetTrigger(0)->GetTriggerType() == kTriggerNoTrig) {
    //a trigger hasn't been run, so we just add all digits to a single SubSample
    m_wcsim_trigger = m_wcsim_event_ID->GetTrigger(0);
    SubSample subid = GetDigits();
    m_data->IDSamples.push_back(subid);
  }
  else {
    //a trigger has been run, so we need to get all digits from all WCSim event triggers, and fill in the relevant TriggerInfo
    SubSample subid_all;
    for(int itrigger = 0; itrigger < m_wcsim_event_ID->GetNumberOfEvents(); itrigger++) {
      m_wcsim_trigger = m_wcsim_event_ID->GetTrigger(itrigger);
      SubSample subid_this = GetDigits();
      double trigger_time = m_wcsim_trigger->GetHeader()->GetDate();
      //this is a hack to get something close to the readout window size:
      // take the first/last hit times
      double first = +9E300;
      double last  = -9E300;
      for(size_t ihit = 0; ihit < subid_this.m_time.size(); ihit++) {
	double time = subid_this.m_time[ihit];
	if(time < first)
	  first = time;
	if(time > last)
	  last  = time;
      }//ihit
      m_data->IDTriggers.AddTrigger(m_wcsim_trigger->GetTriggerType(),
				    first,
				    last,
				    trigger_time,
				    m_wcsim_trigger->GetTriggerInfo());
      subid_all.Append(subid_this);
    }//itrigger
    m_data->IDSamples.push_back(subid_all);
  }//
  //OD
  if(m_wcsim_event_OD) {
    m_wcsim_trigger = m_wcsim_event_OD->GetTrigger(0);
    SubSample subod = GetDigits();
    m_data->ODSamples.push_back(subod);
  }

  //and finally, increment event counter
  m_current_event_num++;

  //and flag to exit the Execute() loop, if appropriate
  if(m_current_event_num >= m_n_events)
    m_data->vars.Set("StopLoop",1);

  return true;
}

SubSample WCSimReader::GetDigits()
{
  // Store times relative to first hit time
  TimeDelta first_time;
  //loop over the digits
  std::vector<int> PMTid;
  std::vector<float> charge;
  std::vector<TimeDelta::short_time_t> time;
  for(int idigi = 0; idigi < m_wcsim_trigger->GetNcherenkovdigihits(); idigi++) {
    //get a digit
    TObject *element = (m_wcsim_trigger->GetCherenkovDigiHits())->At(idigi);
    WCSimRootCherenkovDigiHit *digit =
      dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
    //get the digit information
    int ID = digit->GetTubeId();
    if (idigi == 0){
      // Store times relative to the first digit
      first_time = TimeDelta(digit->GetT());
    }
    float T = (TimeDelta(digit->GetT()) - first_time) / TimeDelta::ns;
    float Q = digit->GetQ();
    PMTid.push_back(ID);
    time.push_back(T);
    charge.push_back(Q);
    //print
    if(idigi < 10 || m_verbose >= DEBUG2) {
      m_ss << "DEBUG: Digit " << idigi 
	 << " has T " << T
	 << ", Q " << Q
	 << " on PMT " << ID;
      StreamToLog(DEBUG1);
    }
  }//idigi  
  m_ss << "DEBUG: Saved information on " << time.size() << " digits";
  StreamToLog(DEBUG1);

  // Set the timestamp of the SubSample to the first digit time (the one every
  // other time is relative to) plus the trigger time as reported by WCSim.
  // WCSim stores all digit times relative to the trigger time.
  //
  // WCSim also adds a 950 ns offset to the digit times, if it no running in
  // the NoTrigger mode. But we should not care about that here.
  TimeDelta timestamp = TimeDelta(m_wcsim_trigger->GetHeader()->GetDate()) + first_time;
  SubSample sub;
  if (not sub.Append(PMTid, time, charge, timestamp)){
    Log("ERROR: Appending hits failed!", ERROR, m_verbose);
  }

  return sub;
}

bool WCSimReader::Finalise(){
  m_ss << "INFO: Read " << m_current_event_num << " WCSim events";
  StreamToLog(INFO);

  delete m_data->CurrentWCSimFiles;

  delete m_chain_opt;
  delete m_chain_event;
  delete m_chain_geom;

  delete m_wcsim_opt;
  delete m_wcsim_event_ID;
  if(m_wcsim_event_OD)
    delete m_wcsim_event_OD;
  delete m_wcsim_geom;

  return true;
}
