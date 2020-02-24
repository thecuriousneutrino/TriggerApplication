#include "WCSimReader.h"

#include "TFile.h"
#include "TVectorT.h"

WCSimReader::WCSimReader():Tool(){}


bool WCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  //config reading
  if(! m_variables.Get("nevents",  fNEvents) ) {
    Log("WARN: nevents configuration not found. Reading all events", WARN, verbose);
    fNEvents = -1;
  }
  if(! (m_variables.Get("infile",   fInFile) != 
	m_variables.Get("filelist", fFileList))) {
    Log("ERROR: You must use exactly one of the following options: infile filelist", ERROR, verbose);
    return false;
  }

  ss << "INFO: fNEvents  \t" << fNEvents;
  StreamToLog(INFO);
  if(fInFile.size())
    ss << "INFO: fInFile   \t" << fInFile;
  else
    ss << "INFO: fFileList \t" << fFileList;
  StreamToLog(INFO);

  //open the trees
  fChainOpt   = new TChain("wcsimRootOptionsT");
  fChainEvent = new TChain("wcsimT");
  fChainGeom  = new TChain("wcsimGeoT");

  if(!ReadTree(fChainOpt))
    return false;
  if(!ReadTree(fChainEvent))
    return false;
  if(!ReadTree(fChainGeom))
    return false;

  fWCOpt = new WCSimRootOptions();
  fChainOpt  ->SetBranchAddress("wcsimrootoptions", &fWCOpt);
  fChainOpt->GetEntry(0);
  fWCEvtID = new WCSimRootEvent();
  fChainEvent->SetBranchAddress("wcsimrootevent",   &fWCEvtID);
  if(fWCOpt->GetGeomHasOD()) {
    Log("INFO: The geometry has an OD. Will add OD digits to m_data", INFO, verbose);
    fWCEvtOD = new WCSimRootEvent();
    fChainEvent->SetBranchAddress("wcsimrootevent_OD",   &fWCEvtOD);
    m_data->HasOD = true;
  }
  else {
    fWCEvtOD = 0;
    m_data->HasOD = false;
  }
  fWCGeo = new WCSimRootGeom();
  fChainGeom ->SetBranchAddress("wcsimrootgeom",    &fWCGeo);

  //set number of events
  if(fNEvents <= 0)
    fNEvents = fChainEvent->GetEntries();
  else if (fNEvents > fChainEvent->GetEntries())
    fNEvents = fChainEvent->GetEntries();
  fCurrEvent = 0;

  //ensure that the geometry & options are the same for each file
  if(!CompareTree(fChainOpt, 0)) {
    fNEvents = 0;
    return false;
  }
  if(!CompareTree(fChainGeom, 1)) {
    fNEvents = 0;
    return false;
  }

  //store the PMT locations
  std::cerr << "OD PMTs are not currently stored in WCSimRootGeom. When they are TODO fill IDGeom & ODGeom depending on where the PMT is" << std::endl;
  fChainGeom->GetEntry(0);
  for(int ipmt = 0; ipmt < fWCGeo->GetWCNumPMT(); ipmt++) {
    WCSimRootPMT pmt = fWCGeo->GetPMT(ipmt);
    PMTInfo pmt_light(pmt.GetTubeNo(), pmt.GetPosition(0), pmt.GetPosition(1), pmt.GetPosition(2));
    m_data->IDGeom.push_back(pmt_light);
  }//ipmt

  //store the pass through information in the transient data model
  m_data->WCSimGeomTree = fChainGeom;
  m_data->WCSimOptionsTree = fChainOpt;
  m_data->WCSimEventTree = fChainEvent;
  m_data->IDWCSimEvent_Raw = fWCEvtID;
  m_data->ODWCSimEvent_Raw = fWCEvtOD;

  //setup the TObjArray to store the filenames
  //int nfiles = fChainEvent->GetListOfFiles()->GetEntries();
  m_data->CurrentWCSimFiles = new TObjArray();
  m_data->CurrentWCSimFiles->SetOwner(true);

  //store the relevant options
  m_data->IsMC = true;
  //geometry
  m_data->IDPMTDarkRate = fWCOpt->GetPMTDarkRate("tank");
  m_data->IDNPMTs = fWCGeo->GetWCNumPMT();
  if(m_data->HasOD) {
    m_data->ODPMTDarkRate = fWCOpt->GetPMTDarkRate("OD");
    m_data->ODNPMTs = fWCGeo->GetODWCNumPMT();
  } else {
    m_data->ODPMTDarkRate = 0;
    m_data->ODNPMTs = 0;
  }

  return true;
}

bool WCSimReader::AddTreeToChain(const char * fname, TChain * chain) {
  if(! chain->Add(fname, -1)) {
    ss << "ERROR: Could not load tree: " << chain->GetName()
       << " in file(s): " << fname;
    StreamToLog(ERROR);
    return false;
  }
  ss << "INFO: Loaded tree: " << chain->GetName()
     << " from file(s): " << fname
     << " Chain: " << chain->GetName()
     << " now has " << chain->GetEntries()
     << " entries";
  StreamToLog(INFO);
  return true;
}

bool WCSimReader::ReadTree(TChain * chain) {
  //use InFile
  if(fInFile.size()) {
    return AddTreeToChain(fInFile.c_str(), chain);
  }
  //use FileList
  else if(fFileList.size()) {
    std::ifstream infile(fFileList.c_str());
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
    Log("ERROR: Must use exactly one of the following options: infile filelist", ERROR, verbose);
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
  if(mode == 0) {
    fWCOpt_Store = new WCSimRootOptions(*fWCOpt);
    modestr = "WCSimRootOptions";
  }
  else if(mode == 1) {
    //fWCGeo_Store = new WCSimRootGeom(*fWCGeo); //this operation doesn't work in WCSim
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
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDetectorName(),
					     fWCOpt->GetDetectorName(),
					     "DetectorName");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetGeomHasOD(),
					   fWCOpt->GetGeomHasOD(),
					   "GeomHasOD");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetPMTQEMethod(),
					  fWCOpt->GetPMTQEMethod(),
					  "PMTQEMethod");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetPMTCollEff(),
					  fWCOpt->GetPMTCollEff(),
					  "PMTCollEff");
      //WCSimWCAddDarkNoise
      std::vector<string> pmtlocs;
      pmtlocs.push_back("tank");
      pmtlocs.push_back("OD");
      for(unsigned int i = 0; i < pmtlocs.size(); i++) {
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetPMTDarkRate(pmtlocs.at(i)),
						 fWCOpt->GetPMTDarkRate(pmtlocs.at(i)),
						 "PMTDarkRate");
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetConvRate(pmtlocs.at(i)),
						 fWCOpt->GetConvRate(pmtlocs.at(i)),
						 "ConvRate");
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkHigh(pmtlocs.at(i)),
						 fWCOpt->GetDarkHigh(pmtlocs.at(i)),
						 "DarkHigh");
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkLow(pmtlocs.at(i)),
						 fWCOpt->GetDarkLow(pmtlocs.at(i)),
						 "DarkLow");
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkWindow(pmtlocs.at(i)),
						 fWCOpt->GetDarkWindow(pmtlocs.at(i)),
						 "DarkWindow");
	diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkMode(pmtlocs.at(i)),
						 fWCOpt->GetDarkMode(pmtlocs.at(i)),
						 "DarkMode");
      }//i
      //WCSimWCDigitizer
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDigitizerClassName(),
					     fWCOpt->GetDigitizerClassName(),
					     "DigitizerClassName");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDigitizerDeadTime(),
					  fWCOpt->GetDigitizerDeadTime(),
					  "DigitizerDeadTime");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDigitizerIntegrationWindow(),
					  fWCOpt->GetDigitizerIntegrationWindow(),
					  "DigitizerIntegrationWindow");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDigitizerTimingPrecision(),
					  fWCOpt->GetDigitizerTimingPrecision(),
					  "DigitizerTimingPrecision");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDigitizerPEPrecision(),
					  fWCOpt->GetDigitizerPEPrecision(),
					  "DigitizerPEPrecision");
      //WCSimTuningParameters
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetRayff(),
					     fWCOpt->GetRayff(),
					     "Rayff");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetBsrff(),
					     fWCOpt->GetBsrff(),
					     "Bsrff");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetAbwff(),
					     fWCOpt->GetAbwff(),
					     "Abwff");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetRgcff(),
					     fWCOpt->GetRgcff(),
					     "Rgcff");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetMieff(),
					     fWCOpt->GetMieff(),
					     "Mieff");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetTvspacing(),
					     fWCOpt->GetTvspacing(),
					     "Tvspacing");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetTopveto(),
					   fWCOpt->GetTopveto(),
					   "Topveto");
      //WCSimPhysicsListFactory
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetPhysicsListName(),
					     fWCOpt->GetPhysicsListName(),
					     "PhysicsListName");
      //WCSimRandomParameters
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetRandomGenerator(),
					  fWCOpt->GetRandomGenerator(),
					  "RandomGenerator");

    }//mode == 0
    else if(mode == 1) {
      diff_file = fWCGeo_Store->CompareAllVariables(fWCGeo);
    }//mode == 1
    if(diff_file) {
      ss << "ERROR: Difference between " << modestr << " tree between input file 0 and " << i;
      StreamToLog(ERROR);
      diff = true;
    }
  }//i
  if(mode == 0) {
    delete fWCOpt_Store;
  }
  else if(mode == 1) {
    //delete fWCGeo_Store;
  }
  if(diff) {
    ss << "ERROR: Difference between " << modestr << " trees";
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
    ss << "WARN: Difference between strings " << v1 << " and " << v2 << " for variable " << tag;
    StreamToLog(WARN);
  }
}

bool WCSimReader::Execute(){
  m_data->IDSamples.clear();
  m_data->ODSamples.clear();

  if(fNEvents <= 0) {
    Log("WARN: Reading 0 events", WARN, verbose);
    m_data->vars.Set("StopLoop",1);
    return true;
  }

  if(fCurrEvent % 100 == 0) {
    ss << "INFO: Event " << fCurrEvent+1 << " of " << fNEvents;
    StreamToLog(INFO);
  }
  else if(verbose >= DEBUG1) {
    ss << "DEBUG: Event " << fCurrEvent+1 << " of " << fNEvents;
    StreamToLog(DEBUG1);
  }
  //get the digits
  if(!fChainEvent->GetEntry(fCurrEvent)) {
    ss << "WARN: Could not read event " << fCurrEvent << " in event TChain";
    StreamToLog(WARN);
    return false;
  }

  //store the WCSim filename(s) and event number(s) for the current event(s)
  m_data->CurrentWCSimFiles->Clear();
  m_data->CurrentWCSimEventNums.clear();
  TObjString * fname = new TObjString(fChainEvent->GetFile()->GetName());
  int event_in_wcsim_file = fCurrEvent - fChainEvent->GetTreeOffset()[fChainEvent->GetTreeNumber()];
  ss << "DEBUG: Current event is event " << event_in_wcsim_file << " from WCSim file " << fname->String() << " " << fChainEvent->GetTreeOffset()[fChainEvent->GetTreeNumber()];
  StreamToLog(DEBUG1);
  m_data->CurrentWCSimFiles->Add(fname);
  m_data->CurrentWCSimEventNums.push_back(event_in_wcsim_file);

  //store digit info in the transient data model
  //ID
  fEvt = fWCEvtID->GetTrigger(0);
  SubSample subid = GetDigits();
  m_data->IDSamples.push_back(subid);
  //OD
  if(fWCEvtOD) {
    fEvt = fWCEvtOD->GetTrigger(0);
    SubSample subod = GetDigits();
    m_data->ODSamples.push_back(subod);
  }

  //and finally, increment event counter
  fCurrEvent++;

  //and flag to exit the Execute() loop, if appropriate
  if(fCurrEvent >= fNEvents)
    m_data->vars.Set("StopLoop",1);

  return true;
}

SubSample WCSimReader::GetDigits()
{
  //loop over the digits
  std::vector<int> PMTid;
  std::vector<float>  time, charge;
  for(int idigi = 0; idigi < fEvt->GetNcherenkovdigihits(); idigi++) {
    //get a digit
    TObject *element = (fEvt->GetCherenkovDigiHits())->At(idigi);
    WCSimRootCherenkovDigiHit *digit = 
      dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
    //get the digit information
    int ID = digit->GetTubeId();
    float T = digit->GetT();
    float Q = digit->GetQ();
    PMTid.push_back(ID);
    time.push_back(T);
    charge.push_back(Q);
    //print
    if(idigi < 10 || verbose >= DEBUG2) {
      ss << "DEBUG: Digit " << idigi 
	 << " has T " << T
	 << ", Q " << Q
	 << " on PMT " << ID;
      StreamToLog(DEBUG1);
    }
  }//idigi  
  ss << "DEBUG: Saved information on " << time.size() << " digits";
  StreamToLog(DEBUG1);

  SubSample sub(PMTid, time, charge);

  return sub;  
}

bool WCSimReader::Finalise(){
  ss << "INFO: Read " << fCurrEvent << " WCSim events";
  StreamToLog(INFO);

  delete m_data->CurrentWCSimFiles;

  delete fChainOpt;
  delete fChainEvent;
  delete fChainGeom;

  delete fWCOpt;
  delete fWCEvtID;
  if(fWCEvtOD)
    delete fWCEvtOD;
  delete fWCGeo;

  return true;
}
