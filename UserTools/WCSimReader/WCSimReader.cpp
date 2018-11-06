#include "WCSimReader.h"

using std::cerr;

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
  if(! (m_variables.Get("infile",   fInFile) ||
	m_variables.Get("filelist", fFileList))) {
    Log("ERROR: You must use one of the following options: infile filelist", ERROR, verbose);
    return false;
  }

  ss << "INFO: fNEvents  \t" << fNEvents;
  StreamToLog(INFO);
  ss << "INFO: fInFile   \t" << fInFile;
  StreamToLog(INFO);
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
  }
  else
    fWCEvtOD = 0;
  fWCGeo = new WCSimRootGeom();
  fChainGeom ->SetBranchAddress("wcsimrootgeom",    &fWCGeo);

  //ensure that the geometry & options are the same for each file
  if(!CompareTree(fChainOpt, 0))
    return false;
  if(!CompareTree(fChainGeom, 1))
    return false;


  //set number of events
  if(fNEvents <= 0)
    fNEvents = fChainEvent->GetEntries();
  else if (fNEvents > fChainEvent->GetEntries())
    fNEvents = fChainEvent->GetEntries();
  fCurrEvent = 0;

  //store the PMT locations
  cerr << "OD PMTs are not currently stored in WCSimRootGeom. When they are TODO fill IDGeom & ODGeom depending on where the PMT is" << std::endl;
  fChainGeom->GetEntry(0);
  for(int ipmt = 0; ipmt < fWCGeo->GetWCNumPMT(); ipmt++) {
    WCSimRootPMT pmt = fWCGeo->GetPMT(ipmt);
    PMTInfo pmt_light(pmt.GetTubeNo(), pmt.GetPosition(0), pmt.GetPosition(1), pmt.GetPosition(2));
    m_data->IDGeom.push_back(pmt_light);
  }//ipmt

  //store the pass through information in the transient data model
  //m_data->WCSimOpt = *fWCOpt;
  //m_data->WCSimEvt = *fWCEvt;
  //m_data->WCSimGeo = *fWCGeo;

  //store the relevant options
  m_data->IsMC = true;
  //geometry
  m_data->IDPMTDarkRate = fWCOpt->GetPMTDarkRate();
  m_data->IDNPMTs = fWCGeo->GetWCNumPMT();
  cerr << "OD Dark rate is not current stored. TODO add when WCSim #246 is merged" << std::endl;
  /* TODO Uncomment this when #246 merged
  m_data->IDPMTDarkRate(fWCOpt->GetPMTDarkRate("tank"));
  m_data->IDNPMTs = fWCGeo->GetWCNumPMT("tank");
  m_data->ODPMTDarkRate(fWCOpt->GetPMTDarkRate("OD"));
  m_data->ODNPMTs = fWCGeo->GetWCNumPMT("OD");
  */

  return true;
}

bool WCSimReader::ReadTree(TChain * chain) {
  //use InFile
  if(fInFile.size()) { 
    if(! chain->Add(fInFile.c_str(), -1)) {
      ss << "ERROR: Could not load tree: " << chain->GetName()
	 << " in file(s): " << fInFile;
      StreamToLog(ERROR);
      return false;
    }
    ss << "INFO: Loaded tree: " << chain->GetName()
       << " from file(s): " << fInFile
       << " with: " << chain->GetEntries()
       << " entries";
    StreamToLog(INFO);
    return true;
  }
  //use FileList
  else if(fFileList.size()) {
    Log("ERROR: FileList not implemented", ERROR, verbose);
    return false;
  }
  else {
    Log("ERROR: Must use one of the following options: infile filelist", ERROR, verbose);
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
  if(mode == 0)
    fWCOpt_Store = new WCSimRootOptions(*fWCOpt);
  else if(mode == 1)
    fWCGeo_Store = new WCSimRootGeom(*fWCGeo);
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
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetPMTDarkRate(),
					     fWCOpt->GetPMTDarkRate(),
					     "PMTDarkRate");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetConvRate(),
					     fWCOpt->GetConvRate(),
					     "ConvRate");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkHigh(),
					     fWCOpt->GetDarkHigh(),
					     "DarkHigh");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkLow(),
					     fWCOpt->GetDarkLow(),
					     "DarkLow");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkWindow(),
					     fWCOpt->GetDarkWindow(),
					     "DarkWindow");
      diff_file = diff_file || CompareVariable(fWCOpt_Store->GetDarkMode(),
					  fWCOpt->GetDarkMode(),
					  "DarkMode");
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
    if(diff_file) {
      ss << "ERROR: Difference between WCSimOptions tree between input file 0 and " << i;
      StreamToLog(ERROR);
      diff = true;
    }
  }//i
  if(diff) {
    ss << "ERROR: Difference between WCSimOptions trees";
    StreamToLog(ERROR);
    return false;
  }
  cerr << "This function to ensure that the geometry (and at least some of the options tree) are identical is not yet implemented" << std::endl;
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
  if(fCurrEvent >= fNEvents)
    m_data->vars.Set("StopLoop",1);

  return true;
}

SubSample WCSimReader::GetDigits()
{
  //loop over the digits
  std::vector<int> PMTid, time, charge;
  for(int idigi = 0; idigi < fEvt->GetNcherenkovdigihits(); idigi++) {
    //get a digit
    TObject *element = (fEvt->GetCherenkovDigiHits())->At(idigi);
    WCSimRootCherenkovDigiHit *digit = 
      dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
    //get the digit information
    PMTid.push_back(digit->GetTubeId());
    time.push_back(digit->GetT());
    charge.push_back(digit->GetQ());
    //print
    if(idigi < 10 || verbose >= DEBUG2) {
      ss << "DEBUG: Digit " << idigi 
	 << " has T " << digit->GetT()
	 << ", Q " << digit->GetQ()
	 << " on PMT " << digit->GetTubeId();
      StreamToLog(DEBUG2);
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
