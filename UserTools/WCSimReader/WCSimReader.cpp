#include "WCSimReader.h"

using std::cout;
using std::cerr;
using std::endl;

WCSimReader::WCSimReader():Tool(){}


bool WCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  //config reading
  if(! m_variables.Get("nevents",  fNEvents) ) {
    cout << "nevents configuration not found. Reading all events" << endl;
    fNEvents = -1;
  }
  if(! (m_variables.Get("infile",   fInFile) ||
	m_variables.Get("filelist", fFileList))) {
    cerr << "You must use one of the following options: "
	 << " infile filelist" << endl;
    return false;
  }

  cout << "fNEvents  \t" << fNEvents  << endl;
  cout << "fInFile   \t" << fInFile   << endl;
  cout << "fFileList \t" << fFileList << endl;

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
  cerr << "Waiting for WCSim PR #251 to be able to save OD digits" << endl;
  if(false) {
  //if(fWCOpt->GetGeomHasOD()) { // TODO uncomment this after WCSim 251 merged
    cout << "The geometry has an OD. Will add OD digits to m_data" << endl;
    fWCEvtOD = new WCSimRootEvent();
    fChainEvent->SetBranchAddress("wcsimrootevent_OD",   &fWCEvtOD);
  }
  else
    fWCEvtOD = 0;
  fWCGeo = new WCSimRootGeom();
  fChainGeom ->SetBranchAddress("wcsimrootgeom",    &fWCGeo);

  //ensure that the geometry & options are the same for each file
  if(!CompareTree(fChainOpt))
    return false;
  if(!CompareTree(fChainGeom))
    return false;


  //set number of events
  if(fNEvents <= 0)
    fNEvents = fChainEvent->GetEntries();
  else if (fNEvents > fChainEvent->GetEntries())
    fNEvents = fChainEvent->GetEntries();
  fCurrEvent = 0;

  //store the PMT locations
  cerr << "OD PMTs are not currently stored in WCSimRootGeom. When they are TODO fill IDGeom & ODGeom depending on where the PMT is" << endl;
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
  m_data->IDPMTDarkRate = fWCOpt->GetPMTDarkRate();
  m_data->IDNPMTs = fWCGeo->GetWCNumPMT();
  cerr << "OD Dark rate is not current stored. TODO add when WCSim #246 is merged" << endl;
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
      cerr << "Could not load tree: " << chain->GetName()
	   << " in file(s): " << fInFile << endl;
      return false;
    }
    cout << "Loaded tree: " << chain->GetName()
	 << " from file(s): " << fInFile
	 << " with: " << chain->GetEntries()
	 << " entries" << endl;
    return true;
  }
  //use FileList
  else if(fFileList.size()) {
    cerr << "FileList not implemented" << endl;
    return false;
  }
  else {
    cerr << "Must use one of the following options: "
	 << " infile filelist" << endl;
    return false;
  }
}

bool WCSimReader::CompareTree(TChain * chain)
{
  cerr << "This function to ensure that the geometry (and at least some of the options tree) are identical is not yet implemented" << endl;
  return true;
}

bool WCSimReader::Execute(){
  cout << "Event " << fCurrEvent+1 << " of " << fNEvents << endl;
  //get the digits
  if(!fChainEvent->GetEntry(fCurrEvent)) {
    cerr << "Could not read event " << fCurrEvent << " in event TChain" << endl;
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
    if(idigi < 10)
    cout << "Digit " << idigi 
	 << " has T " << digit->GetT()
	 << ", Q " << digit->GetQ()
	 << " on PMT " << digit->GetTubeId() << endl;
  }//idigi  
  cout << "Saved information on " << time.size() << " digits" << endl;

  SubSample sub(PMTid, time, charge);

  return sub;  
}

bool WCSimReader::Finalise(){
  cout << "Read " << fCurrEvent << " WCSim events" << endl;

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
