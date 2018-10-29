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
  fWCEvt = new WCSimRootEvent();
  fChainEvent->SetBranchAddress("wcsimrootevent",   &fWCEvt);
  fWCGeo = new WCSimRootGeom();
  fChainGeom ->SetBranchAddress("wcsimrootgeom",    &fWCGeo);
  //TODO OD digits

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

  //store the pass through information in the transient data model
  m_data->WCSimOpt = *fWCOpt;
  m_data->WCSimEvt = *fWCEvt;
  m_data->WCSimGeo = *fWCGeo;

  //store the PMT locations
  /*
  for(int ipmt = 0; ipmt < fWCGeo->GetWCNumPMT(); ipmt++) {
    WCSimRootPMT pmt = fWCGeo->GetPMT(ipmt);
    PMTInfo pmt_light(pmt.GetTubeNo(), pmt.GetPosition(0), pmt.GetPosition(1), pmt.GetPosition(2));
    m_data->IDGeom.push_back(pmt_light);
    //TODO differentiate OD/ID PMTs
  }//ipmt
  */

  //store the relevant options
  //TODO dark noise

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
  fEvt = fWCEvt->GetTrigger(0);

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

  //store digit info in the transient data model
  SubSample sub(PMTid, time, charge);
  m_data->Samples.push_back(sub);

  //and finally, increment event counter
  fCurrEvent++;
  if(fCurrEvent >= fNEvents)
    m_data->vars.Set("StopLoop",1);

  return true;
}


bool WCSimReader::Finalise(){

  delete fChainOpt;
  delete fChainEvent;
  delete fChainGeom;

  return true;
}
