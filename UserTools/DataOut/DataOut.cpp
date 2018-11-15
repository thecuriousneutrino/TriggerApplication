#include "DataOut.h"

DataOut::DataOut():Tool(){}


bool DataOut::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  //open the output file
  if(! m_variables.Get("outfilename", fOutFilename)) {
    Log("ERROR: outfilename configuration not found. Cancelling initialisation", ERROR, verbose);
    return false;
  }
  fOutFile.Open(fOutFilename.c_str(), "RECREATE");

  //setup the out event tree
  // Nevents unique event objects
  Int_t bufsize = 64000;
  fTreeEvent = new TTree("wcsimT","WCSim Tree");
  fWCSimEventID = new WCSimRootEvent();
  fTreeEvent->Branch("wcsimrootevent", "WCSimRootEvent", &fWCSimEventID, bufsize,2);
  if(m_data->WCSimEventOD) {
    fWCSimEventOD = new WCSimRootEvent();
    fTreeEvent->Branch("wcsimrootevent_OD", "WCSimRootEvent", &fWCSimEventOD, bufsize,2);
  }
  fTreeEvent->Branch("wcsimfilename", &(m_data->CurrentWCSimFiles), bufsize, 0);
  fTreeEvent->Branch("wcsimeventnums", &(m_data->CurrentWCSimEventNums), bufsize, 0);

  //fill the output event-independent trees
  //There are 1 unique geom objects, so this is a simple clone of 1 entry
  fTreeGeom = m_data->WCSimGeomTree->CloneTree(1);
  fTreeGeom->Write();
  delete fTreeGeom;

  //There are Nfiles unique options objects, so this is a clone of all entries
  // plus a new branch with the wcsim filename 
  fTreeOptions = m_data->WCSimOptionsTree->CloneTree();
  TObjString * wcsimfilename = new TObjString();
  TBranch * branch = fTreeOptions->Branch("wcsimfilename", &wcsimfilename);
  for(int i = 0; i < fTreeOptions->GetEntries(); i++) {
    m_data->WCSimOptionsTree->GetEntry(i);
    fTreeOptions->GetEntry(i);
    wcsimfilename->SetString(m_data->WCSimOptionsTree->GetFile()->GetName());
    branch->Fill();
  }//i
  fTreeOptions->Write();
  delete wcsimfilename;
  delete fTreeOptions;

  return true;
}


bool DataOut::Execute(){

  Log("INFO: DataOut::Execute Starting", verbose, INFO);
  std::cerr << "Trigger vectors not yet stored in DataModel. Just using a fixed cutoff of 1000 ns" << std::endl;

  //get the WCSim event
  (*fWCSimEventID) = (*(m_data->WCSimEventID));
  //remove the digits that aren't in the trigger window(s)
  RemoveDigits(fWCSimEventID);

  if(m_data->WCSimEventOD) {
    (*fWCSimEventOD) = (*(m_data->WCSimEventOD));
    RemoveDigits(fWCSimEventOD);
  }

  fTreeEvent->Fill();

  std::cerr << "DataOut::Execute Done" << std::endl;
  return true;
}

void DataOut::RemoveDigits(WCSimRootEvent * WCSimEvent) {
  WCSimRootTrigger * trig = WCSimEvent->GetTrigger(0);
  TClonesArray * digits = trig->GetCherenkovDigiHits();
  int ndigits = trig->GetNcherenkovdigihits();
  int ndigits_slots = trig->GetNcherenkovdigihits_slots();
  for(int i = 0; i < ndigits_slots; i++) {
    WCSimRootCherenkovDigiHit * d = (WCSimRootCherenkovDigiHit*)digits->At(i);
    if(!d)
      continue;
    double time = d->GetT();
    if(!TimeInRange(time)) {
      trig->RemoveCherenkovDigiHit(d);
    }
  }//i
  ss << "INFO: RemoveDigits() has reduced number of digits from "
     << ndigits << " to " << trig->GetNcherenkovdigihits();
  StreamToLog(INFO);
}

bool DataOut::TimeInRange(double time) {
  if(time > 1000)
    return false;
  return true;
}

bool DataOut::Finalise(){
  fTreeEvent->Write();
  fOutFile.Close();

  delete fTreeEvent;
  delete fWCSimEventID;
  if(fWCSimEventOD)
    delete fWCSimEventOD;

  return true;
}
