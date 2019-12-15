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
  m_data->IDWCSimEvent_Triggered = new WCSimRootEvent();
  fTreeEvent->Branch("wcsimrootevent", "WCSimRootEvent", &m_data->IDWCSimEvent_Triggered, bufsize,2);
  if(m_data->HasOD) {
    m_data->ODWCSimEvent_Triggered = new WCSimRootEvent();
    fTreeEvent->Branch("wcsimrootevent_OD", "WCSimRootEvent", &m_data->ODWCSimEvent_Triggered, bufsize,2);
  }
  else {
    m_data->ODWCSimEvent_Triggered = 0;
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

  fTriggers = new TriggerInfo();
  fEvtNum = 0;

  return true;
}


bool DataOut::Execute(){

  Log("DEBUG: DataOut::Execute Starting", DEBUG1, verbose);

  //Gather together all the trigger windows
  fTriggers->Clear();
  fTriggers->AddTriggers(&(m_data->IDTriggers));
  if(m_data->HasOD)
    fTriggers->AddTriggers(&(m_data->ODTriggers));
  fTriggers->SortByStartTime();
  ss << "INFO: Have " << fTriggers->m_N << " triggers to save times:";
  StreamToLog(INFO);
  for(int i = 0; i < fTriggers->m_N; i++) {
    ss << "INFO: \t[" << fTriggers->m_starttime.at(i)
       << ", " << fTriggers->m_endtime.at(i) << "] "
       << fTriggers->m_triggertime.at(i) << " ns with type "
       << WCSimEnumerations::EnumAsString(fTriggers->m_type.at(i)) << " extra info";
    for(unsigned int ii = 0; ii < fTriggers->m_info.at(i).size(); ii++)
      ss << " " << fTriggers->m_info.at(i).at(ii);
    StreamToLog(INFO);
  }//i

  //Note: the ranges vector can contain overlapping ranges
  //we want to make sure the triggers output aren't overlapping
  // This is actually handled in DataOut::RemoveDigits()
  // It puts digits into the output event in the earliest trigger they belong to

  //get the WCSim event
  (*m_data->IDWCSimEvent_Triggered) = (*(m_data->IDWCSimEvent_Raw));
  //prepare the subtriggers
  CreateSubEvents(m_data->IDWCSimEvent_Triggered);
  //remove the digits that aren't in the trigger window(s)
  // also move digits from the 0th trigger to the trigger window it's in
  RemoveDigits(m_data->IDWCSimEvent_Triggered);
  //set some trigger header infromation that requires all the digits to be 
  // present to calculate e.g. sumq
  FinaliseSubEvents(m_data->IDWCSimEvent_Triggered);
  
  if(m_data->HasOD) {
    (*m_data->ODWCSimEvent_Triggered) = (*(m_data->ODWCSimEvent_Raw));
    CreateSubEvents(m_data->ODWCSimEvent_Triggered);
    RemoveDigits(m_data->ODWCSimEvent_Triggered);
    FinaliseSubEvents(m_data->ODWCSimEvent_Triggered);
  }

  fTreeEvent->Fill();

  //make sure the triggers are reset for the next event
  m_data->IDTriggers.Clear();
  m_data->ODTriggers.Clear();

  //increment event number
  fEvtNum++;

  Log("DEBUG: DataOut::Execute() Done", DEBUG1, verbose);
  return true;
}

void DataOut::CreateSubEvents(WCSimRootEvent * WCSimEvent)
{
  const int n = fTriggers->m_N;
  for(int i = 0; i < n; i++) {
    if(i)
      WCSimEvent->AddSubEvent();
    WCSimRootTrigger * trig = WCSimEvent->GetTrigger(i);
    trig->SetHeader(fEvtNum, 0, fTriggers->m_triggertime.at(i) - 950, i+1);
    trig->SetTriggerInfo(fTriggers->m_type.at(i), fTriggers->m_info.at(i));
    //trig->SetMode(jhfNtuple.mode);
  }//i
}

void DataOut::FinaliseSubEvents(WCSimRootEvent * WCSimEvent)
{
  const int n = fTriggers->m_N;
  for(int i = 0; i < n; i++) {
    WCSimRootTrigger * trig = WCSimEvent->GetTrigger(i);
    TClonesArray * digits = trig->GetCherenkovDigiHits();
    float sumq = 0;
    for(int j = 0; j < trig->GetNcherenkovdigihits_slots(); j++) {
      WCSimRootCherenkovDigiHit * digi = (WCSimRootCherenkovDigiHit *)digits->At(j);
      if(digi)
	sumq += digi->GetQ();
    }
    trig->SetSumQ(sumq);
  }//i
}

void DataOut::RemoveDigits(WCSimRootEvent * WCSimEvent)
{
  if(!fTriggers->m_N) {
    ss << "DEBUG: No trigger intervals to save";
    StreamToLog(DEBUG1);
  }
  WCSimRootTrigger * trig = WCSimEvent->GetTrigger(0);
  TClonesArray * digits = trig->GetCherenkovDigiHits();
  int ndigits = trig->GetNcherenkovdigihits();
  int ndigits_slots = trig->GetNcherenkovdigihits_slots();
  for(int i = 0; i < ndigits_slots; i++) {
    WCSimRootCherenkovDigiHit * d = (WCSimRootCherenkovDigiHit*)digits->At(i);
    if(!d)
      continue;
    double time = d->GetT();
    int window = TimeInTriggerWindow(time);
    if(window >= 0) {
      //need to apply an offset to the digit time using the trigger time
      d->SetT(time - (fTriggers->m_triggertime.at(window) - 950));
    }
    if(window > 0) {
      //need to add digit to a new trigger window
      WCSimEvent->GetTrigger(window)->AddCherenkovDigiHit(d);
    }
    if(window) {
      //either not in a trigger window (window = -1)
      //or not in the 0th trigger window (window >= 1)
      trig->RemoveCherenkovDigiHit(d);
    }
  }//i
  ss << "INFO: RemoveDigits() has reduced number of digits from "
     << ndigits << " to " << trig->GetNcherenkovdigihits();
  StreamToLog(INFO);
}

int DataOut::TimeInTriggerWindow(double time) {
  for(unsigned int i = 0; i < fTriggers->m_N; i++) {
    double lo = fTriggers->m_starttime.at(i);
    double hi = fTriggers->m_endtime.at(i);
    if(time >= lo && time <= hi)
      return i;
  }//it
  return -1;
}

bool DataOut::Finalise(){
  //multiple TFiles may be open. Ensure we save to the correct one
  fOutFile.cd(TString::Format("%s:/", fOutFilename.c_str()));
  fTreeEvent->Write();
  fOutFile.Close();

  delete fTreeEvent;
  delete m_data->IDWCSimEvent_Triggered;
  if(m_data->ODWCSimEvent_Triggered)
    delete m_data->ODWCSimEvent_Triggered;

  delete fTriggers;

  return true;
}
