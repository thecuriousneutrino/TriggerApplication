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

  //other options
  fSaveMultiDigiPerTrigger = true;
  m_variables.Get("save_multiple_digits_per_trigger", fSaveMultiDigiPerTrigger);
  fTriggerOffset = 0;
  m_variables.Get("trigger_offset", fTriggerOffset);

  //setup the out event tree
  // Nevents unique event objects
  Int_t bufsize = 64000;
  fTreeEvent = new TTree("wcsimT","WCSim Tree");
  fWCSimEventID = new WCSimRootEvent();
  fTreeEvent->Branch("wcsimrootevent", "WCSimRootEvent", &fWCSimEventID, bufsize,2);
  if(m_data->HasOD) {
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

  fTriggers = new TriggerInfo();
  fEvtNum = 0;

  for(int i = 0; i <= m_data->IDNPMTs; i++)
    fIDNDigitPerPMTPerTriggerMap[i] = std::map<int, bool>();
  for(int i = 0; i <= m_data->ODNPMTs; i++)
    fODNDigitPerPMTPerTriggerMap[i] = std::map<int, bool>();

  return true;
}


bool DataOut::Execute(){

  Log("DEBUG: DataOut::Execute Starting", DEBUG1, verbose);

  for(int i = 0; i <= m_data->IDNPMTs; i++)
    fIDNDigitPerPMTPerTriggerMap[i].clear();
  for(int i = 0; i <= m_data->ODNPMTs; i++)
    fODNDigitPerPMTPerTriggerMap[i].clear();

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
  (*fWCSimEventID) = (*(m_data->WCSimEventID));
  //prepare the subtriggers
  CreateSubEvents(fWCSimEventID);
  //remove the digits that aren't in the trigger window(s)
  // also move digits from the 0th trigger to the trigger window it's in
  RemoveDigits(fWCSimEventID, fIDNDigitPerPMTPerTriggerMap);
  //set some trigger header infromation that requires all the digits to be 
  // present to calculate e.g. sumq
  FinaliseSubEvents(fWCSimEventID);
  
  if(m_data->HasOD) {
    (*fWCSimEventOD) = (*(m_data->WCSimEventOD));
    CreateSubEvents(fWCSimEventOD);
    RemoveDigits(fWCSimEventOD, fODNDigitPerPMTPerTriggerMap);
    FinaliseSubEvents(fWCSimEventOD);
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
    double offset = fTriggerOffset;
    trig->SetHeader(fEvtNum, 0, fTriggers->m_triggertime.at(i) - offset, i+1);
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
    int ntubeshit = 0;
    for(int j = 0; j < trig->GetNcherenkovdigihits_slots(); j++) {
      WCSimRootCherenkovDigiHit * digi = (WCSimRootCherenkovDigiHit *)digits->At(j);
      if(digi) {
	sumq += digi->GetQ();
	ntubeshit++;
      }
    }//j
    trig->SetSumQ(sumq);
    //this is actually number of digits, not number of unique PMTs with digits
    trig->SetNumDigitizedTubes(ntubeshit);
  }//i
}

void DataOut::RemoveDigits(WCSimRootEvent * WCSimEvent, std::map<int, std::map<int, bool> > & NDigitPerPMTPerTriggerMap)
{
  if(!fTriggers->m_N) {
    ss << "DEBUG: No trigger intervals to save";
    StreamToLog(DEBUG1);
  }
  WCSimRootTrigger * trig0 = WCSimEvent->GetTrigger(0);
  TClonesArray * digits = trig0->GetCherenkovDigiHits();
  int ndigits = trig0->GetNcherenkovdigihits();
  int ndigits_slots = trig0->GetNcherenkovdigihits_slots();
  for(int i = 0; i < ndigits_slots; i++) {
    WCSimRootCherenkovDigiHit * d = (WCSimRootCherenkovDigiHit*)digits->At(i);
    if(!d)
      continue;
    double time = d->GetT();
    int window = TimeInTriggerWindow(time);
    int pmt = d->GetTubeId();
    if(window >= 0) {
      //need to apply an offset to the digit time using the trigger time
      d->SetT(time + fTriggerOffset - fTriggers->m_triggertime.at(window));
    }
    if(window > 0 &&
       (!fSaveMultiDigiPerTrigger && !NDigitPerPMTPerTriggerMap[pmt][window])) {
      //need to add digit to a new trigger window
      //but not if we've already saved the 1 digit from this pmt in this window we're allowed
      WCSimEvent->GetTrigger(window)->AddCherenkovDigiHit(d);
    }
    if(window ||
       (window == 0 && !fSaveMultiDigiPerTrigger && NDigitPerPMTPerTriggerMap[pmt][window])) {
      //either not in a trigger window (window = -1)
      //or not in the 0th trigger window (window >= 1)
      //or in 0th window but we've already saved the 1 digit from this pmt in this window we're allowed
      trig0->RemoveCherenkovDigiHit(d);
    }
    if(window >= 0) {
      //save the fact that we've used this PMT
      NDigitPerPMTPerTriggerMap[pmt][window] = true;
    }
  }//i
  ss << "INFO: RemoveDigits() has reduced number of digits from "
     << ndigits << " to " << trig0->GetNcherenkovdigihits();
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
  fTreeEvent->Write();
  fOutFile.Close();

  delete fTreeEvent;
  delete fWCSimEventID;
  if(fWCSimEventOD)
    delete fWCSimEventOD;

  delete fTriggers;

  return true;
}
