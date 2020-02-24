#include "DataOut.h"

DataOut::DataOut():Tool(){}

/////////////////////////////////////////////////////////////////

bool DataOut::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  // Log needs m_data
  Log("DEBUG: DataOut::Initialise starting", DEBUG1, verbose);

  //open the output file
  Log("DEBUG: DataOut::Initialise opening output file...", DEBUG2, verbose);
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
  fSaveOnlyFailedDigits = false;
  m_variables.Get("save_only_failed_digits", fSaveOnlyFailedDigits);

  //setup the out event tree
  Log("DEBUG: DataOut::Initialise setting up output event tree...", DEBUG2, verbose);
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
  Log("DEBUG: DataOut::Initialise filling event-independent trees...", DEBUG2, verbose);
  Log("DEBUG:   Geometry...", DEBUG2, verbose);
  fTreeGeom = m_data->WCSimGeomTree->CloneTree(1);
  fTreeGeom->Write();
  delete fTreeGeom;

  //There are Nfiles unique options objects, so this is a clone of all entries
  // plus a new branch with the wcsim filename 
  Log("DEBUG:   Options & file names...", DEBUG2, verbose);
  fTreeOptions = m_data->WCSimOptionsTree->CloneTree();
  ss << "DEBUG:     entries: " << fTreeOptions->GetEntries();
  StreamToLog(DEBUG2);
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

  Log("DEBUG: DataOut::Initialise creating trigger info...", DEBUG2, verbose);
  fTriggers = new TriggerInfo();
  fEvtNum = 0;

  Log("DEBUG: DataOut::Initialise creating ID trigger maps...", DEBUG2, verbose);
  ss << "DEBUG:   entries: " << m_data->IDNPMTs;
  StreamToLog(DEBUG2);
  for(int i = 0; i <= m_data->IDNPMTs; i++)
    fIDNDigitPerPMTPerTriggerMap[i] = std::map<int, bool>();
  Log("DEBUG: DataOut::Initialise creating OD trigger maps...", DEBUG2, verbose);
  ss << "DEBUG:   entries: " << m_data->ODNPMTs;
  StreamToLog(DEBUG2);
  for(int i = 0; i <= m_data->ODNPMTs; i++)
    fODNDigitPerPMTPerTriggerMap[i] = std::map<int, bool>();

  Log("DEBUG: DataOut::Initialise done", DEBUG1, verbose);
  return true;
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

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
  (*m_data->IDWCSimEvent_Triggered) = (*(m_data->IDWCSimEvent_Raw));
  //prepare the subtriggers
  CreateSubEvents(m_data->IDWCSimEvent_Triggered);
  //remove the digits that aren't in the trigger window(s)
  // also move digits from the 0th trigger to the trigger window it's in
  RemoveDigits(m_data->IDWCSimEvent_Triggered, fIDNDigitPerPMTPerTriggerMap);
  //also redistribute some true tracks
  MoveTracks(m_data->IDWCSimEvent_Triggered);
  //set some trigger header infromation that requires all the digits to be 
  // present to calculate e.g. sumq
  FinaliseSubEvents(m_data->IDWCSimEvent_Triggered);
  
  if(m_data->HasOD) {
    (*m_data->ODWCSimEvent_Triggered) = (*(m_data->ODWCSimEvent_Raw));
    CreateSubEvents(m_data->ODWCSimEvent_Triggered);
    RemoveDigits(m_data->ODWCSimEvent_Triggered, fODNDigitPerPMTPerTriggerMap);
    MoveTracks(m_data->ODWCSimEvent_Triggered);
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
/////////////////////////////////////////////////////////////////
void DataOut::CreateSubEvents(WCSimRootEvent * WCSimEvent)
{
  const int n = fTriggers->m_N;
  for(int i = 0; i < n; i++) {
    if(i)
      WCSimEvent->AddSubEvent();
    WCSimRootTrigger * trig = WCSimEvent->GetTrigger(i);
    double offset = fTriggerOffset;
    trig->SetHeader(fEvtNum, 0, fTriggers->m_triggertime.at(i), i+1);
    trig->SetTriggerInfo(fTriggers->m_type.at(i), fTriggers->m_info.at(i));
    trig->SetMode(0);
  }//i
}
/////////////////////////////////////////////////////////////////
void DataOut::FinaliseSubEvents(WCSimRootEvent * WCSimEvent)
{
  const int n = fTriggers->m_N;
  for(int i = 0; i < n; i++) {
    WCSimRootTrigger * trig = WCSimEvent->GetTrigger(i);
    TClonesArray * digits = trig->GetCherenkovDigiHits();
    double sumq = 0;
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
/////////////////////////////////////////////////////////////////
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
    if(!fSaveOnlyFailedDigits) {
      //we're saving only things in the trigger window
      if(window >= 0) {
	//need to apply an offset to the digit time using the trigger time
	//do it this slightly odd way to mirror what WCSim does
	double t = time;
	t += fTriggerOffset
	  - (float)fTriggers->m_triggertime.at(window);
	d->SetT(t);
      }
      if(window > 0 &&
	 (fSaveMultiDigiPerTrigger ||
	  (!fSaveMultiDigiPerTrigger && !NDigitPerPMTPerTriggerMap[pmt][window]))) {
	//need to add digit to a new trigger window
	//but not if we've already saved the 1 digit from this pmt in this window we're allowed
	ss << "DEBUG: Adding digit to trigger " << window;
	StreamToLog(DEBUG3);
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
    }//!fSaveOnlyFailedDigits
    else {
      //We want to save digits that *haven't* passed any trigger
      //To keep it simple:
      // Remove anything that's in a trigger window
      // Keep everything else in the 0th trigger
      if(window >= 0)
	trig0->RemoveCherenkovDigiHit(d);
    }//fSaveOnlyFailedDigits
  }//i
  ss << "INFO: RemoveDigits() has reduced number of digits in the 0th trigger from "
     << ndigits << " to " << trig0->GetNcherenkovdigihits();
  StreamToLog(INFO);
}
/////////////////////////////////////////////////////////////////
void DataOut::MoveTracks(WCSimRootEvent * WCSimEvent)
{
  if(fTriggers->m_N < 2)
    return;
  WCSimRootTrigger * trig0 = WCSimEvent->GetTrigger(0);
  TClonesArray * tracks = trig0->GetTracks();
  int ntracks = trig0->GetNtrack();
  int ntracks_slots = trig0->GetNtrack_slots();
  for(int i = 0; i < ntracks_slots; i++) {
    WCSimRootTrack * t = (WCSimRootTrack*)tracks->At(i);
    if(!t)
      continue;
    double time = t->GetTime();
    int window = TimeInTriggerWindowNoDelete(time);
    if(window > 0) {
      ss << "DEBUG: Moving track from 0th  to " << window << " trigger";
      StreamToLog(DEBUG3);
      //need to add track to a new trigger window
      WCSimEvent->GetTrigger(window)->AddTrack(t);
      //and remove from the 0th
      trig0->RemoveTrack(t);
    }
  }//i
  ss << "INFO: MoveTracks() has reduced number of tracks in the 0th trigger from "
     << ntracks << " to " << trig0->GetNtrack();
  StreamToLog(INFO);
}
/////////////////////////////////////////////////////////////////
int DataOut::TimeInTriggerWindow(double time) {
  for(unsigned int i = 0; i < fTriggers->m_N; i++) {
    double lo = fTriggers->m_starttime.at(i);
    double hi = fTriggers->m_endtime.at(i);
    if(time >= lo && time <= hi)
      return i;
  }//it
  return -1;
}
/////////////////////////////////////////////////////////////////
unsigned int DataOut::TimeInTriggerWindowNoDelete(double time) {
  //we can't return -1 in this (i.e. we don't want to delete tracks)
  //the logic is:
  // if it's anytime before the 0th trigger + postrigger readout window, store in 0th trigger
  // if it's anytime after the 0th trigger readout window and before the end of the 1st trigger readout window, store in the 1st trigger
  // etc
  //with the caveat that we don't create a WCSimRootTrigger just to store some tracks
  // therefore return value is at maximum the number of triggers
  const int N = fTriggers->m_N;
  for(unsigned int i = 0; i < N; i++) {
    double hi = fTriggers->m_endtime.at(i);
    if(time <= hi)
      return i;
  }//it
  ss << "WARNING DataOut::TimeInTriggerWindowNoDelete() could not find a trigger that track with time " << time << " can live in. Returning maximum trigger number " << N - 1;
  StreamToLog(WARN);
  return N - 1;
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

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
