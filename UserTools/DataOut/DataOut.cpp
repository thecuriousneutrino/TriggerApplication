#include "DataOut.h"
#include "TimeDelta.h"

DataOut::DataOut():Tool(){}

/////////////////////////////////////////////////////////////////
bool DataOut::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("DataOut") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  //open the output file
  Log("DEBUG: DataOut::Initialise opening output file...", DEBUG2, m_verbose);
  if(! m_variables.Get("outfilename", m_output_filename)) {
    Log("ERROR: outfilename configuration not found. Cancelling initialisation", ERROR, m_verbose);
    return false;
  }
  m_output_file = new TFile(m_output_filename.c_str(), "RECREATE");

  //other options
  m_save_multiple_hits_per_trigger = true;
  m_variables.Get("save_multiple_hits_per_trigger", m_save_multiple_hits_per_trigger);
  Log("WARN: TODO save_multiple_hits_per_trigger is not currently implemented", WARN, m_verbose);
  double trigger_offset_temp = 0;
  m_variables.Get("trigger_offset", trigger_offset_temp);
  m_trigger_offset = TimeDelta(trigger_offset_temp);
  m_save_only_failed_hits = false;
  m_variables.Get("save_only_failed_hits", m_save_only_failed_hits);
  Log("WARN: TODO save_only_failed_hits is not currently implemented", WARN, m_verbose);

  //setup the out event tree
  Log("DEBUG: DataOut::Initialise setting up output event tree...", DEBUG2, m_verbose);
  // Nevents unique event objects
  Int_t bufsize = 64000;
  m_event_tree = new TTree("wcsimT","WCSim Tree");
  m_id_wcsimevent_triggered = new WCSimRootEvent();
  m_id_wcsimevent_triggered->Initialize();
  m_event_tree->Branch("wcsimrootevent", "WCSimRootEvent", &m_id_wcsimevent_triggered, bufsize,2);
  if(m_data->HasOD) {
    m_od_wcsimevent_triggered = new WCSimRootEvent();
    m_od_wcsimevent_triggered->Initialize();
    m_event_tree->Branch("wcsimrootevent_OD", "WCSimRootEvent", &m_od_wcsimevent_triggered, bufsize,2);
  }
  else {
    m_od_wcsimevent_triggered = 0;
  }
  m_event_tree->Branch("wcsimfilename", &(m_data->CurrentWCSimFile));
  m_event_tree->Branch("wcsimeventnum", &(m_data->CurrentWCSimEventNum));

  //fill the output event-independent trees
  Log("DEBUG: DataOut::Initialise filling event-independent trees...", DEBUG2, m_verbose);

  //There are 1 unique geom objects, so this is a simple clone of 1 entry
  if(m_data->IsMC) {
    Log("DEBUG:   Geometry...", DEBUG2, m_verbose);
    TTree * geom_tree = m_data->WCSimGeomTree->CloneTree(1);
    geom_tree->Write();
    delete geom_tree;
  }
  else {
    Log("WARN: TODO Geometry tree filling is not yet implemented for data");
  }

  //There are Nfiles unique options objects, so this is a clone of all entries
  // plus a new branch with the wcsim filename 
  if(m_data->IsMC) {
    Log("DEBUG:   Options & file names...", DEBUG2, m_verbose);
    TTree * options_tree = m_data->WCSimOptionsTree->CloneTree();
    m_ss << "DEBUG:     entries: " << options_tree->GetEntries();
    StreamToLog(DEBUG2);
    TObjString * wcsimfilename = new TObjString();
    TBranch * branch = options_tree->Branch("wcsimfilename", &wcsimfilename);
    for(int i = 0; i < options_tree->GetEntries(); i++) {
      m_data->WCSimOptionsTree->GetEntry(i);
      options_tree->GetEntry(i);
      wcsimfilename->SetString(m_data->WCSimOptionsTree->GetFile()->GetName());
      branch->Fill();
    }//i
    options_tree->Write();
    delete wcsimfilename;
    delete options_tree;
  }

  Log("DEBUG: DataOut::Initialise creating trigger info...", DEBUG2, m_verbose);
  m_all_triggers = new TriggerInfo();
  m_event_num = 0;

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

bool DataOut::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  //Gather together all the trigger windows
  m_all_triggers->Clear();
  m_all_triggers->AddTriggers(&(m_data->IDTriggers));
  if(m_data->HasOD)
    m_all_triggers->AddTriggers(&(m_data->ODTriggers));
  m_ss << "INFO: Have " << m_all_triggers->m_num_triggers << " triggers to save times:";
  StreamToLog(INFO);
  for(int i = 0; i < m_all_triggers->m_num_triggers; i++) {
    m_ss << "INFO: \t[" << m_all_triggers->m_readout_start_time.at(i)
       << ", " << m_all_triggers->m_readout_end_time.at(i) << "] "
       << m_all_triggers->m_trigger_time.at(i) << " ns with type "
       << WCSimEnumerations::EnumAsString(m_all_triggers->m_type.at(i)) << " extra info";
    for(unsigned int ii = 0; ii < m_all_triggers->m_info.at(i).size(); ii++)
      m_ss << " " << m_all_triggers->m_info.at(i).at(ii);
    StreamToLog(INFO);
  }//i

  //Note: the trigger ranges vector can contain overlapping ranges
  //we want to make sure the triggers output aren't overlapping
  // This is actually handled in DataOut::FillHits()
  // It puts hits into the output event in the earliest trigger they belong to

  //Fill the WCSimRootEvent with hit/trigger information, and truth information (if available)
  if(m_all_triggers->m_num_triggers) {
    ExecuteSubDet(m_id_wcsimevent_triggered, m_data->IDSamples, m_data->IDWCSimEvent_Raw);
    if(m_data->HasOD) {
      ExecuteSubDet(m_od_wcsimevent_triggered, m_data->ODSamples, m_data->ODWCSimEvent_Raw);
    }
  }//>=1 trigger found

  //Fill the tree with what we've just created
  m_event_tree->Fill();

  //increment event number
  m_event_num++;

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}
/////////////////////////////////////////////////////////////////
void DataOut::ExecuteSubDet(WCSimRootEvent * wcsim_event, std::vector<SubSample> & samples, WCSimRootEvent * original_wcsim_event) {
  //ReInitialise the WCSimRootEvent
  // This clears the TObjArray of WCSimRootTrigger's,
  // and creates a WCSimRootTrigger in the first slot
  wcsim_event->ReInitialize();

  //If there are multiple triggers in the event,
  // create subevents (i.e. new WCSimRootTrigger's) in the WCSimRootEvent
  //Also sets the time correctly
  CreateSubEvents(wcsim_event);

  //Get the WCSim "date", used later to give the hits the correct absolute time.
  // Also add the trigger offset from the config file
  TimeDelta time_shift = GetOffset(original_wcsim_event);

  //For every hit, if it's in a trigger window,
  //add it to the appropriate WCSimRootTrigger in the WCSimRootEvent
  FillHits(wcsim_event, samples);

  //If this is an MC file, we also need to add
  // - true tracks
  // - true hits
  if(m_data->IsMC)
    AddTruthInfo(wcsim_event, original_wcsim_event, time_shift);

  //Set some trigger header infromation that requires all the hits to be 
  // present to calculate e.g. sumq
  FinaliseSubEvents(wcsim_event);  
}
/////////////////////////////////////////////////////////////////
void DataOut::CreateSubEvents(WCSimRootEvent * wcsim_event) {
  const int n = m_all_triggers->m_num_triggers;
  if (n==0) return;

  // Change trigger times and create new SubEvents where necessary
  for(int i = 0; i < n; i++) {
    if(i)
      wcsim_event->AddSubEvent();
    WCSimRootTrigger * trig = wcsim_event->GetTrigger(i);
    trig->SetHeader(m_event_num, 0, (m_all_triggers->m_trigger_time.at(i) / TimeDelta::ns), i+1);
    trig->SetTriggerInfo(m_all_triggers->m_type.at(i), m_all_triggers->m_info.at(i));
    trig->SetMode(0);
  }//i
}
/////////////////////////////////////////////////////////////////
TimeDelta DataOut::GetOffset(WCSimRootEvent * original_wcsim_event) {

  TimeDelta time_shift(0);

  if(m_data->IsMC) {
    // Get the original event trigger time ("date"). Subtraction of this is used
    // when we want to store the hits with their new trigger offsets
    WCSimRootTrigger * trig0 = original_wcsim_event->GetTrigger(0);
    // The old time (stored in ns)
    TimeDelta old_trigger_time = trig0->GetHeader()->GetDate() * TimeDelta::ns;
    time_shift += old_trigger_time;
    m_ss << "DEBUG: Trigger date shift from input WCSim file is " << old_trigger_time;
    StreamToLog(DEBUG2);
  }

  m_ss << "DEBUG: Adding additional user-defined time shift of "
       << m_trigger_offset;
  StreamToLog(DEBUG2);

  time_shift += m_trigger_offset;
  m_ss << "DEBUG: Total time shift is " << time_shift;
  StreamToLog(DEBUG1);

  return time_shift;
}
/////////////////////////////////////////////////////////////////
void DataOut::FillHits(WCSimRootEvent * wcsim_event, std::vector<SubSample> & samples) {
  unsigned int trigger_window;
  TimeDelta time;
  std::vector<int> photon_id_temp;
  WCSimRootTrigger * wcsim_trigger;
  //Loop over all SubSamples
  for(std::vector<SubSample>::iterator is=samples.begin(); is!=samples.end(); ++is){
    const size_t nhits = is->m_time.size();
    unsigned int counter = 0;
    for(size_t ihit = 0; ihit < nhits; ihit++) {
      //skip if hit is not in a readout window
      if(!is->m_trigger_readout_windows[ihit].size())
	continue;

      //Find out which window it's in.
      // We're taking the first one it's associated with
      trigger_window = is->m_trigger_readout_windows[ihit][0];
      wcsim_trigger = wcsim_event->GetTrigger(trigger_window);

      //Get the time
      time = is->AbsoluteDigitTime(ihit);
      m_ss << "Hit " << ihit << " is at time " << time << std::endl;
      StreamToLog(DEBUG3);

      //Apply the time offsets
      // + m_trigger_offset adds the user-defined "offset"
      // - trigger time ("Date") because hits are defined relative to their trigger time
      time += m_trigger_offset -
	(wcsim_trigger->GetHeader()->GetDate() * TimeDelta::ns);

      //hit is in this window. Let's save it
      wcsim_trigger->AddCherenkovDigiHit(is->m_charge[ihit],
					 time / TimeDelta::ns,
					 is->m_PMTid[ihit],
					 photon_id_temp);
      m_ss << "Saved hit " << counter++;
      StreamToLog(DEBUG3);
    }//ihit
  }//loop over SubSamples
}
/////////////////////////////////////////////////////////////////
void DataOut::AddTruthInfo(WCSimRootEvent * wcsim_event, WCSimRootEvent * original_wcsim_event, const TimeDelta & time_shift) {
  //get the "triggers", where everything is stored
  WCSimRootTrigger * new_trig = wcsim_event->GetTrigger(0);
  WCSimRootTrigger * old_trig = original_wcsim_event->GetTrigger(0);

  //set vertex info
  const int nvtx = old_trig->GetNvtxs();
  new_trig->SetNvtxs(nvtx);
  for(int ivtx = 0; ivtx < nvtx; ivtx++) {
    new_trig->SetVtxsvol(ivtx, old_trig->GetVtxsvol(ivtx));
    for(int idim = 0; idim < 3; idim++) {
      new_trig->SetVtxs(ivtx, idim, old_trig->GetVtxs(ivtx, idim));
    }//idim
  }//ivtx

  //set generic event info
  new_trig->SetMode(old_trig->GetMode());
  new_trig->SetVecRecNumber(old_trig->GetVecRecNumber());
  new_trig->SetJmu(old_trig->GetJmu());
  new_trig->SetJp(old_trig->GetJp());
  new_trig->SetNpar(old_trig->GetNpar());

  //set pi0 info
  const WCSimRootPi0 * pi0 = old_trig->GetPi0Info();
  Double_t pi0_vtx[3];
  Int_t   pi0_gamma_id[2];
  Double_t pi0_gamma_e[2];
  Double_t pi0_gamma_vtx[2][3];
  for(int i = 0; i < 3; i++)
    pi0_vtx[i] = pi0->GetPi0Vtx(i);
  for(int i = 0; i < 2; i++) {
    pi0_gamma_id[i] = pi0->GetGammaID(i);
    pi0_gamma_e [i] = pi0->GetGammaE (i);
    for(int j = 0; j < 3; j++)
      pi0_gamma_vtx[i][j] = pi0->GetGammaVtx(i, j);
  }//i
  new_trig->SetPi0Info(pi0_vtx, pi0_gamma_id, pi0_gamma_e, pi0_gamma_vtx);

  //set true hit info
  new_trig->SetNumTubesHit(old_trig->GetNumTubesHit());
  WCSimRootCherenkovHit * hit;
  WCSimRootCherenkovHitTime * hit_time;
  for(int ihit = 0; ihit < old_trig->GetNcherenkovhits(); ihit++) {
    TObject * obj = old_trig->GetCherenkovHits()->At(ihit);
    if(!obj) continue;
    hit = dynamic_cast<WCSimRootCherenkovHit*>(obj);
    int tube_id = hit->GetTubeID();
    std::vector<double> true_times;
    std::vector<int>   primary_parent_id;
    for(int itime = hit->GetTotalPe(0); itime < hit->GetTotalPe(0) + hit->GetTotalPe(1); itime++) {
      TObject * obj = old_trig->GetCherenkovHitTimes()->At(itime);
      if(!obj) continue;
      hit_time = dynamic_cast<WCSimRootCherenkovHitTime*>(obj);
      true_times       .push_back(hit_time->GetTruetime());
      primary_parent_id.push_back(hit_time->GetParentID());
    }//itime
    new_trig->AddCherenkovHit(tube_id, true_times, primary_parent_id);
  }//ihit

  //set true track info
  WCSimRootTrack * track;
  for(int itrack = 0; itrack < old_trig->GetNtrack_slots(); itrack++) {
    TObject * obj = old_trig->GetTracks()->At(itrack);
    if(!obj) continue;
    track = dynamic_cast<WCSimRootTrack*>(obj);
    new_trig->AddTrack(track);
  }//itrack

  //set true digit parent info
  // This is messy, since the WCSim digit order
  // is different to the TriggerApplication digit order
  //we're looping over triggers here
  for(int itrigger = 0; itrigger < m_all_triggers->m_num_triggers; itrigger++) {
    if(itrigger)
      new_trig = wcsim_event->GetTrigger(itrigger);
    double this_trigger_time = new_trig->GetHeader()->GetDate();
    WCSimRootCherenkovDigiHit * new_digit;
    WCSimRootCherenkovDigiHit * old_digit;
    //not looping over "slots", because we just wrote this file, so we know there are no gaps
    for(int idigit = 0; idigit < new_trig->GetNcherenkovdigihits(); idigit++) {
      TObject * obj = new_trig->GetCherenkovDigiHits()->At(idigit);
      if(!obj) continue;
      new_digit = dynamic_cast<WCSimRootCherenkovDigiHit*>(obj);
      int tube_id = new_digit->GetTubeId();
      //get the time
      double time = new_digit->GetT();
      //and convert it back to the original (i.e. relative to WCSim's Date(), rather than 
      // TriggerApp's individual triggers' Date()s)
      // - time_shift subtracts the WCSim Date, and subtracts the user-defined "offset"
      // + this_trigger_time because hits are defined relative to their trigger time
      time += - (time_shift / TimeDelta::ns) + this_trigger_time;

      //Now we find the new digit
      bool found = false;
      for(int idigit_old = 0; idigit_old < old_trig->GetNcherenkovdigihits_slots(); idigit_old++) {
	TObject * obj = old_trig->GetCherenkovDigiHits()->At(idigit_old);
	if(!obj) continue;
	old_digit = dynamic_cast<WCSimRootCherenkovDigiHit*>(obj);
	// First check tube id
	if(tube_id != old_digit->GetTubeId()) continue;
	// Then the time. Assume if we're within 0.5 ns it's the same hit
	if(abs(time - old_digit->GetT()) > 0.5) continue;
	found = true;
	break;
      }//idigit_old
      if(found) {
	//new_digit->SetPhotonIds(old_digit->GetPhotonIds());
      }
    }//idigit
  }//itrigger
  Log("TODO after WCSim/WCSim#286 is merged, uncomment SetPhotonIds", WARN, m_verbose);
}
/////////////////////////////////////////////////////////////////
void DataOut::FinaliseSubEvents(WCSimRootEvent * wcsim_event) {
  const int n = m_all_triggers->m_num_triggers;
  for(int i = 0; i < n; i++) {
    WCSimRootTrigger * trig = wcsim_event->GetTrigger(i);
    TClonesArray * hits = trig->GetCherenkovDigiHits();
    double sumq = 0;
    int nhits = 0;
    for(int j = 0; j < trig->GetNcherenkovdigihits_slots(); j++) {
      WCSimRootCherenkovDigiHit * digi = (WCSimRootCherenkovDigiHit *)hits->At(j);
      if(digi) {
	sumq += digi->GetQ();
	nhits++;
      }
    }//j
    trig->SetSumQ(sumq);
    //this is actually number of hits, not number of unique PMTs with hits
    trig->SetNumDigitizedTubes(nhits);
    m_ss << "DEBUG: Trigger " << i << " has " << nhits << " hits";
    StreamToLog(DEBUG1);
  }//i
}
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

bool DataOut::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  //multiple TFiles may be open. Ensure we save to the correct one
  m_output_file->cd(TString::Format("%s:/", m_output_filename.c_str()));
  m_event_tree->Write();

  delete m_event_tree;
  delete m_id_wcsimevent_triggered;
  if(m_od_wcsimevent_triggered)
    delete m_od_wcsimevent_triggered;

  delete m_all_triggers;

  m_output_file->Close();
  delete m_output_file;

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
