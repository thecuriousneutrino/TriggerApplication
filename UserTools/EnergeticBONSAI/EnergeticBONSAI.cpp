#include "EnergeticBONSAI.h"

EnergeticBONSAI::EnergeticBONSAI():Tool(){}


bool EnergeticBONSAI::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("EnergeticBONSAI") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  //setup energetic BONSAI with the geometry info
  int ebonsai_verbose = 0;
  m_variables.Get("ebonsai_verbose", ebonsai_verbose);
  WCSimRootGeom * geo = 0;
  m_data->WCSimGeomTree->SetBranchAddress("wcsimrootgeom", &geo);
  m_data->WCSimGeomTree->GetEntry(0);
  m_variables.Get("detector_name", m_detector_name);
  Log("TODO: detector_name should come from the geometry, rather than the parameter file", WARN, m_verbose);
  m_overwrite_nearest = false;
  m_variables.Get("overwrite_nearest_neighbours", m_overwrite_nearest);
  m_ebonsai = new WCSimEBonsai(m_detector_name.c_str(), geo, m_overwrite_nearest, ebonsai_verbose);

  //override any energetic BONSAI assumptions
  m_ebonsai->SetDarkRate(m_data->IDPMTDarkRate);
  m_ebonsai->SetNPMTs(m_data->IDNPMTs);
  
  //tell energetic BONSAI how many PMTs are turned off
  m_n_working_pmts = m_data->IDNPMTs;
  m_variables.Get("n_working_pmts", m_n_working_pmts);
  if(m_n_working_pmts > m_data->IDNPMTs) {
    m_ss << "WARN: Config value of number of working PMTs " << m_n_working_pmts
	 << " is more than the total number of PMTs " << m_data->IDNPMTs
	 << ". Setting the number of working PMTs to the total number of PMTs";
    StreamToLog(WARN);
    m_n_working_pmts = m_data->IDNPMTs;
  }
  m_ebonsai->SetNWorkingPMTs(m_n_working_pmts);

  //Get the reconstructed events filter you want to save
  if(!m_variables.Get("input_filter_name", m_input_filter_name)) {
    Log("INFO: input_filter_name not given. Using ALL", WARN, m_verbose);
    m_input_filter_name = "ALL";
  }
  m_input_filter  = m_data->GetFilter(m_input_filter_name, false);
  if(!m_input_filter) {
    m_ss << "FATAL: no filter named " << m_input_filter_name << " found. Returning false";
    StreamToLog(FATAL);
    return false;
  }

  //allocate memory for the hit vectors
  m_variables.Get("nhitsmin", m_nhits_min);
  m_variables.Get("nhitsmax", m_nhits_max);
  m_in_PMTIDs = new std::vector<int>  (m_nhits_max);
  m_in_Ts     = new std::vector<float>(m_nhits_max);

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool EnergeticBONSAI::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  //Loop over reconstructed objects. Each should have a vertex associated to a trigger
  for(int ireco = 0; ireco < m_input_filter->GetNRecons(); ireco++) {
    //get the vertex
    Pos3D vertex = m_input_filter->GetVertex(ireco);
    m_vertex[0] = vertex.x;
    m_vertex[1] = vertex.y;
    m_vertex[2] = vertex.z;

    //get the trigger number this reconstructed object is associated with
    const int trigger_num = m_input_filter->GetTriggerNum(ireco);

    //clear the previous triggers' hit information
    m_in_PMTIDs->clear();
    m_in_Ts->clear();

    //fill the inputs to BONSAI with the current triggers' hit information
    //Loop over SubSamples
    for(std::vector<SubSample>::iterator is = m_data->IDSamples.begin(); is != m_data->IDSamples.end(); ++is){
      //loop over hits
      const size_t nhits_in_subsample = is->m_time.size();
      //starting at m_first_unique, rather than 0, to avoid double-counting hits
      // that are in multiple SubSamples
      for(size_t ihit = is->m_first_unique; ihit < nhits_in_subsample; ihit++) {
	//see if the hit belongs to this trigger
	if(std::find(is->m_trigger_readout_windows[ihit].begin(),
		     is->m_trigger_readout_windows[ihit].end(),
		     trigger_num) == is->m_trigger_readout_windows[ihit].end())
	  continue;

	//it belongs. Add it to the BONSAI input arrays
	m_ss << "DEBUG: Hit " << ihit << " at time " << is->m_time[ihit];
	StreamToLog(DEBUG2);
	m_in_PMTIDs->push_back(is->m_PMTid[ihit]);
	m_in_Ts    ->push_back(is->m_time[ihit]);
      }//ihit
    }//SubSamples

    //get the number of hits
    m_in_nhits = m_in_PMTIDs->size();

    //don't run energetic bonsai on large or small events
    if(m_in_nhits < m_nhits_min || m_in_nhits > m_nhits_max) {
      m_ss << "INFO: " << m_in_nhits << " hits in current trigger. Not running Energetic BONSAI";
      StreamToLog(INFO);
      continue;
    }

    m_ss << "DEBUG: Energetic BONSAI running over " << m_in_nhits << " hits";
    StreamToLog(DEBUG1);

    //get the energy
    double energy = m_ebonsai->GetEnergy(*m_in_Ts, *m_in_PMTIDs, &(m_vertex[0]));

    m_ss << "INFO: Energetic BONSAI reconstructed energy " << energy;
    StreamToLog(INFO);

    m_input_filter->SetEnergy(ireco, energy);

  }//ireco

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool EnergeticBONSAI::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  delete m_ebonsai;
  delete m_in_PMTIDs;
  delete m_in_Ts;

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
