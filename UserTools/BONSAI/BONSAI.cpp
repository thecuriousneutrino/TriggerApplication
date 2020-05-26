#include "BONSAI.h"

#include "Utilities.h"

BONSAI::BONSAI():Tool(){}

bool BONSAI::Initialise(std::string configfile, DataModel &data){
  
  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("BONSAI") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  if(!util::FileExists(std::getenv("BONSAIDIR"), "libWCSimBonsai.so")) {
    Log("FATAL: BONSAI library not found. Ensure the BONSAI library exists at $BONSAIDIR/libWCSimBonsai.so. For more information about BONSAI, see https://github.com/hyperk/hk-BONSAI", FATAL, m_verbose);
    return false;
  }

  m_variables.Get("nhitsmin", m_nhits_min);
  m_variables.Get("nhitsmax", m_nhits_max);

  //setup BONSAI with the geometry
  m_bonsai = new WCSimBonsai();
  WCSimRootGeom * geo = 0;
  m_data->WCSimGeomTree->SetBranchAddress("wcsimrootgeom", &geo);
  m_data->WCSimGeomTree->GetEntry(0);
  m_bonsai->Init(geo);
  m_data->WCSimGeomTree->ResetBranchAddresses();

  //allocate memory for the hit vectors
  m_in_PMTIDs = new std::vector<int>  (m_nhits_max);
  m_in_Ts     = new std::vector<float>(m_nhits_max);
  m_in_Qs     = new std::vector<float>(m_nhits_max);

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool BONSAI::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  float out_vertex[4], out_direction[6], out_maxlike[3];
  int   out_nsel[2];
  double dout_vertex[3], dout_direction[3], dout_cone[2];

  //loop over ID triggers
  for (int itrigger = 0; itrigger < m_data->IDTriggers.m_num_triggers; itrigger++) {
    //clear the previous triggers' hit information
    m_in_PMTIDs->clear();
    m_in_Ts->clear();
    m_in_Qs->clear();

    //Loop over ID SubSamples
    for(std::vector<SubSample>::iterator is = m_data->IDSamples.begin(); is != m_data->IDSamples.end(); ++is){

      //fill the inputs to BONSAI with the current triggers' hit information
      //loop over all hits
      const size_t nhits_in_subsample = is->m_time.size();
      //starting at m_first_unique, rather than 0, to avoid double-counting hits
      // that are in multiple SubSamples
      for(size_t ihit = is->m_first_unique; ihit < nhits_in_subsample; ihit++) {
	//see if the hit belongs to this trigger
	if(std::find(is->m_trigger_readout_windows[ihit].begin(),
		     is->m_trigger_readout_windows[ihit].end(),
		     itrigger) == is->m_trigger_readout_windows[ihit].end())
	  continue;

	//it belongs. Add it to the BONSAI input arrays
	m_ss << "DEBUG: Hit " << ihit << " at time " << is->m_time[ihit];
	StreamToLog(DEBUG2);
	m_in_PMTIDs->push_back(is->m_PMTid[ihit]);
	m_in_Ts    ->push_back(is->m_time[ihit]);
	m_in_Qs    ->push_back(is->m_charge[ihit]);
      }//ihit
    }//ID SubSamples
  
    //get the number of hits
    m_in_nhits = m_in_PMTIDs->size();

    //don't run bonsai on large or small events
    if(m_in_nhits < m_nhits_min || m_in_nhits > m_nhits_max) {
      m_ss << "INFO: " << m_in_nhits << " hits in current trigger. Not running BONSAI";
      StreamToLog(INFO);
      continue;
    }
    
    m_ss << "DEBUG: BONSAI running over " << m_in_nhits << " hits";
    StreamToLog(DEBUG1);
    m_ss << "DEBUG: First hit time relative to sample: " << m_in_Ts->at(0);
    StreamToLog(DEBUG1);

    //call BONSAI
    bool success = true;
    try {
      m_bonsai->BonsaiFit( out_vertex, out_direction, out_maxlike, out_nsel, &m_in_nhits, m_in_PMTIDs->data(), m_in_Ts->data(), m_in_Qs->data());
    } catch (int e) {
      Log("BONSAI threw an exception!", WARN, m_verbose);
      success = false;
    }

    if (success) {
      m_ss << "DEBUG: Vertex reconstructed at x, y, z, t:";
      for(int i = 0; i < 4; i++) {
        m_ss << " " << out_vertex[i] << ",";
      }
      StreamToLog(DEBUG1);

      //fill the data model with the result
      // need to convert to double...
      for(int i = 0; i < 3; i++) {
        dout_vertex[i]    = out_vertex[i];
        dout_direction[i] = out_direction[i];
      }
      for(int i = 0; i < 2; i++)
        dout_cone[i] = out_direction[i+3];

      TimeDelta vertex_time = m_data->IDTriggers.m_trigger_time.at(itrigger) + TimeDelta(out_vertex[3] * TimeDelta::ns);
      m_data->RecoInfo.AddRecon(kReconBONSAI, itrigger, m_in_nhits,
                                vertex_time, &(dout_vertex[0]), out_maxlike[2], out_maxlike[1],
                                &(dout_direction[0]), &(dout_cone[0]), out_direction[5]);
    }
  }//itrigger

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool BONSAI::Finalise(){

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  delete m_bonsai;
  delete m_in_PMTIDs;
  delete m_in_Ts;
  delete m_in_Qs;

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }
  
  return true;
}
