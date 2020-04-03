#include "BONSAI.h"

BONSAI::BONSAI():Tool(){}

bool BONSAI::FileExists(std::string pathname, std::string filename) {
  string filepath = pathname + "/" + filename;
  bool exists = access(filepath.c_str(), F_OK) != -1;
  if(!exists) {
    m_ss << "FATAL: " << filepath << " not found or inaccessible";
    StreamToLog(FATAL);
    return false;
  }
  return true;
}

bool BONSAI::Initialise(std::string configfile, DataModel &data){
  
  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  m_data= &data;

  if(!FileExists(std::getenv("BONSAIDIR"), "libWCSimBonsai.so")) {
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

  //allocate memory for the digit vectors
  m_in_PMTIDs = new std::vector<int>  (m_nhits_max);
  m_in_Ts     = new std::vector<float>(m_nhits_max);
  m_in_Qs     = new std::vector<float>(m_nhits_max);

  return true;
}


bool BONSAI::Execute(){

  float out_vertex[4], out_direction[6], out_maxlike[500];
  int   out_nsel[2];
  double dout_vertex[3], dout_direction[3], dout_cone[2];
  
  for (int itrigger = 0 ; itrigger < m_data->IDWCSimEvent_Triggered->GetNumberOfEvents(); itrigger++) {
    m_trigger = m_data->IDWCSimEvent_Triggered->GetTrigger(itrigger);

    //clear the previous triggers' digit information
    m_in_PMTIDs->clear();
    m_in_Ts->clear();
    m_in_Qs->clear();

    //get the number of digits
    m_in_nhits = m_trigger->GetNcherenkovdigihits();
    int nhits_slots = m_trigger->GetNcherenkovdigihits_slots();

    //don't run bonsai on large or small events
    if(m_in_nhits < m_nhits_min || m_in_nhits > m_nhits_max) {
      m_ss << "INFO: " << m_in_nhits << " digits in current trigger. Not running BONSAI";
      StreamToLog(INFO);
      return true;
    }

    //fill the inputs to BONSAI with the current triggers' digit information
    long n_not_found = 0;
    for (long idigi=0; idigi < nhits_slots; idigi++) {
      TObject *element = (m_trigger->GetCherenkovDigiHits())->At(idigi);
      WCSimRootCherenkovDigiHit *digi = 
	dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
      if(!digi) {
	n_not_found++;
	//this happens regularly because removing digits doesn't shrink the TClonesArray
	m_ss << "DEBUG: Digit " << idigi << " of " << m_in_nhits << " not found in WCSimRootTrigger";
	StreamToLog(DEBUG2);
	continue;
      }
      m_ss << "DEBUG: Digit " << idigi << " at time " << digi->GetT();
      StreamToLog(DEBUG2);
      m_in_PMTIDs->push_back(digi->GetTubeId());
      m_in_Ts    ->push_back(digi->GetT());
      m_in_Qs    ->push_back(digi->GetQ());
    }//idigi
    int digits_found = nhits_slots - n_not_found;
    if(m_in_nhits != digits_found) {
      m_ss << "WARN: BONSAI expected " << m_in_nhits << " digits. Found " << digits_found;
      StreamToLog(WARN);
      m_in_nhits = digits_found;
    }
    
    m_ss << "DEBUG: BONSAI running over " << m_in_nhits << " digits";
    StreamToLog(DEBUG1);

    //call BONSAI
    m_bonsai->BonsaiFit( out_vertex, out_direction, out_maxlike, out_nsel, &m_in_nhits, m_in_PMTIDs->data(), m_in_Ts->data(), m_in_Qs->data());

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
    
    m_data->RecoInfo.AddRecon(kReconBONSAI, itrigger, m_in_nhits, out_vertex[3], &(dout_vertex[0]), out_maxlike[2], out_maxlike[1],
			      &(dout_direction[0]), &(dout_cone[0]), out_direction[5]);

  }//itrigger

  return true;
}


bool BONSAI::Finalise(){

  delete m_bonsai;
  delete m_in_PMTIDs;
  delete m_in_Ts;
  delete m_in_Qs;
  
  return true;
}
