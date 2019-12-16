#include "ReconDataOut.h"

ReconDataOut::ReconDataOut():Tool(){}


bool ReconDataOut::Initialise(std::string configfile, DataModel &data){

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

  //open & format the tree
  fTreeRecon = new TTree("reconTree", "Reconstruction information");
  fTreeRecon->Branch("EventNum", &fEvtNum);
  fTreeRecon->Branch("TriggerNum", &fRTTriggerNum);
  fTreeRecon->Branch("NDigits", &fRTNHits);
  fTreeRecon->Branch("Reconstructer", &fRTReconstructer, "Reconstruter/I");
  fTreeRecon->Branch("Time", &fRTTime);
  fTreeRecon->Branch("Vertex", fRTVertex, "Vertex[3]/D");
  fTreeRecon->Branch("GoodnessOfFit", &fRTGoodnessOfFit);
  fTreeRecon->Branch("GoodnessOfTimeFit", &fRTGoodnessOfTimeFit);
  fTreeRecon->Branch("HasDirection", &fRTHasDirection);
  fTreeRecon->Branch("DirectionEuler", fRTDirectionEuler, "DirectionEuler[3]/D");
  fTreeRecon->Branch("CherenkovCone", fRTCherenkovCone, "CherenkovCone[2]/D");
  fTreeRecon->Branch("DirectionLikelihood", &fRTDirectionLikelihood);

  fEvtNum = 0;

  return true;
}


bool ReconDataOut::Execute(){

  Log("DEBUG: ReconDataOut::Execute() Starting", DEBUG1, verbose);

  const int nrecons = m_data->RecoInfo.GetNRecons();
  ss << "DEBUG: Saving the result of " << nrecons << " reconstructions";
  StreamToLog(DEBUG1);
  for(int irecon = 0; irecon < nrecons; irecon++) {
    fRTTriggerNum = m_data->RecoInfo.GetTriggerNum(irecon);
    fRTNHits = m_data->RecoInfo.GetNHits(irecon);
    fRTReconstructer = m_data->RecoInfo.GetReconstructer(irecon);
    fRTTime = m_data->RecoInfo.GetTime(irecon);
    Pos3D pos = m_data->RecoInfo.GetVertex(irecon);
    fRTVertex[0] = pos.x;
    fRTVertex[1] = pos.y;
    fRTVertex[2] = pos.z;
    fRTGoodnessOfFit = m_data->RecoInfo.GetGoodnessOfFit(irecon);
    fRTGoodnessOfTimeFit = m_data->RecoInfo.GetGoodnessOfTimeFit(irecon);
    //Direction
    fRTHasDirection = m_data->RecoInfo.GetHasDirection(irecon);
    if(fRTHasDirection) {
      DirectionEuler direct = m_data->RecoInfo.GetDirectionEuler(irecon);
      fRTDirectionEuler[0] = direct.theta;
      fRTDirectionEuler[1] = direct.phi;
      fRTDirectionEuler[2] = direct.alpha;
      CherenkovCone cone = m_data->RecoInfo.GetCherenkovCone(irecon);
      fRTCherenkovCone[0] = cone.cos_angle;
      fRTCherenkovCone[1] = cone.ellipticity;
      fRTDirectionLikelihood = m_data->RecoInfo.GetDirectionLikelihood(irecon);
    }
    else {
      for(int i = 0; i < 3; i++)
	fRTDirectionEuler[i] = -9E7;
      for(int i = 0; i < 2; i++)
	fRTCherenkovCone [i] = -9E7;
    }
    //Fill the tree
    fTreeRecon->Fill();
  }
  m_data->RecoInfo.Reset();

  //increment event number
  fEvtNum++;

  Log("DEBUG: ReconDataOut::Execute() Done", DEBUG1, verbose);

  return true;
}


bool ReconDataOut::Finalise(){
  //multiple TFiles may be open. Ensure we save to the correct one
  fOutFile.cd(TString::Format("%s:/", fOutFilename.c_str()));
  fTreeRecon->Write();
  fOutFile.Close();
  delete fTreeRecon;

  return true;
}
