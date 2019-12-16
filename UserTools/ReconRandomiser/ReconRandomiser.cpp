#include "ReconRandomiser.h"

#include "TMath.h"

ReconRandomiser::ReconRandomiser():Tool(){}


bool ReconRandomiser::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  if(!m_variables.Get("n_vertices_mean", fNVerticesMean)) {
    Log("FATAL: Must specify n_vertices_mean", FATAL, verbose);
    return false;
  }

  if(!m_variables.Get("x_mean_pos", fXMean)) {
    Log("FATAL: Must specify x_mean_pos", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("x_width", fXWidth)) {
    Log("FATAL: Must specify x_width", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("y_mean_pos", fYMean)) {
    Log("FATAL: Must specify y_mean_pos", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("y_width", fYWidth)) {
    Log("FATAL: Must specify y_width", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("z_mean_pos", fZMean)) {
    Log("FATAL: Must specify z_mean_pos", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("z_width", fZWidth)) {
    Log("FATAL: Must specify z_width", FATAL, verbose);
    return false;
  }

  if(!m_variables.Get("max_z_pos", fMaxZPos)) {
    Log("FATAL: Must specify max_z_pos", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("max_r_pos", fMaxRPos)) {
    Log("FATAL: Must specify max_r_pos", FATAL, verbose);
    return false;
  }

  if(!m_variables.Get("flat_r", fFlatR)) {
    Log("FATAL: Must specify flat_r", FATAL, verbose);
    return false;
  }

  if(fXWidth < 1E-6)
    fUniformX = true;
  if(fYWidth < 1E-6)
    fUniformY = true;
  if(fZWidth < 1E-6)
    fUniformZ = true;

  if(!m_variables.Get("t_min", fTMin)) {
    Log("FATAL: Must specify t_min", FATAL, verbose);
    return false;
  }
  if(!m_variables.Get("t_max", fTMax)) {
    Log("FATAL: Must specify t_max", FATAL, verbose);
    return false;
  }

  int seed = 0;
  if(!m_variables.Get("seed", seed)) {
    Log("WARN: No seed specified. Using default 0. Your results are not reproducable!", FATAL, verbose);
  }
  fRand = new TRandom3(seed);

  fRandomDirection = false;
  //TODO set this to true if the user wants random directions

  return true;
}


bool ReconRandomiser::Execute(){

  //Determine how many vertices to generate
  const int N = fRand->Poisson(fNVerticesMean);
  const double likelihood = 9E7;
  const int nhits = 1E4;
  double pos[3];
  for(int iv = 0; iv < N; iv++) {
    CreateVertex(pos);
    double time = fRand->Uniform(fTMin, fTMax);

    ss << "DEBUG: Created event at x,y,z";
    for(int i = 0; i < 3; i++)
      ss << " " << pos[i];
    ss << " time " << time;
    StreamToLog(DEBUG1);

    if(!fRandomDirection) {
      //fill the transient data model
      m_data->RecoInfo.AddRecon(kReconRandomNoDirection, iv, nhits, time, pos, likelihood, likelihood);
    }
    else {
      //m_data->RecoInfo.AddRecon(kReconRandom, iv, nhits, time, pos, likelihood, likelihood, ....);
    }
  }//iv
  return true;
}


bool ReconRandomiser::Finalise(){

  delete fRand;

  return true;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// The following are randomising functions
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void ReconRandomiser::CreateVertex(double * pos)
{
  double x, y, r, z, xdir, ydir;

  //create a flat distribution in r and phi
  if(fFlatR && fUniformX && fUniformY) {
    r = fRand->Uniform(0, +fMaxRPos);
    fRand->Circle(xdir, ydir, 1);
    x = r * xdir;
    y = r * ydir;
  }
  //create a flat distribution in x and y
  else {
    r = fMaxRPos + 1;
    while(r > fMaxRPos) {
      //x
      if(fUniformX)
	x = fRand->Uniform(-fMaxRPos, +fMaxRPos);
      else
	x = fRand->Gaus(fXMean, fXWidth);
      //y
      if(fUniformY)
	y = fRand->Uniform(-fMaxRPos, +fMaxRPos);
      else
	y = fRand->Gaus(fXMean, fXWidth);
      //r
      r = TMath::Sqrt(TMath::Power(x, 2) + TMath::Power(y, 2));
    }
  }
  //z
  if(fUniformZ)
    z = fRand->Uniform(-fMaxZPos, +fMaxZPos);
  else
    z = fRand->Gaus(fZMean, fZWidth);

  pos[0] = x;
  pos[1] = y;
  pos[2] = z;
}

///////////////////////////////////////////////////////////////////////

