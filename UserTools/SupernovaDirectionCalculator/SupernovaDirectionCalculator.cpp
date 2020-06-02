#include "SupernovaDirectionCalculator.h"

SupernovaDirectionCalculator::SupernovaDirectionCalculator():Tool(){}


bool SupernovaDirectionCalculator::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("SupernovaDirectionCalculator") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;


  if(!m_variables.Get("input_filter_name", m_input_filter_name)) {
    Log("INFO: input_filter_name not given. Using ALL", WARN, m_verbose);
    m_input_filter_name = "ALL";
  }
  m_in_filter  = m_data->GetFilter(m_input_filter_name, false);
  if(!m_in_filter) {
    m_ss << "FATAL: no filter named " << m_input_filter_name << " found. Returning false";
    StreamToLog(FATAL);
    return false;
  }

  // Apply weights?
  m_weight_events = false;
  m_variables.Get("weight_events", m_weight_events);
  // Load weights for interpolation
  if (m_weight_events){
    std::string weights_file, line;
    m_variables.Get("weights_file", weights_file);
    ifstream myfile(weights_file);
    if (myfile.is_open()){
      // Skip first line
      getline(myfile,line);
      double logE, weight;
      char _; // Dump variable to ignore commas
      while (myfile >> logE >> _ >> weight){
        m_log10_energy.push_back(logE);
        m_weight.push_back(weight);
      }
      myfile.close();
    } else {
      Log("Could not open weights file!", ERROR, m_verbose);
      return false;
    }
  }


  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool SupernovaDirectionCalculator::Execute(){

  if(m_stopwatch) m_stopwatch->Start();

  const int N = m_in_filter->GetNRecons();

  m_direction[0] = m_direction[1] = m_direction[2] = 0;

  for(int irecon = 0; irecon < N; irecon++) {
    //get the vertex position
    DirectionEuler dir = m_in_filter->GetDirectionEuler(irecon);
    double dir_z = cos(dir.theta);
    double dir_y = sin(dir.theta) * sin(dir.phi);
    double dir_x = sin(dir.theta) * cos(dir.phi);

    double weight = 1.;
    if (m_weight_events){
      double E = m_in_filter->GetEnergy(irecon);
      weight = GetEventWeight(log10(E));
    }

    m_direction[0] += dir_x * weight;
    m_direction[1] += dir_y * weight;
    m_direction[2] += dir_z * weight;

  }//irecon

  // Normalise direction vector
  double r = m_direction[0]*m_direction[0];
  r +=  m_direction[1]*m_direction[1];
  r +=  m_direction[2]*m_direction[2];
  r = sqrt(r);
  m_direction[0] /= r;
  m_direction[1] /= r;
  m_direction[2] /= r;

  m_ss << "Reconstructed SN neutrino direction from " << N << " events x, y, z: " << m_direction[0] << ", " << m_direction[1] << ", " << m_direction[2] << std::endl;
  StreamToLog(INFO);

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool SupernovaDirectionCalculator::Finalise(){

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }


  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}

double SupernovaDirectionCalculator::GetEventWeight(double log10_energy){
  double weight = 0;
  int i = -1;
  // Find interpolation position
  for(int j=0; j<m_log10_energy.size(); ++j){
    if (m_log10_energy[j] > log10_energy) break;
    i = j;
  }

  // Out of range
  if (i < 0) return m_weight[0];
  if (i == m_log10_energy.size() - 1) return m_weight[i];

  // Interpolate
  double x0 = m_log10_energy[i];
  double x1 = m_log10_energy[i+1];
  double x = log10_energy;
  double y0 = m_weight[i];
  double y1 = m_weight[i+1];
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}
