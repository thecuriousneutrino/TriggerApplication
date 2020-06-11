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

  float direction[3] = {0,0,1};

  // First round
  // (Weighted) average over all directions
  CalculateDirection(direction, -1.);
  m_ss << "First pass: average over all events\n";
  m_ss << "First pass SN neutrino direction x, y, z: " << direction[0] << ", " << direction[1] << ", " << direction[2];
  StreamToLog(INFO);

  // Second round
  // (Weighted) average over all directions w/ cos(theta) w.r.t to previous estimate >= 0.0
  // Other events are used to estimate uniform contribution
  CalculateDirection(direction, 0.0);
  m_ss << "Second pass: average over events w/ cos(theta) >= 0.0\n";
  m_ss << "Second pass SN neutrino direction x, y, z: " << direction[0] << ", " << direction[1] << ", " << direction[2];
  StreamToLog(INFO);

  // Third round
  // (Weighted) average over all directions w/ cos(theta) w.r.t to previous estimate >= 0.5
  // Other events are used to estimate uniform contribution
  CalculateDirection(direction, 0.5);
  m_ss << "Third pass: average over events w/ cos(theta) >= 0.5\n";
  m_ss << "Third pass SN neutrino direction x, y, z: " << direction[0] << ", " << direction[1] << ", " << direction[2];
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

void SupernovaDirectionCalculator::CalculateDirection(float direction[3], float costheta_cut){
  const int N = m_in_filter->GetNRecons();
  double temp_direction[3] = {0,0,0};
  double flat_density = 0.;
  for(int irecon = 0; irecon < N; irecon++) {
    //get the event direction
    DirectionEuler dir = m_in_filter->GetDirectionEuler(irecon);
    double dir_z = cos(dir.theta);
    double dir_y = sin(dir.theta) * sin(dir.phi);
    double dir_x = sin(dir.theta) * cos(dir.phi);

    double weight = 1.;
    if (m_weight_events){
      double E = m_in_filter->GetEnergy(irecon);
      weight = GetEventWeight(log10(E));
    }

    double costheta = dir_x * direction[0] + dir_y * direction[1] + dir_z * direction[2];
    if (costheta < costheta_cut){
        // Add those events to the flat weight density estimtor
        flat_density += weight;
    } else {
        temp_direction[0] += dir_x * weight;
        temp_direction[1] += dir_y * weight;
        temp_direction[2] += dir_z * weight;
    }
  }//irecon

  // Calculate density
  if (flat_density > 0) {
    flat_density /= 2 * M_PI * (1.+costheta_cut);

    // Subtract expected flat contribution from vector
    // Expectation = flat_density * surface_of_sperical_cap * centre_of_mass_of_spherical_cap
    double h = (1.-costheta_cut);
    double expected = flat_density * (2.*M_PI*h) * (3*pow(2-h, 2)) / (4*(3-h));
    temp_direction[0] -= direction[0] * expected;
    temp_direction[1] -= direction[1] * expected;
    temp_direction[2] -= direction[2] * expected;
  }

  // Normalise direction vector
  double r = temp_direction[0]*temp_direction[0];
  r +=  temp_direction[1]*temp_direction[1];
  r +=  temp_direction[2]*temp_direction[2];
  r = sqrt(r);
  temp_direction[0] /= r;
  temp_direction[1] /= r;
  temp_direction[2] /= r;
  direction[0] = temp_direction[0];
  direction[1] = temp_direction[1];
  direction[2] = temp_direction[2];
}
