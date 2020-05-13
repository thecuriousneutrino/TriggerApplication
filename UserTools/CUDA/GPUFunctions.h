#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include <vector>

namespace GPU_daq {

int test_vertices_initialize();
 int test_vertices_initialize_ToolDAQ(double detector_length, double detector_radius, double pmt_radius, std::string ParameterFile, std::vector<int> tube_no,std::vector<float> tube_x,std::vector<float> tube_y,std::vector<float> tube_z, float f_dark_rate,
  float distance_between_vertices,
  float wall_like_distance,
  float water_like_threshold_number_of_pmts,
  float wall_like_threshold_number_of_pmts,
  float coalesce_time,
  float trigger_gate_up,
  float trigger_gate_down,
  int output_txt,
  int correct_mode,
  int n_direction_bins_theta,
  bool cylindrical_grid,
  float costheta_cone_cut,
  bool select_based_on_cone,
  bool  trigger_threshold_adjust_for_noise,
  int f_max_n_hits_per_job,
  int f_num_blocks_y,
  int f_num_threads_per_block_y,
  int f_num_threads_per_block_x,
  int f_write_output_mode
);
int test_vertices_execute();
int test_vertices_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts);
int test_vertices_finalize();
int nhits_initialize();
 int nhits_initialize_ToolDAQ(std::string ParameterFile, int nPMTs, int fTriggerSearchWindow, int fTriggerSearchWindowStep, int fTriggerThreshold, int fTriggerSaveWindowPre, int fTriggerSaveWindowPost);
int nhits_execute();
int nhits_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts);
int nhits_finalize();

//int CUDAFunction(std::vector<int> PMTid, std::vector<int> time);

}

#endif
