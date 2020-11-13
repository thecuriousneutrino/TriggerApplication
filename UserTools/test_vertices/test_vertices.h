#ifndef test_vertices_H
#define test_vertices_H

#include <string>
#include <iostream>

#include "Tool.h"

#include "GPUFunctions.h"
#include "Stopwatch.h"

class test_vertices: public Tool {


 public:

  test_vertices();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  float m_distance_between_vertices;
  float m_wall_like_distance;
  float m_water_like_threshold_number_of_pmts;
  float m_wall_like_threshold_number_of_pmts;
  float m_coalesce_time;
  float m_trigger_gate_up;
  float m_trigger_gate_down;
  int m_output_txt;
  int m_correct_mode;
  int m_n_direction_bins_theta;
  bool m_cylindrical_grid;
  float m_costheta_cone_cut;
  bool m_select_based_on_cone;
  bool  m_trigger_threshold_adjust_for_noise;
  int m_max_n_hits_per_job;
  int m_num_blocks_y;
  int m_num_threads_per_block_y;
  int m_num_threads_per_block_x;
  int m_write_output_mode;

  /// CPU version of the algorithm
  typedef unsigned short offset_t;
  typedef unsigned int histogram_t;
  typedef unsigned int time_of_flight_t;
  int CPU_test_vertices_initialize();
  int CPU_test_vertices_finalize();
  int CPU_test_vertices_execute(std::vector<int> PMTid, std::vector<int> time, std::vector<int> * trigger_ns, std::vector<int> * trigger_ts);

  void make_test_vertices();
  void make_table_of_tofs();
  void make_table_of_directions();
  unsigned int get_distance_index(unsigned int pmt_id, unsigned int vertex_block);
  unsigned int get_time_index(unsigned int hit_index, unsigned int vertex_block);
  unsigned int get_direction_index_at_pmt(unsigned int pmt_id, unsigned int vertex_index, unsigned int direction_index);
  unsigned int get_direction_index_at_angles(unsigned int iphi, unsigned int itheta);
  unsigned int get_direction_index_at_time(unsigned int time_bin, unsigned int vertex_index, unsigned int direction_index);
  void free_global_memories();
  bool read_the_input_ToolDAQ(std::vector<int> PMTid, std::vector<int> time, int * earliest_time);
  void write_output();
  void allocate_candidates_memory_on_host();
  void correct_times_and_get_histo_per_vertex_shared(unsigned int *ct);
  void find_vertex_with_max_npmts_in_timebin(histogram_t * np, histogram_t * mnp, unsigned int * vmnp);
  void choose_candidates_above_threshold();
  void coalesce_triggers();
  void separate_triggers_into_gates(std::vector<int> * trigger_ns, std::vector<int> * trigger_ts);
  void free_event_memories();
  
  /// CPU parameters
  double distance_between_vertices; // linear distance between test vertices
  double wall_like_distance; // distance from wall (in units of distance_between_vertices) to define wall-like events
  unsigned int time_step_size; // time binning for the trigger
  unsigned int water_like_threshold_number_of_pmts; // number of pmts above which a trigger is possible for water-like events
  unsigned int wall_like_threshold_number_of_pmts; // number of pmts above which a trigger is possible for wall-like events
  unsigned int nhits_threshold_min, nhits_threshold_max;
  double coalesce_time; // time such that if two triggers are closer than this they are coalesced into a single trigger
  double trigger_gate_up; // duration to be saved after the trigger time
  double trigger_gate_down; // duration to be saved before the trigger time
  unsigned int max_n_hits_per_job; // max n of pmt hits per job
  double dark_rate;
  float costheta_cone_cut; // max distance between measured costheta and cerenkov costheta
  bool select_based_on_cone; // for mode == 10, set to 0 to select based on vertices, to 1 to select based on cone
  /// detector
  double detector_height; // detector height
  double detector_radius; // detector radius
  /// pmts
  unsigned int n_PMTs; // number of pmts in the detector
  double * PMT_x, *PMT_y, *PMT_z; // coordinates of the pmts in the detector
  /// vertices
  unsigned int n_test_vertices; // number of test vertices
  unsigned int n_water_like_test_vertices; // number of test vertices
  double * vertex_x, * vertex_y, * vertex_z; // coordinates of test vertices
  /// hits
  offset_t time_offset;  // ns, offset to make times positive
  unsigned int n_time_bins; // number of time bins 
  unsigned int n_direction_bins_theta; // number of direction bins 
  unsigned int n_direction_bins_phi; // number of direction bins 
  unsigned int n_direction_bins; // number of direction bins 
  unsigned int n_hits; // number of input hits from the detector
  unsigned int * host_ids; // pmt id of a hit
  unsigned int * host_times;  // time of a hit
  // corrected tim bin of each hit (for each vertex)
  unsigned int * host_time_bin_of_hit;
  // npmts per time bin
  unsigned int * host_n_pmts_per_time_bin;
  unsigned int * host_n_pmts_nhits;
  //unsigned int * host_time_nhits;
  // tof
  double speed_light_water;
  double cerenkov_angle_water;
  float cerenkov_costheta;
  double twopi;
  bool cylindrical_grid;
  time_of_flight_t *host_times_of_flight;
  float *host_light_dx;
  float *host_light_dy;
  float *host_light_dz;
  float *host_light_dr;
  bool *host_directions_for_vertex_and_pmt;
  // triggers
  std::vector<std::pair<unsigned int,unsigned int> > candidate_trigger_pair_vertex_time;  // pair = (v, t) = (a vertex, a time at the end of the 2nd of two coalesced bins)
  std::vector<unsigned int> candidate_trigger_npmts_in_time_bin; // npmts in time bin
  std::vector<unsigned int> candidate_trigger_npmts_in_cone_in_time_bin; 
  std::vector<std::pair<unsigned int,unsigned int> > trigger_pair_vertex_time;
  std::vector<unsigned int> trigger_npmts_in_time_bin;
  std::vector<unsigned int> trigger_npmts_in_cone_in_time_bin;
  std::vector<std::pair<unsigned int,unsigned int> > final_trigger_pair_vertex_time;
  std::vector<double> output_trigger_information;
  // make output txt file for plotting?
  bool output_txt;
  unsigned int correct_mode;
  unsigned int write_output_mode;
  // find candidates
  histogram_t * host_max_number_of_pmts_in_time_bin;
  unsigned int *  host_vertex_with_max_n_pmts;
  unsigned int * host_max_number_of_pmts_in_cone_in_time_bin;
  // gpu properties
  int max_n_threads_per_block;
  int max_n_blocks;
  // verbosity level
  bool use_verbose;
  bool use_timing;
  // files
  std::string event_file;
  std::string detector_file;
  std::string pmts_file;
  std::string output_file;
  std::string event_file_base;
  std::string event_file_suffix;
  std::string output_file_base;
  float elapsed_parameters, elapsed_pmts, elapsed_detector, elapsed_vertices,
    elapsed_threads, elapsed_tof, elapsed_directions, elapsed_memory_tofs_dev, elapsed_memory_directions_dev, elapsed_memory_candidates_host, elapsed_tofs_copy_dev,  elapsed_directions_copy_dev,
    elapsed_input, elapsed_memory_dev, elapsed_copy_dev, elapsed_kernel_correct_times_and_get_n_pmts_per_time_bin, 
    elapsed_threads_candidates, elapsed_candidates_memory_dev, elapsed_candidates_kernel,
    elapsed_candidates_copy_host, choose_candidates, elapsed_coalesce, elapsed_gates, elapsed_free, elapsed_total,
    elapsed_tofs_free, elapsed_reset, elapsed_write_output;
  unsigned int greatest_divisor;
  unsigned int the_max_time;
  unsigned int nhits_window;
  int n_events;


#ifdef GPU   
  /// integer times to run over GPU card
  std::vector<int> m_time_int;
#else
  std::vector<int> m_time_int;
#endif

  /// The stopwatch, if we're using one
  util::Stopwatch * m_stopwatch;
  /// Image filename to save the histogram to, if required
  std::string m_stopwatch_file;

  /// Verbosity level, as defined in tool parameter file
  int m_verbose;

  /// For easy formatting of Log messages
  std::stringstream m_ss;

  /// Print the current value of the streamer at the set log level,
  ///  then clear the streamer
  void StreamToLog(int level) {
    Log(m_ss.str(), level, m_verbose);
    m_ss.str("");
  }

  /// Log level enumerations
  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};

};


#endif
