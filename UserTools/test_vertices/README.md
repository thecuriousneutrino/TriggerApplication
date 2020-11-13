# test_vertices

test_vertices

## Data

Describe any data formats test_vertices creates, destroys, changes, analyzes, or its usage.




## Configuration

Describe any configuration variables for test_vertices.

```
## "typical" (not exact) distance between test vertices (cm)
 distance_between_vertices 500.

## can split vertices between wall-like and water-like
## will be wall-like if within a distance from the wall
 wall_like_distance 0

## trigger threshold (separately for water-vertices and wall-vertices)
 water_like_threshold_number_of_pmts 18
 wall_like_threshold_number_of_pmts 18

## triggers closer than this (ns) are merged
 coalesce_time 500.

## returned triggered window around trigger time
 trigger_gate_up 950.0
 trigger_gate_down -400.0

## GPU configuration
 max_n_hits_per_job 5000
 num_blocks_y 1
 num_threads_per_block_y 1024
 num_threads_per_block_x 1

## not used
 output_txt 0
 write_output_mode 1

## different algorithm options - please leave at 8, others are experimental
 correct_mode 8

## parameters for cone finder (not used by default)
 n_direction_bins_theta 3
 costheta_cone_cut 0.2
 select_based_on_cone 1

## are vertices arranged along circles or on cartesian grid?
## cylindrical grid works better
 cylindrical_grid 1

## increase threshold by mean noise occupancy
 trigger_threshold_adjust_for_noise 1

```
