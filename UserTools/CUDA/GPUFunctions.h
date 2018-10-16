#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include <vector>

namespace GPU_daq {

int test_vertices_initialize();
int test_vertices_execute();
int test_vertices_finalize();
int nhits_initialize();
int nhits_execute();
int nhits_finalize();

//int CUDAFunction(std::vector<int> PMTid, std::vector<int> time);

};

#endif
