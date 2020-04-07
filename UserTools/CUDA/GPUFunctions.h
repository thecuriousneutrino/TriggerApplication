#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include <vector>

namespace GPU_daq {

int test_vertices_initialize();
int test_vertices_execute();
int test_vertices_finalize();
int nhits_initialize();
int nhits_initialize_ToolDAQ(std::string PMTFile, std::string DetectorFile, std::string ParameterFile);
int nhits_execute();
int nhits_execute(std::vector<int> PMTid, std::vector<int> time);
int nhits_finalize();

//int CUDAFunction(std::vector<int> PMTid, std::vector<int> time);

}

#endif
