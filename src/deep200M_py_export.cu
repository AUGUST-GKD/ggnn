#include "pybind11/pybind11.h"

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <cub/cub.cuh>

// only needed for file_exists check
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

#include <iostream>
#include <vector>

#include "ggnn/cuda_knn_ggnn_multi_gpu.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"

template <DistanceMeasure measure,
          typename KeyT, typename ValueT, typename GAddrT, typename BaseT,
          typename BAddrT, int D, int KBuild, int KF, int KQuery, int S>
class GGNNWrapper{
  using GGNN_t=GGNNMultiGPU<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S>;
 public:
  GGNNWrapper(const int L, const float tau_build) {
    instance.reset(new GGNN_t(L, tau_build));
  }

  void configure(const std::vector<int>& gpus,
                 const int N_shard, const std::string& graph_dir,
                 const int refinement_iterations){
    this->gpus=gpus;
    this->graph_dir=graph_dir;
    this->N_shard=N_shard;
    this->refinement_iterations=refinement_iterations;
  }

  void build_index(const std::string& basePath,
                   const size_t N_base = std::numeric_limits<size_t>::max()) {
    instance->loadBase(basePath, N_base, Dataset::ds_filetype::XBIN);
    instance->configure(gpus, true, N_shard, graph_dir);
    instance->build();
  }

  void save_index(){
    instance->store();
  }

  void load_index(){
    instance->load();
  }

  std::pair<std::vector<KeyT>, std::vector<ValueT>> search_index(
      const std::vector<BaseT>& query_v, const KeyT N_query) {
    instance->importQuery(query_v, N_query);
    std::pair<std::vector<KeyT>, std::vector<ValueT>> res =
        instance->query(0.5f);
    return ret;
  }

 private:
  std::unique_ptr<GGNN_t> instance;
  // parameters
  std::vector<int> gpus;
  std::string graph_dir;
  int N_shard, refinement_iterations;
};

