#pragma once

#define MSL_THSPTHG    64
#define MSL_THSPTHGSQR 8

#ifndef __METAL_VERSION__

#include <cstddef>
#include <cstdint>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#pragma clang diagnostic pop

#define USE_CELL_INDICES
#define SAMPLES_PER_COMMIT 10000

#include "../libs/geometry.h"
#include "../libs/pcg/pcg_basic.h"

class MetalIsingMetropolis {
  public:
    MetalIsingMetropolis(
      const char*    exePath,
      uint32_t       L,
      float          beta,
      pcg32_random_t *rng
    );
    ~MetalIsingMetropolis();

    void simulation(
      uint32_t      UPS, // updates per sample
      uint32_t      SAMPLES,
      std::ofstream &ofile
    );

  private:
    uint32_t L;
    size_t   volume;

    MTL::Device *_device;
    MTL::CommandQueue *_cmdQueue;
    MTL::ComputePipelineState *_initializePipeline;
    MTL::ComputePipelineState *_updatePipeline;
    MTL::ComputePipelineState *_redsumPipeline;
    MTL::ComputePipelineState *_energMagnPipeline;

    MTL::Buffer *prob;

    MTL::Buffer *lattice;
    MTL::Buffer *rng;

    uint32_t redSumSteps;
    MTL::Buffer **redSumsM;
    MTL::Buffer **redSumsE;
    size_t *noOfElements;

    MTL::Buffer *totMagnet;
    int64_t *totMagnetCPU;
    MTL::Buffer *totEnergy;
    int64_t *totEnergyCPU;
};

#endif