#include "./MetalIsingMetropolis.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <semaphore>
#include "../libs/errors.h"
#include "../libs/progressBar.h"

#define METAL_LIB_EXTENSION ".metallib"

inline int loadKernel(const char *name, MTL::Library* library, MTL::ComputePipelineState* &pipeline, MTL::Device* device, NS::Error** error) {
  auto nsname= NS::String::string(name, NS::ASCIIStringEncoding);
  MTL::Function *function = library->newFunction(nsname);
  if (function == nullptr)
  {
    std::cout << "Failed to find " << name << "." << std::endl;
    return TEINVAL;
  }
  pipeline = device->newComputePipelineState(function, error);
  if (pipeline == nullptr)
  {
    std::cout << "Failed to create " << name << " pipeline state object." << std::endl;
    return TEINVAL;
  }
  return 0;
}

MetalIsingMetropolis::MetalIsingMetropolis(
  const char     *exePath,
  uint32_t       L,
  float          beta,
  pcg32_random_t *rng
) {
  this->L = L;
  this->volume = VOL_FROM_L(L);

  // Get the device
  this->_device = MTL::CreateSystemDefaultDevice();
  NS::Error *error = nullptr;

  std::string libPath = std::string(exePath) + METAL_LIB_EXTENSION;

  // Load Metal library
  auto filepath = NS::String::string((const char*)libPath.c_str(), NS::ASCIIStringEncoding);
  MTL::Library *opLibrary = _device->newLibrary(filepath, &error);

  if(opLibrary == nullptr) {
    std::cerr << "Failed to find the default library. Error: "
              << error->description()->utf8String() << std::endl;
    return;
  }

  // Load Metal kernels
  assert(loadKernel("initialize", opLibrary, this->_initializePipeline, this->_device, &error) == 0);
  assert(loadKernel("update", opLibrary, this->_updatePipeline, this->_device, &error) == 0);
  assert(loadKernel("redsum", opLibrary, this->_redsumPipeline, this->_device, &error) == 0);
  assert(loadKernel("energy_magnetization", opLibrary, this->_energMagnPipeline, this->_device, &error) == 0);

  // Create a Command Queue
  this->_cmdQueue = this->_device->newCommandQueue();
  if (this->_cmdQueue == nullptr)
  {
      std::cerr << "Failed to find the command queue." << std::endl;
      return;
  }

  // Allocate needed buffers
  this->prob     = _device->newBuffer(
    (N_NEIGH+1)*sizeof(uint32_t),
    MTL::ResourceStorageModeShared
  );
  uint32_t* probs_CPU = (uint32_t*)this->prob->contents();

  this->lattice   = _device->newBuffer(
    volume*sizeof(int8_t),
    MTL::ResourceStorageModePrivate
  );
  this->rng       = _device->newBuffer(
    volume*sizeof(pcg32_random_t),
    MTL::ResourceStorageModePrivate
  );
  
  // Buffers for ricorsive reduced summations
  uint32_t sumSteps = 0;
  size_t   samples  = volume, firstSamples;
    // let the cycle run at least one time, in order to move data to the end buffers
  // First reduction has 2D grid! Different factor
  samples = ( L + MSL_THSPTHGSQR - 1 ) / MSL_THSPTHGSQR *
#if LATTICE==LATTICE_HEX
           ( ( 2*L + MSL_THSPTHG/MSL_THSPTHGSQR - 1 ) / (MSL_THSPTHG/MSL_THSPTHGSQR) );
#else
           ( (   L + MSL_THSPTHG/MSL_THSPTHGSQR - 1 ) / (MSL_THSPTHG/MSL_THSPTHGSQR) );
#endif
  firstSamples = samples;
  sumSteps++;
  while( samples > 1 ) {
    samples = ( samples + MSL_THSPTHG - 1 ) / MSL_THSPTHG;
    sumSteps++;
  }

  this->redSumSteps = sumSteps;
  if( sumSteps > 1 ) {
  this->redSumsE = new MTL::Buffer*[ sumSteps-1 ];
  this->redSumsM = new MTL::Buffer*[ sumSteps-1 ];
  }
  this->noOfElements = new size_t[ sumSteps ];
  samples = firstSamples;
  if( sumSteps > 1 ) {
    this->redSumsM[0] = this->_device->newBuffer( samples*sizeof(int64_t), MTL::ResourceStorageModePrivate );
    this->redSumsE[0] = this->_device->newBuffer( samples*sizeof(int64_t), MTL::ResourceStorageModePrivate );
    this->noOfElements[0] = samples; 
  }
  for(uint32_t i = 1; i < sumSteps-1; i++) {
    samples = ( samples + MSL_THSPTHG - 1 ) / MSL_THSPTHG;
    this->redSumsM[i] = this->_device->newBuffer( samples*sizeof(int64_t), MTL::ResourceStorageModePrivate );
    this->redSumsE[i] = this->_device->newBuffer( samples*sizeof(int64_t), MTL::ResourceStorageModePrivate );
    this->noOfElements[i] = samples;
  }
  this->noOfElements[sumSteps-1] = 1;
  // *2 due to offset_per_semaphore
  this->totMagnet = this->_device->newBuffer( sizeof(int64_t)*SAMPLES_PER_COMMIT*2, MTL::ResourceStorageModeShared );
  this->totMagnetCPU = (int64_t*)this->totMagnet->contents();
  this->totEnergy = this->_device->newBuffer( sizeof(int64_t)*SAMPLES_PER_COMMIT*2, MTL::ResourceStorageModeShared );
  this->totEnergyCPU = (int64_t*)this->totEnergy->contents();

  // Init spin flip probabilities
  for(int32_t i=0; i<N_NEIGH+1; i++){
    double p_f = 1. / ( 1. + exp( -2. * beta * ( N_NEIGH - 2*i ) ) );
    probs_CPU[i] = (uint64_t)(0x000100000000*p_f) & 0xffffffff;
  }

  // Init lattice
  MTL::CommandBuffer         *cmdBuff = this -> _cmdQueue -> commandBuffer();
  MTL::ComputeCommandEncoder *cptEnc  = cmdBuff -> computeCommandEncoder();

  NS::UInteger nThreads = this -> _initializePipeline -> threadExecutionWidth();
  NS::UInteger tgSize   = this -> _initializePipeline -> maxTotalThreadsPerThreadgroup() / nThreads;

  // Create random
  MTL::Buffer* rndBuff = this->_device->newBuffer(
#if LATTICE==LATTICE_HEX
    volume*4*sizeof(uint64_t),
#else
    volume*2*sizeof(uint64_t),
#endif
    MTL::ResourceStorageModeShared
  );
  uint64_t* rndBuff_CPU = (uint64_t*)rndBuff->contents();
  for(size_t i = 0; i < 2*volume; i++) {
    uint64_t random_number = 0;
    // Ensure 0x10 < random_number < 0xfffffffffffffff0
    while( random_number <= 0x10 || random_number >= 0xfffffffffffffff0 )
      random_number = ( (uint64_t)pcg32_random_r(rng) << 32 ) + pcg32_random_r(rng);
    rndBuff_CPU[i] = random_number;
  }

  // Load init parameters
  cptEnc -> setComputePipelineState(this->_initializePipeline);
  cptEnc -> setBuffer(this->lattice, 0,                0);
  cptEnc -> setBytes (&(this->L),    sizeof(uint32_t), 1);
  cptEnc -> setBuffer(this->rng,     0,                2);
  cptEnc -> setBuffer(rndBuff,       0,                3);

  // Prepare grid
  MTL::Size gridSize = MTL::Size::Make(L,L,1);
  MTL::Size thrGSize = MTL::Size::Make(nThreads,tgSize,1);

  cptEnc -> dispatchThreads(
    gridSize,
    thrGSize
  );

  // Commit to GPU
  cptEnc  -> endEncoding();
  cmdBuff -> commit();

  cmdBuff -> waitUntilCompleted();
  cptEnc  -> release();
  rndBuff -> release();
  cmdBuff -> release();
}

MetalIsingMetropolis::~MetalIsingMetropolis() {
  this->_initializePipeline->release();
  this->_updatePipeline->release();
  this->_redsumPipeline->release();
  this->_energMagnPipeline->release();
  this->prob->release();
  this->lattice->release();
  this->rng->release();
  for(uint32_t i=0;i<this->redSumSteps-1;i++) {
    this->redSumsE[i]->release();
    this->redSumsM[i]->release();
  }
  delete[] this->redSumsE;
  delete[] this->redSumsM;
  delete[] this->noOfElements;
  this->totMagnet->release();
  this->totEnergy->release();
  this->_cmdQueue->release();
  this->_device->release();
}

void MetalIsingMetropolis::simulation(
  uint32_t      UPS, // updates per sample
  uint32_t      SAMPLES,
  std::ofstream &ofile
) {
  uint32_t binary[2] = {0,1};

  // In order to sync simulation and taking measures
  
  NS::UInteger width    =
#if LATTICE==LATTICE_HEX
                          L;
#else
                          L/2;
#endif
  size_t stdWidth = (size_t) width;

  // Threads grid, threadgroups grid and size per threadgroup, respectively
  // 1D lattice
//  const MTL::Size  gridThSize1D = MTL::Size::Make(volume,             1, 1);
//  const MTL::Size  gridTgSize1D = MTL::Size::Make(volume/MSL_THSPTHG, 1, 1);
  const MTL::Size  thrGrpSize1D = MTL::Size::Make(MSL_THSPTHG,        1, 1);

  // threadgroup size
  const size_t thgW = MSL_THSPTHG/MSL_THSPTHGSQR;
  const size_t thgH = MSL_THSPTHGSQR;

  // threadgroup grid size
  const size_t thGrH = (L          + thgH - 1)/thgH;
  const size_t thGrW = (2*stdWidth + thgW - 1)/thgW;
//  const size_t hthGrW = (L/2      + thgW - 1)/thgW;
//  const size_t hthGrH = (stdWidth + thgH - 1)/thgH;

  // 2D lattice
//  const MTL::Size  gridThSize2D = MTL::Size::Make(L,     2*stdWidth, 1);
  const MTL::Size  gridTgSize2D = MTL::Size::Make(thGrH, thGrW,      1);
  const MTL::Size  thrGrpSize2D = MTL::Size::Make(thgH,  thgW,       1);

  // 2D lattice, halfed
  const MTL::Size hgridThSize2D = MTL::Size::Make(L/2,    stdWidth, 1);
//  const MTL::Size hgridTgSize2D = MTL::Size::Make(hthGrH, hthGrW,   1);
    // thrGrpSize2D is not influenced by halfing the lattice

  MTL::Size gridSize;

  uint32_t leftSamples = SAMPLES;

  MTL::CommandBuffer* last_cmdBuff;

  //MTL::Fence* fence = this->_device->newFence();

  asyncProgressBar* pb = new asyncProgressBar(leftSamples, 80);

  std::counting_semaphore<2> semaphore{2};
  size_t offset_for_semaphore = 0;

  while(leftSamples > 0) {
  semaphore.acquire();

  // Get CommandBuffer and ComputeCommandEncoder
  MTL::CommandBuffer* cmdBuff = this->_cmdQueue->commandBuffer();
  MTL::ComputeCommandEncoder* cptEnc = cmdBuff->computeCommandEncoder();

  uint32_t samples = SAMPLES_PER_COMMIT;

  if(samples * UPS > 100000)
    samples = ( 100000 + UPS - 1 ) / UPS;

  if( leftSamples < samples )
    samples = leftSamples;

  leftSamples -= samples;

  for(uint32_t step = 0; step < samples; step++) {
    // 1. Update the lattice
    cptEnc->setComputePipelineState(this->_updatePipeline);
    cptEnc->setBuffer(this->lattice, 0, 0);
    cptEnc->setBuffer(this->rng,     0, 1);
    cptEnc->setBuffer(this->prob,    0, 2);
    for(uint32_t s = 0; s < UPS; s++) {
      for(uint32_t j=0;j<2;j++) for(uint32_t k=0;k<2;k++) {
        cptEnc->setBytes (&binary[j], sizeof(uint32_t), 3); // row    offset
        cptEnc->setBytes (&binary[k], sizeof(uint32_t), 4); // column offset
        cptEnc->dispatchThreads( hgridThSize2D, thrGrpSize2D );
      }
    }

    // 2.1. Compute magnetization and energy
    cptEnc->setComputePipelineState(this->_energMagnPipeline);
    cptEnc->setBuffer(this->lattice,  0, 0);
    if( this->redSumSteps == 1 ) {
      cptEnc->setBuffer(this->totEnergy, sizeof(int64_t)*(step+offset_for_semaphore), 1);
      cptEnc->setBuffer(this->totMagnet, sizeof(int64_t)*(step+offset_for_semaphore), 2);
    } else {
      cptEnc->setBuffer(this->redSumsE[0], 0, 1);
      cptEnc->setBuffer(this->redSumsM[0], 0, 2);
    }
    cptEnc->setBytes (&(this->L), sizeof(uint32_t), 3);
    // ! Achtung !  Not dispatchThread here!
    cptEnc->dispatchThreadgroups( gridTgSize2D, thrGrpSize2D );

    // 2.2. Summation over the lattice
    cptEnc->setComputePipelineState(this->_redsumPipeline);
    for(uint32_t i=0; i<(this->redSumSteps)-1; i++) {
      cptEnc->setBuffer(this->redSumsE[i  ],              0, 0);
      cptEnc->setBuffer(this->redSumsM[i  ],              0, 1);
      if( i == this->redSumSteps-2 ) {
        gridSize = MTL::Size::Make(1,1,1);
        cptEnc->setBuffer(this->totEnergy, sizeof(int64_t)*(step+offset_for_semaphore), 2);
        cptEnc->setBuffer(this->totMagnet, sizeof(int64_t)*(step+offset_for_semaphore), 3);
      } else {
        gridSize = MTL::Size::Make( this->noOfElements[i+1], 1, 1 );
        cptEnc->setBuffer(this->redSumsE[i+1], 0, 2);
        cptEnc->setBuffer(this->redSumsM[i+1], 0, 3);
      }
      cptEnc->setBytes (&(this->noOfElements[i]), sizeof(size_t), 4);

      // ! Warning !  Not dispatchThread here!
      cptEnc->dispatchThreadgroups( gridSize, thrGrpSize1D );
    }
  }
  cptEnc->endEncoding();
  cptEnc->release();

  // Data handler + Commit to GPU
  cmdBuff->addCompletedHandler([this, &ofile, samples, leftSamples, pb, &semaphore, offset_for_semaphore](MTL::CommandBuffer* buffer) {
    for(uint32_t i=0; i<samples; i++) {
      ofile << this->totMagnetCPU[offset_for_semaphore+i] << ", " << this->totEnergyCPU[i]/2 << std::endl;
    }
    semaphore.release();
    if( leftSamples != 0 )
      buffer->release();
    pb->update(samples);
  });
  
  cmdBuff->commit();

  last_cmdBuff = cmdBuff;
  offset_for_semaphore = SAMPLES_PER_COMMIT - offset_for_semaphore;
  }

  last_cmdBuff->waitUntilCompleted();
  last_cmdBuff->release();
}