#pragma once

#if __cplusplus
extern "C" {
#endif

#ifndef __METAL_VERSION__

#include<inttypes.h>
#define DEVICE_QM 

#else

#define DEVICE_QM device

#endif

#ifdef USE_CELL_INDICES

#define NEIGH_T int32_t
#define NEIGH_Z
#define NEIGH_V(lattice,site,idx) lattice[(site).neighbors[idx]].value

#else

#define NEIGH_T DEVICE_QM struct cell*
#define NEIGH_Z (lattice)
#define NEIGH_V(lattice,site,idx) (site).neighbors[idx]->value

#endif

#define nextIdx(i,L) ( (i) <  (L)-1 ? (i)+1 : (i)+1-(L) )
#define prevIdx(i,L) ( (i) >      0 ? (i)-1 : (i)+(L)-1 )

#define LATTICE_SQR 1
#define LATTICE_TRI 2
#define LATTICE_HEX 3

#ifndef LATTICE
#define LATTICE LATTICE_SQR
#endif

#if   LATTICE==LATTICE_SQR

#define N_NEIGH 4
#define initCell(lattice,i,j,L) {                                          \
  (lattice)[(i)*(L)+(j)].value = 1;                                        \
  (lattice)[(i)*(L)+(j)].neighbors[0] = NEIGH_Z + (i)*(L)+nextIdx(j, L) ;  \
  (lattice)[(i)*(L)+(j)].neighbors[1] = NEIGH_Z + (i)*(L)+prevIdx(j, L) ;  \
  (lattice)[(i)*(L)+(j)].neighbors[2] = NEIGH_Z + nextIdx(i, L)*(L)+(j) ;  \
  (lattice)[(i)*(L)+(j)].neighbors[3] = NEIGH_Z + prevIdx(i, L)*(L)+(j) ;  \
}
#ifndef __METAL_VERSION__
#define VOL_FROM_L(L) (L)*(L)
#endif

#elif LATTICE==LATTICE_TRI

#define N_NEIGH 6
#define initCell(lattice,i,j,L) {                                            \
  (lattice)[(i)*(L)+(j)].value = 1;                                          \
  (lattice)[(i)*(L)+(j)].neighbors[0] = NEIGH_Z + (i)*(L)+nextIdx(j, L);     \
  (lattice)[(i)*(L)+(j)].neighbors[1] = NEIGH_Z + (i)*(L)+prevIdx(j, L);     \
  (lattice)[(i)*(L)+(j)].neighbors[2] = NEIGH_Z + nextIdx(i, L)*(L)+(j);     \
  (lattice)[(i)*(L)+(j)].neighbors[3] = NEIGH_Z + prevIdx(i, L)*(L)+(j);     \
  (lattice)[(i)*(L)+(j)].neighbors[4] = NEIGH_Z + nextIdx(j, L)              \
                                                        + nextIdx(i, L)*(L); \
  (lattice)[(i)*(L)+(j)].neighbors[5] = NEIGH_Z + prevIdx(j, L)              \
                                                + prevIdx(i, L)*(L);         \
}
#ifndef __METAL_VERSION__
#define VOL_FROM_L(L) (L)*(L)
#endif

#elif LATTICE==LATTICE_HEX

#define N_NEIGH 3
#define initCell(lattice,i,j,L) {                                                                    \
  (lattice)[2*((i)*(L)+(j))  ].value = 1;                                                            \
  (lattice)[2*((i)*(L)+(j))+1].value = 1;                                                            \
  (lattice)[2*((i)*(L)+(j))  ].neighbors[0] = NEIGH_Z + 2*( nextIdx(i,L)*(L) + prevIdx(j,L) ) + 1;   \
  (lattice)[2*((i)*(L)+(j))  ].neighbors[1] = NEIGH_Z + 2*( prevIdx(i,L)*(L) +          (j) ) + 1;   \
  (lattice)[2*((i)*(L)+(j))  ].neighbors[2] = NEIGH_Z + 2*(          (i)*(L) +          (j) ) + 1;   \
  (lattice)[2*((i)*(L)+(j))+1].neighbors[0] = NEIGH_Z + 2*(          (i)*(L) +          (j) );       \
  (lattice)[2*((i)*(L)+(j))+1].neighbors[1] = NEIGH_Z + 2*( nextIdx(i,L)*(L) +          (j) );       \
  (lattice)[2*((i)*(L)+(j))+1].neighbors[2] = NEIGH_Z + 2*( prevIdx(i,L)*(L) + nextIdx(j,L) );       \
}
#ifndef __METAL_VERSION__
#define VOL_FROM_L(L) 2*(L)*(L)
#endif

#else

#error "Unknown lattice proposed. Only LATTICE_SQR=1, LATTICE_TRI=2 and LATTICE_HEX=3 are available."

#endif

struct cell {
  int8_t value;
  NEIGH_T neighbors[N_NEIGH];
};

typedef struct cell cell_t;

#ifndef __METAL_VERSION__
inline int64_t energy(cell_t* lattice, uint32_t volume) {
  int64_t energy = 0;
  for(uint32_t i=0; i<volume; i++)
    for(uint32_t k=0; k<N_NEIGH; k++)
      energy -= lattice[i].value * NEIGH_V(lattice,lattice[i],k);
  return energy/2;
}
#endif

#if __cplusplus
}
#endif
