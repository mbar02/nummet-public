# Nota: Ottimizza i vicini nel modo sensato per lettura
# sequenziale della memoria:
#   1 (i,j-1)
#   2 (i,j+1)
#   3 (i-1,j)
#   4 (i+1,j)

const L::Int32       = UInt32(100)
const SAMPLES::Int32 = UInt32(2000000)
const UPS::Int32     = UInt32(1)

using ProgressBars
using Printf



@inline function nextIdx(idx)
  idx += 1
  idx > L ? idx - L : idx
end

@inline function prevIdx(idx)
  idx -= 1
  idx < 1 ? idx + L : idx
end

mutable struct Cell
  value::Int8
  neighs::NTuple{4,UInt16}
end

@inline function energy(lattice)
  @inbounds begin
  energy = Int32(0)
  for i in 0:(L-1), j in 1:L
    energy -= lattice[i*L+j].value *
            lattice[lattice[i*L+j].neighs[1]].value +
            + lattice[i*L+j].value *
            lattice[lattice[i*L+j].neighs[3]].value
  end
  return energy
end
end

function buildCluster(
  lattice,
  cluster,
  magnets,
  p
)
@inbounds begin
  nOld::UInt16 = 0
  nNew::UInt16 = 1
  oldState::Int8 = 0

  cluster[1] = rand(1:(L*L))

  oldState = lattice[cluster[1]].value
  lattice[cluster[1]].value = -oldState

  while(nOld < nNew)
    nOld += 1
    for i=1:4
      if (
        lattice[lattice[cluster[nOld]].neighs[i]].value ==
          oldState && rand() <= p
      )
        nNew += 1
        cluster[nNew] = lattice[cluster[nOld]].neighs[i]
        lattice[cluster[nNew]].value = -oldState
      end
    end
  end
  return magnets - 2nNew * oldState
end
end

function singleClusterUpdate(beta::Float32)
  lattice = Vector{Cell}(undef, L*L)
  cluster = Vector{UInt16}(undef, L*L)
  for i=0:(L-1), j=1:L
    lattice[i*L+j] = Cell(
      1,
      (
        i*L+prevIdx(j),
        i*L+nextIdx(j),
        (prevIdx(i+1)-1)*L+j,
        (nextIdx(i+1)-1)*L+j,
      )
    )
  end

  p = 1.0 - exp(-2.0beta)

  magns = Vector{Int32}(undef,SAMPLES)
  eners = Vector{Int32}(undef,SAMPLES)

  magns[1] = buildCluster(lattice, cluster, L*L, p)
  eners[1] = energy(lattice)

  for i in ProgressBar(2:SAMPLES)
#    for k1=0:(L-1)
#      for k2=1:L
#        l = lattice[k1*L+k2].value
#        if l == +1
#          print("\033[1m██\033[0m")
#        elseif l == -1
#          print("\033[2m██\033[0m")
#        else
#          @printf("%+02d ", l)
#        end
#      end
#      println()
#    end
#    println()
    magns[i] = buildCluster(lattice, cluster, magns[i-1], p)
    eners[i] = energy(lattice)
  end
end

singleClusterUpdate(Float32(0.8))
