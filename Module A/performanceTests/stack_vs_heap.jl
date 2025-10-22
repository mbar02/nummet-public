using BenchmarkTools
using StaticArrays
using Printf

const N = 1000000

function mysum(vector, samples, reps)
  s = 0
  for i = 1:reps
    s -= s/i
    for j = 1:samples
      s += vector[j]
    end
  end
  return s
end

function (@main)(args)
  samples = ( length(args) >= 1 ) ? parse(Int, args[1]) : N;
  s = 0;
  r = ( length(args) >= 2 ) ? parse(Int, args[2]) : 1000;

  # Init Data
  heapArray  = rand(Int, samples)
  stackArray = MVector{samples,Int}(undef)
  for i in 1:samples
    stackArray[i] = heapArray[i]
  end

  @printf("Test with %d samples:\n", samples);

  # Heap Allocation
  println("Heap allocation")
  @time begin
    s = mysum(heapArray, samples, r);
  end
  println("result: ", s)

  # Stack Allocation
  println("Stack allocation")
  @time begin
    s = mysum(stackArray, samples, r);
  end
  println("result: ", s)
end
