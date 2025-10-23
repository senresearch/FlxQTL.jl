using Distributed
addprocs(4)
@everywhere using FlxQTL, Random,Test, LinearAlgebra, Statistics

FlxQTL.Util.setSeed(1023);
include("scan1test.jl")
include("scan4cross.jl")
include("scantest1.jl")
