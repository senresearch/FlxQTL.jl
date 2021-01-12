# QTL analysis

This section describes a step-by-step guide for QTL analysis.

## Input data file format

The package `FlxQTL` does not require any particular data format.  Any file readable in Julia is fine, but the input should contain traits (or phenotypes), genotype (probability), marker information on marker names, chromosomes, and marker positions, and optionally climatic information.  All inputs are types of 
Arrays in Julia and should have no missing values, i.e. imputation is required if missing values exist.

## Reading the data files and processing arrays

Use any Julia package able to read data files (`.txt`, `.csv`, etc.).  Julia's built-in module `DelimitedFiles` supports read, and write files. 
Let's try using an example dataset in `FlxQTL`. It is plant data: Arabidopsis thaliana in the `data` folder.  Detailed description on the data can be 
referred to `README` in the folder.

```julia
julia> using DelimitedFiles

julia> pheno = readdlm("data/Arabidopsis_fitness.csv",",";skipstart=1); # skip to read the first row (column names) to obtain a matrix only

julia> geno = readdlm("data/Arabidopsis_genotypes.csv",",";skipstart=1); 

julia> markerinfo = readdlm("data/Arabidopsis_markerinfo_1d.csv",',';skipstart=1);

```

For efficient computation, the normalization of matrices is necessary.  The phenotype matrix labelled as `pheno` here composes of wide range of values 
from 1.774 to 34.133, so that it is better to narow the range of values in [0,1], [-1,1], or any narrower interval for easy computation.  Note that 
the dimension of a phenotype matrix should be `the number of traits x the number of individuals`.

```julia
julia> Y=convert(Array{Float64,2},pheno'); #convert from transposed one to a Float64 matrix
julia> Ystd=(Y.-mean(Y,dims=2))./std(Y,dims=2); # sitewise normalization
```

In the genotype data, `1`, `2` indicate Italian, Swedish parents, respectively. You can rescale the genotypes for efficiency. 

```julia
julia> geno[geno.==1.0].=0.0;geno[geno.==2.0].=1.0; 

```
For genome scan, we need restructure the standardized genotype matrix combined with marker information.  Note that the genome scan in `FlxQTL` is 
implemented by CPU parallelization, so we need to add workers (or processes) before the genome scan.  Depending on the computer CPU, one can add as many 
processes as possible. If your computer has 16 cores, then you can add 15 or little more.  Note that you need to type `@everywhere` followed by `using PackageName` for parallel computing.  The dimension of a genotype (probability) matrix should be 
`the number of markers x the number of individuals`.

```julia
julia> addprocs(16) 
julia> @everywhere using FlxQTL 
julia> XX=FlxQTL.Markers(markerinfo[:,1],markerinfo[:,2],markerinfo[:,3],geno') # marker names, chromosomes, marker positions, genotypes

```
Optionally, one can generate a fixed (low-dimensional) trait covariate matrix (Z).  The first column indicates overall mean between the two regions, and 
the second implies site difference: `-1` for Italy, and `1` for Sweden.

```@repl
Z=hcat(ones(6),vcat(-ones(3),ones(3)))
m,q = size(Z) # check the dimension
```

## Computing a genetic (or climatic) relatedness matrix

The submodule `GRM` contains functions for computing kinship matrices, `kinshipMan`, `kinship4way`, `kinshipGs`, `kinshipLin`, `kinshipCtr`, and computing 
3D array of kinship matrices for LOCO (Leave One Chromosome Out) with (or without) a shrinkage method for nonpositive definiteness for the both, 
`shrinkg`, `shrinkgLoco`, `kinshipLoco`.  
Note that the shrinkage option is only for `kinshipMan`, `kinship4way`.

For the Arabidopsis genotype data, we will use a genetic relatedness matrix using manhattan distance measure, `kinshipMan` with a shrinkage and a 
LOCO option.

```julia
julia> Kg = FlxQTL.shrinkgLoco(FlxQTL.kinshipMan,50,XX)
```
For no LOCO option with shrinkage,

```julia
julia> K = FlxQTL.shrinkg(FlxQTL.kinshipMan,50,XX.X)
```


If you have climatic information on your trait data, you can compute the relatedness matrix using one of the above functions, but it is recommended using 
`kinshipGs`,`kinshipLin`,`kinshipCtr` after normalization.  Since the climatic information is not available, we use an identity matrix.

```@repl
using LinearAlgebra
Kc = Matrix(1.0I,6,6) # 3 years x 2 sites
```

## 1D genome scan

Once all input matrices are ready, we need to proceed the eigen-decomposition to two relatedness matrices. 
For a non-identity climatic relatedness, and a kinship with LOCO, you can do eigen-decomposition simultaneously.  Since we use the identity climatic 
relatedness, you can use `Matrix(1.0I,6,6)` for a matrix of eigenvectors and `ones(6)` for a vector of eigenvalues.

```julia
julia> Tg,Λg,Tc,λc = FlxQTL.K2Eig(Kg,Kc,true); # the last argument: LOCO::Bool = false (default)
```

For no LOCO option,

```julia
julia> T,λ = FlxQTL.K2eig(K)
```
Now start with 1D genome scan with (or without) LOCO including `Z` or not.  
For the genome scan with LOCO including `Z`, 

```julia
julia> LODs,B,est0 = FlxQTL.geneScan(1,Tg,Tc,Λg,λc,Ystd,XX,Z,true); 
```
For the genome scan with LOCO excluding `Z`, i.e. an identity matrix, 
```julia
julia> LODs,B,est0 = FlxQTL.geneScan(1,Tg,Tc,Λg,λc,Ystd,XX,true); 
```
Note that the first argument in `geneScan` is `cross::Int64`, which indicates a type of genotype or genotype probability.  For instance, if you use a 
genotype matrix whose entry is one of 0,1,2, type `1`. If you use genotype probability matrices, depending on the number of alleles or genotypes in a marker, one can type the corresponding number. i.e. `4-way cross: 4`, `HS DO mouse: 8 for alleles, 32 for genotypes`, etc.   

For no LOCO option,

```julia
julia> LODs,B,est0 = FlxQTL.geneScan(1,Tg,Tc,Λg,λc,Ystd,XX,Z);

julia> LODs,B,est0 = FlxQTL.geneScan(1,Tg,Tc,Λg,λc,Ystd,XX);
```
The function `geneScan` has three arguments: `LOD scores (LODs)`, `effects matrix under H1 (B)`, and `parameter estimates under H0 (est0)`.  
In particular, you can extract values from each matrix in `B` (3D array of matrices) to generate an effects plot.


## Generating plots

To produce a plot (or plots) for LOD scores or effects, you need first a struct of arrays, `layers` consisting of chromosomes, marker positions, 
LOD scores (or effects).

```julia
julia> Arab_lod = FlxQTL.layers(markerinfo[:,2],markerinfo[:,3],LODs)

julia> plot1d(Arab_lod;title= "LOD scores for Arabidopsis thaliana",ylabel="LOD")
```
The function `plot1d` has more keyword argument options: `yint=[]` for a vector of y-intercept(s), `yint_color=["red"]` for a vector of y-intercept 
color(s), `Legend=[]` for multiple graphs, `loc="upper right"` for the location of `Legend`.


## Performing a permutation test

Since the statistical inference for `FlxQTL` relies on LOD scores and LOD scores, the function `permTest` finds thresholds for a type I error.  The first 
argument is `nperm::Int64` to set the number of permutations for the test. For `Z = I`, type `Matrix(1.0I,6,6)` for the Arabidopsis thaliana data.

```julia
julia> maxLODs, H1par_perm, cutoff = FlxQTL.permTest(1000,1,Kg,Kc,Ystd,XX,Z;pval=[0.05 0.01])
```
