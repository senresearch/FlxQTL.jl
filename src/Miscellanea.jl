
"""

   Util

A module for utility functions.

"""
module Util

using Random
using Distributed
import LossFunctions: HuberLoss
import Distributions: Chisq,ccdf
import StatsBase: mad, sample
import Statistics: mean, var, median


"""

    setSeed(lw::Int64,up::Int64,replace::Bool=false)

Assigns different numbers of seeds to workers (or processes).

# Arguments

- `lw` : A lower bound to set seeds.
- `up` : A upper bound to set seeds.
- `replace` : Sampling seed numbers between `lw` and `up` with/without replacement. Default is `false: without replacement`. 
          Since the function itself recognize the number of processes (or workers) and their id, a wider range of seed numbers needs to set for default.

# Examples

```@example
using .Util
using Random
addprocs(10)
setSeed(1,20)

```

"""
function setSeed(lw::Int64,up::Int64,replace::Bool=false)
np=nprocs();pid=procs()
seed=sample(lw:up,np,replace=replace)
for i=1:np
remotecall_fetch(()->Random.seed!(seed[i]),pid[i])
end
end

### 'using InteractiveUtils' to use @spawnat 3 whos 


"""

    Markers(name::Array{String,1},chr::Array{Any,1},pos::Array{Float64,1},X::Array{Float64,2})

A struct of arrays creating genotype or genotype probability data for genome scan.

# Arguments

- `name` : A vector of marker names
- `chr`  : A vector of Chromosomes
- `pos`  : A vector of marker positions (cM)
- `X` : A matrix of genotypes or genotype probabilities

"""
struct Markers
name::Array{String,1}
chr::Array{Any,1}
pos::Array{Float64,1}
X::Array{Float64,2} 
end



"""

    ordrMarkers(markers::Array{Any,2})

Rearrange by CPU parallelization marker information composed of marker name, chr, position obtained from rqtl2, which is not listed in order (excluding `X` chromosome).

# Argument

- `markers` : An array of marker information.

"""
function ordrMarkers(markers::Array{Any,2})
Chr=sort(unique(markers[:,2]));nChr=length(Chr)
idx=pmap(findall,[markers[:,2].==Chr[j] for j=1:nChr])
newmarkers= @distributed (vcat) for j=1:nChr
         markers[idx[j],:]
  end
return newmarkers
end


## stacking a matrix to a vector (vectorizing a matrix)
"""

   mat2vec(mat)

Stacks a matrix to a vector, i.e. vectorizing a matrix.

"""
function mat2vec(mat)
col=size(mat,2);
   vec=mat[:,1]
for j=2:col
    vec=vcat(vec,mat[:,j])
end
return vec
end



"""

    mat2array(cross::Int64,X0)

Returns a matrix of genotype probabilities to 3-d array. size(X0)=(p1,n) --> (p,cross,n), where `p1` = ` cross*p` for `p` markers, 
`cross` alleles or genotypes, and `n` individuals.

# Argument

- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
- `X0` : A matrix of genotype probability data computed from r/qtl or r/qtl2.  

See [`array2mat`](@ref).

"""
function mat2array(cross::Int64,X0)
n=size(X0,2); p=Int(size(X0,1)/cross)
X=zeros(p,cross,n); 
@inbounds @views for j=1:p
 X[j,:,:]=  X0[cross*j-(cross-1):cross*j,:]
          end
return X
end


"""

    array2mat(cross::Int64,X0::Array{Float64,3})

Returns a 3-d array to a matrix of genotype probabilities. size(X0)=(p,cross,n) --> (p1,n), where `p1` = ` cross*p` for `p` markers, 
`cross` alleles or genotypes, and `n` individuals.
See [`mat2array`](@ref).

"""
function array2mat(cross::Int64,X0::Array{Float64,3})
p=size(X0,1); n=size(X0,3)
X=zeros(p*cross,n)
@inbounds @views for j=1:p
    X[cross*j-(cross-1):cross*j,:]= X0[j,:,:]
        end
return X
end


### changing a 3d-array to a matrix with covariates or a row of intercepts
## adding covariates to build up a matrix of qtl for forward selection 
# default of Xnul = 1's (before transformation). 
#If want to add covariates, Xnul= vcat(ones(1,n),covariates) then transform it.
function array2mat(qtl,cross,Xnul_t,Xt)
if (cross!=1)
    ### building a covariate matrix from 3d-arrays   
    covar=[Xnul_t ;@view Xt[qtl[1],2:end,:]]
    for j=2:length(qtl)
        covar=[covar;@view Xt[qtl[j],2:end,:]]
    end
    return covar
    else #cross=1
    return vcat(Xnul_t,@view Xt[qtl,:])
end


end

### changing a 3d-array to a matrix with covariates or a row of intercepts
### adding covariates (or an intercept)to the full model obtained in the forward selection step 
#(for backward elimination)
# 
function array2mat(cross,Xnul_t,Xful)
if (cross!=1)
    ## size(Xful)=(p,cross,n)
    nqtl=size(Xful,1); 

    Xf=[Xnul_t ; Xful[1,2:end,:]]
    for j=2:nqtl
        Xf=[Xf; Xful[j,2:end,:]]
    end
    return Xf 
else
    return [Xnul_t;Xful]
end

end




"""

    getGenoidx(GenoData::Array{Any,2},maf::Float64=0.025)

Attains genotype indices to drop correlated or bad markers.

# Arguments
- `GenoData` : A matrix of genotype data. size(GenoData)= (p,n) for `p` markers and `n` individuals.
- `maf` : A scalar for dropping criteron of markers. Default is `0,025` i.e. markers of MAF < 0.025 are dropped.

"""
function getGenoidx(GenoData::Array{Any,2},maf::Float64=0.025)
#   id1=(var(GenoData,dims=2).==0.0)
#   getId=LinearIndices(id1)[findall(id1.==false)]
#     return getindex.(findall(var(GenoData,dims=2).>= 0.1),1)  
return getindex.(findall(mean(GenoData,dims=2)/2.0.>= maf),1)   
end


"""

    getFinoidx(phenoData::Array{Union{Missing,Float64},2})

Attains indices of phenotype data without missing values.

# Argument

- `phenoData` : A matrix of phenotype (or trait) data including missing values. size(phenoData) = (m,n) for `m` traits and `n` individuals.

"""
function getFinoidx(phenoData::Array{Union{Missing,Float64},2})

return getindex.(findall(ismissing.(sum(phenoData,dims=1)),2))

end

## change lods to asymptotic scales (-log(10,p))
##Input : v :degrees of freedom for Chi-squared distribution
"""

    lod2logP(LODs::Union{Array{Float64,1},Array{Any,1}},v::Int64)

Caculates ``-log[10]{P}`` from LOD scores.

# Arguments
    
- `LODs` : A vector of LOD scores computed from genome scan.
- `v` : A degree of freedom for Chi-squared distribution.
    
"""
function lod2logP(LODs::Union{Array{Float64,1},Array{Any,1}},v::Int64)
return -log.(10,(ccdf.(Chisq(v),2*log(10)*LODs)))
end


## get markers by sorting out markers every 2cM (step=2) for 2D scan
# size(markers)=(n individuals,p markers)
    
"""
    
        sortBycM(Chr::Any,XX::Markers,cross::Int64,cM::Int64=2)

Returns marker indices in Chromosome `Chr` and the corresponding genotype probabilities keeping only markers positioned in every `cM` centimorgans 
for 2-d genome scan to avoid singularity.

# Arguments

- `Chr` : A type of Any indicating a particular chromosome to sort markers out.
- `XX` : A type of `Markers`. See [`Markers`](@ref).
- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
- `cM` : An integer of dropping criterion of markers. Default is 2, i.e. keeping only markers in every 2 cM, or dropping markers within 2cM between 2 markers.
    
See also [`newMarkers`](@ref)
      
"""
function sortBycM(Chr::Any,XX::Markers,cross::Int64,cM::Int64=2)

    idx=findall(XX.chr.==Chr)
    Int_pos=floor.(Int,XX.pos[idx])
    idx1=first(idx); npos=length(unique(Int_pos))
    #getting actual indices for distinct marker positions
    for l=2:length(idx)
        if(Int_pos[l-1]!=Int_pos[l])
            idx1=[idx1;idx[l]]
        end
    end
    #getting markers in every # cM 
#         M=Int(ceil(npos/cM))
#         oidx=[cM*i-(cM-1) for i=1:M]
    M=Int(floor(npos/cM))
    oidx=[cM*i for i=1:M]
    pidx=[cross*idx1[oidx][l]-(cross-1):cross*idx1[oidx][l] for l in eachindex(idx1[oidx])]
    geno=convert(Array{Float64,2},XX.X')
        pr2=geno[:,pidx[1]]
    for s=2:M
        pr2=[pr2 geno[:,pidx[s]]]
    end

return idx1[oidx], pr2'

end

## get a struct of a new Marker after sorting markers out by every #cM
    
"""
    
    newMarkers(XX::Markers,cross::Int64,cM::Int64=2)
    
Returns a struct of Markers by keeping only markers positioned in every `cM` centimorgans for 2-d genome scan to avoid singularity.
    
# Arguments
    
- `XX` : A type of `Markers`. See [`Markers`](@ref).
- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
- `cM` : An integer of dropping criterion of markers. Default is 2, i.e. keeping only markers in every 2 cM, or dropping markers within 2cM between 2 markers.
       
"""
function newMarkers(XX::Markers,cross::Int64,cM::Int64=2)

chr=XX.chr;Chr=unique(XX.chr); nchr=length(Chr)

    maridx,genopr= sortBycM(Chr[1],XX,cross,cM)

for j=2:nchr
    maridx1,genopr1=sortBycM(Chr[j],XX,cross,cM)
    maridx=[maridx;maridx1]
    genopr=[genopr; genopr1]
end

return Markers(XX.name[maridx],XX.chr[maridx],XX.pos[maridx],genopr)
end



function huberize(y::Vector{Float64})
m = median(y)
s = mad(y,normalize=true)
z = (y.-m)./s
l = value.(HuberLoss(1),z)
x = sign.(z).* sqrt.(2*l)
return m .+ s.*x
end

"""
    
    Y_huber(Y::Array{Float64,2})
    
Rescale Y (phenotype or trait data) to be less sensitive to outliers using by Huber loss function and MAD (median absolute deviation). 
size(Y)=(m,n) for m trait and n individuals.
    
"""       
function Y_huber(Y::Array{Float64,2})

midx = axes(Y,2)
@fastmath @inbounds for i = midx
   Y[:,i]=huberize(Y[:,i])
end
return Y
end

# export setSeed, Markers, mat2vec, getGenoidx,lod2logP,ordrMarkers,mat2array, array2mat,sortBycM,newMarkers,huberize,Y_huber


end