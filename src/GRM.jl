"""

     GRM

A module for computing Genetic Relatedness Matrix (or kinship).


"""
module GRM

#  __precompile__(true)

using Statistics
using Distributed
using LinearAlgebra
import StatsBase: sample

# include("Miscellanea.jl")
import ..Util: Markers



"""

      kinshipMan(genematrix::Array{Float64,2})

Calculates a kinship matrix using a manhattan distance. Missing values need to be either omitted or imputed. 
This function is for recombinant inbred line (RIL) (AA/BB), not for 4-way cross genotype data.  See [`kinship4way`](@ref).

# Argument

- `genematrix` : A matrix of genotypes, i.e. 0,1 (or 1,2).  size(genematrix)= (p,n) for `p` genetic markers x `n` individuals(or lines).
               

# Output

Returns a n x n symmetric matrix containing 1's on the diagonal. 

"""
function kinshipMan(genematrix::Array{Float64,2})
#    c0=findall(.!isna.(genematrix[1,:]));
#    c1=findall(.!isna.(genematrix[2,:]));
#    ckeep=intersect(c0,c1);
#                  for j=3:nrow
#                  ck=find(.!isna.(genematrix[j,:]));
#                  ckeep=intersect(ck,ckeep);
#                   end
#                         geneupdate = genematrix[:,ckeep];
                        col = axes(genematrix,2)
                    kin=zeros(col,col);
                        @views for c=col, r=c:length(col)
                            kin[r,c]= 1.0-mean(abs.(genematrix[:,c]-genematrix[:,r]))
                                    kin[c,r]=kin[r,c]     
                        end

return kin
    
end


#The genotype data can be extracted from `cross object` using [r-qtl](https://rqtl.org/tutorials/) or [r-qtl2](https://kbroman.org/qtl2/assets/vignettes/user_guide.html).

"""
  
     kinship4way(genmat::Array{Float64,2})

Computes a kinship for four-way cross data counting different alleles between two markers: ex. AB-AB=0; AB-AC=1; AB-CD=2,``\\dots``
Note: In [R/qtl](https://cran.r-project.org/web/packages/qtl/qtl.pdf), genotypes are labeled as 1=AC; 2=BC; 3=AD; 4=BD by the function, `read.cross`.


# Argument

- `genmat` : A matrix of genotypes for `four-way cross` ``(1,2, \\dots)``. 
           size(genematrix)= (p,n), for `p` genetic markers x `n` individuals(or lines).

# Output 

Returns a n x n symmetric matrix containing 1's on the diagonal. 

"""
function kinship4way(genmat::Array{Float64,2})
    nmar, nind=axes(genmat)
    kmat=zeros(nind,nind); 
    dist=zeros(nmar);

    for i=nind 
        for j=i:length(nind)
            for k=nmar
                if (genmat[k,i]==genmat[k,j]) 
                    dist[k]= 0.0
                elseif (genmat[k,i]+genmat[k,j]==5)
                    dist[k]=2.0
                else
                    dist[k]=1.0
                end

            end
        kmat[j,i]=1.0-0.5*mean(dist)   
        kmat[i,j]=kmat[j,i]
        end
    end

return kmat

end




"""
  
     kinshipGs(climate::Array{Float64,2},ρ::Float64)
 
Computes a kinship matrix using the Gaussian Kernel.  

# Arguments

- `climate` : A matrix of genotype, or climate information data. size(climate)=(r,m), such that `r` genotype markers (or days/years of climate factors, 
            i.e. precipitations, temperatures, etc.), and `m` individuals (or environments/sites)
- `ρ` : A free parameter determining the width of the kernel. Could be attained empirically.  

# Output

Returns a m x m symmetric (positive definite) matrix containing 1's on the diagonal.

"""
function kinshipGs(climate::Array{Float64,2},ρ::Float64)
 env=axes(climate,2); 
 Kc=zeros(env,env);

    @views for c=env, r=c:length(env)
            Kc[r,c]=exp(-mean(abs.(climate[:,c]-climate[:,r]).^2)/ρ)
            Kc[c,r]=Kc[r,c]
    end

    return Kc

end


"""

    kinshipLin(mat,cross)

Calculates a kinship (or climatic relatedness, [`kinshipGs`](@ref)) matrix by linear kernel. 

# Arguments

- `mat` : A matrix of genotype (or allele) probabilities usually extracted from [R/qtl](https://rqtl.org/tutorials/rqtltour.pdf), 
        [R/qtl2](https://kbroman.org/qtl2/assets/vignettes/user_guide.html), or the counterpart packages. size(mat)= (p,n) for p genetic markers x n individuals. 
- `cross` : A scalar indicating instances of alleles or genotypes in a genetic marker. ex. 1 for genotypes (labeled as 0,1,2), 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.

# Output

Returns a n x n symmetric (positive definite) matrix containing 1's on the diagonal. 
 
See also [`kinshipCtr`](@ref), [`kinshipStd`](@ref) for genetype data.


"""
function kinshipLin(mat,cross)
r=size(mat,1)/cross; n=size(mat,2)
     K=Symmetric(BLAS.syrk('U','T',1.0,mat))/r
   @views for j=1:n
        K[j,j]=1.0
          end
    return convert(Array{Float64,2},K)
end



"""

     kinshipCtr(genmat::Array{Float64,2})

Calculates a kinship by a centered genotype matrix (linear kernel), i.e. genotypes subtracted by marker mean.

# Argument

- `genmat` : A matrix of genotype data (0,1,2). size(genmat)=(p,n) for `p` markers x `n` individuals

# Output

Returns a n x n symmetric matrix.
See also [`kinshipStd`](@ref).

"""
function kinshipCtr(genmat::Array{Float64,2})
   p=size(genmat,1)
    cgene= genmat.-mean(genmat,dims=2)
    # K=(1-ρ)*transpose(cgene)*cgene/n+ρ*Matrix(1.0I,n,n)
     #K=cgene'*cgene/p
     K=Symmetric(BLAS.syrk('U','T',1.0,cgene))/p
    return convert(Array{Float64,2},K)

end


"""

     kinshipStd(genmat::Array{Float64,2})


Calculates a kinship by a standardized (or normalized) genotype matrix (linear kernel), i.e. genotypes subtracted by marker mean and divided by marker standard deviation.  
Can also do with climatic information data. See [`kinshipGs`](@ref).

# Argument

- `genmat` : A matrix of genotype data (0,1,2). size(genmat)=(p,n) for `p` markers x `n` individuals

# Output

Returns a n x n symmetric matrix.
See also [`kinshipCtr`](@ref).

"""
function kinshipStd(genmat::Array{Float64,2})
    p=size(genmat,1)
    sgene=(genmat.-mean(genmat,dims=2))./std(genmat,dims=2)
   #(1-ρ)*transpose(sgene)*sgene/n+ρ*Matrix(1.0I,n,n)
     K=Symmetric(BLAS.syrk('U','T',1.0,sgene))/p
 
    return convert(Array{Float64,2},K)
end




"""

     shrinkg(f,nb::Int64,geno)

Estimates a full-rank positive definite kinship matrix by shrinkage intensity estimation (bootstrap).  Can only use with [`kinshipMan`](@ref), [`kinship4way`](@ref).
This function runs faster by CPU parallelization.  Add workers/processes using `addprocs()` function before running for speedup. 

# Arguments

- `f `: A function of computing a kinship. Can only use with [`kinshipMan`](@ref), [`kinship4way`](@ref).
- `nb` : An integer indicating the number of bootstrap. It does not have to be a large number.  
- `geno` : A matrix of genotypes. See [`kinshipMan`](@ref), [`kinship4way`](@ref) for dimension.

# Example

```julia
julia> using flxQTL
julia> addprocs(8) 
julia> K = shinkage(kinshipMan,20,myGeno)
```

# Output

Returns a full-rank symmetric positive definite matrix.

"""
function shrinkg(f,nb::Int64,geno)
    # generate a kinship matrix 
    K0=f(geno)
    n=size(K0,1);
    # a target matrix
#             Id=Matrix(1.0I,n,n)
    #compute an optimal regularization parameter λ_hat
    denom=norm(I-K0)^2
    #np=nprocs(); 
    GG=[];

    idx=pmap(sample,[1:n for i=1:nb],[n for i=1:nb])
    Genematrix=pmap(f,[geno[:,idx[i]] for i=1:nb])
    GG=[GG;Genematrix]

    Ks=zeros(n,n,nb)
   @views for t=1:nb
        Ks[:,:,t]=GG[t]
    end
    kinVar=var(Ks;dims=3);
λ_hat=sum(kinVar)/denom;
K=λ_hat*I+(1-λ_hat)*K0;

# println("λ_hat is"," ",λ_hat,".")
return K
end

# function shrinkg(f,nb,cross,geno)
#     # generate a kinship matrix 
#     K0=f(geno,cross)
#     n=size(K0,1);
#     # a target matrix
# #             Id=Matrix(1.0I,n,n)
#     #compute an optimal regularization parameter λ_hat
#     denom=norm(I-K0)^2
#    #np=nprocs(); itr=fld(nb,np)  #floor(nb/np);
#     GG=[];

#     idx=pmap(sample,[1:n for i=1:nb],[n for i=1:nb])
#     Genematrix=pmap(f,[geno[:,idx[i]] for i=1:nb],[cross for i=1:nb])
#     GG=[GG;Genematrix]

#     Ks=zeros(n,n,nb)
#    @views for t=1:nb
#         Ks[:,:,t]=GG[t]
#     end
#     kinVar=var(Ks;dims=3);
# λ_hat=sum(kinVar)/denom;
# K=λ_hat*I+(1-λ_hat)*K0;
# # println("λ_hat is"," ",λ_hat,".")
# return K
# end

### Shrinkage estimation of kinships using LOCO (leave one chromome out)
## compute a kinship using LOCO via shrinkage estimation
##Inputs :
## kin : any function for computing a kinship matrix, i.e. kinshipMan, kinship4way, only.
## nb: # of bootstrap iterations, the total iteration over all processes (parallel computing) is ceil(nboot/#of procs).
## g : a type of EcmNestrv.Markers.  See Markers in EcmNestrv
## Example : K_loco=shrinkgLoco(kinship4way,100,X)

"""

       shrinkgLoco(kin,nb,g::Markers)


Generates 3-d array of full-rank positive definite kinship matrices by shrinkage intensity estimation (bootstrap) using a LOCO (Leave One Chromosome Out) scheme.


# Argument

- `kin` :  A function of computing a kinship. Can only use with [`kinshipMan`](@ref), [`kinship4way`](@ref)
- `nb` : An integer indicating the number of bootstrap.
- `g` : A struct of arrays, type [`Markers`](@ref).
   

# Output

Returns 3-d array of n x n symmetric positive definite matrices as many as Chromosomes.

"""
function shrinkgLoco(kin,nb::Int64,g::Markers)
    Chr=unique(g.chr);  nChr=length(Chr); ind=size(g.X,2)
    K_loco=zeros(ind,ind,nChr)
    for j=1:nChr
        K_loco[:,:,j]=shrinkg(kin,nb,g.X[findall(g.chr.!=Chr[j]),:])
        println("Positive definiteness dropping chromosome $(Chr[j]) is ", isposdef(K_loco[:,:,j]),".")
    end
    return K_loco
end

# function shrinkgLoco(kin,nb,cross,g::Markers)
#     Chr=unique(g.chr);  nChr=length(Chr); ind=size(g.X,2)
#     K_loco=zeros(ind,ind,nChr)
#     for j=1:nChr
#         K_loco[:,:,j]=shrinkg(kin,nb,cross,g.X[findall(g.chr.!=Chr[j]),:])
#          println("Kinship leaving Chr $(j) out is completed.")
#     end
#     return K_loco
# end



"""

     kinshipLoco(kin,g::Markers,cross::Int64=1)

Generates a 3-d array of symmetric positive definite kinship matrices using LOCO (Leave One Chromosome Out) witout shrinkage intensity estimation.  
When a kinship is not positive definate, a tweak like a weighted average of kinship and Identity is used to correct minuscule negative eigenvalues. 

# Arguments

- `kin` :  A function of computing a kinship. Can only use with [`kinshipCtr`](@ref), [`kinshipStd`](@ref) for genotypes, and with [`kinshipLin`](@ref) 
          for genotype (or allele) probabilities.
- `g`   : A struct of arrays, type  [`Markers`](@ref).
- `cross` :  A scalar indicating instances of alleles or genotypes in a genetic marker. 
             ex. 1 for genotypes (0,1,2) as default, 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.  

# Output

Returns 3-d array of n x n symmetric positive definite matrices as many as Chromosomes.
Refer to [`shrinkgLoco`](@ref).

"""
function kinshipLoco(kin,g::Markers,cross::Int64=1)

    Chr=unique(g.chr);  nChr=length(Chr); ind=size(g.X,2)
    K_loco=zeros(ind,ind,nChr); 

     if(cross>1)
         K=pmap(kin,[g.X[findall(g.chr.!=Chr[j]),:] for j=1:nChr],[cross for j=1:nChr])

        @views for l=1:nChr
                     if(!isposdef(K[l]))
                        K_loco[:,:,l]=K[l]+0.01I
                       else
                        K_loco[:,:,l]=K[l]
                     end
          println("Positive definiteness dropping chromosome $(Chr[l]) is ", isposdef(K_loco[:,:,l]),".")
            end

        else #cross=1: Kinship functions not having 'cross' argument
            K=pmap(kin,[g.X[findall(g.chr.!=Chr[j]),:] for j=1:nChr])

            @views for l=1:nChr
                    if(!isposdef(K[l]))
                        K_loco[:,:,l]=K[l]+0.01I
                      else
                        K_loco[:,:,l]=K[l]
                     end
                println("Positive definiteness dropping chromosome $(Chr[l]) is ", isposdef(K_loco[:,:,l]),".")
                  end
     end

  return K_loco

end

# export kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr
# export shrinkage,shrinkgLoco,kinshipLoco

end






            







