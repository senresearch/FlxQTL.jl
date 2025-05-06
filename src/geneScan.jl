
function marker1Scan(nmar,cross,nulpar::Estimat,Y,Xnul,X,Z,reml)
    
    if(cross!=1)
        lod= @distributed (vcat) for j=1:nmar
            est1 = mGLM(Y,hcat(Xnul,X[:,2:end,j]),Z,reml)
            [(est1.loglik-nulpar.loglik)/log(10) est1]
        end

     else #cross =1
        lod = @distributed (vcat) for j=1:nmar
        est1 = mGLM(Y,hcat(Xnul,X[:,j]),Z,reml)
        [(est1.loglik-nulpar.loglik)/log(10) est1]
         end

    end
     return lod[:,1], lod[:,2]
end

#Z=I
function marker1Scan(nmar,cross,nulpar::Estimat,Y,Xnul,X,reml)
    
    if(cross!=1)
        lod= @distributed (vcat) for j=1:nmar
            est1 = mGLM(Y,hcat(Xnul,X[:,2:end,j]),reml)
            [(est1.loglik-nulpar.loglik)/log(10) est1]
        end

     else #cross =1
        lod = @distributed (vcat) for j=1:nmar
        est1 = mGLM(Y,hcat(Xnul,X[:,j]),reml)
        [(est1.loglik-nulpar.loglik)/log(10) est1]
         end

    end
     return lod[:,1], lod[:,2]
end


function marker2Scan!(LODs,mindex::Array{Int64,1},cross,nulpar::Estimat,Y,Xnul,X,Z,reml)
      M=length(mindex)
    if(cross!=1)
        for j=1:M-1
            lod=@distributed (vcat) for l=j+1:M
            est1=mGLM(Y,hcat(Xnul,X[:,2:end,j],X[:,2:end,l]),Z,reml)
            (est1.loglik-nulpar.loglik)/log(10)
                  end
            LODs[mindex[j+1:end],mindex[j]].=lod
        end
    else #cross=1
        for j=1:M-1
            lod=@distributed (vcat) for l=j+1:M
            est1=mGLM(Y,hcat(Xnul,X[:,[j,l]]),Z,reml)
            (est1.loglik-nulpar.loglik)/log(10) 
               end
            LODs[mindex[j+1:end],mindex[j]].=lod
        end
    end

end

#Z=I
function marker2Scan!(LODs,mindex::Array{Int64,1},cross,nulpar::Estimat,Y,Xnul,X,reml)
    M=length(mindex)
  if(cross!=1)
      for j=1:M-1
          lod=@distributed (vcat) for l=j+1:M
          est1=mGLM(Y,hcat(Xnul,X[:,2:end,j],X[:,2:end,l]),reml)
          (est1.loglik-nulpar.loglik)/log(10) 
           end
          LODs[mindex[j+1:end],mindex[j]].=lod
      end
  else #cross=1
      for j=1:M-1
          lod=@distributed (vcat) for l=j+1:M
          est1=mGLM(Y,hcat(Xnul,X[:,[j,l]]),reml)
          (est1.loglik-nulpar.loglik)/log(10) 
            end
          LODs[mindex[j+1:end],mindex[j]].=lod
      end
    end

end

function mat2Array(cross::Int64,p,X)
   #size(X)=(n,p)
    n= size(X,1)
   X0=zeros(n,cross,p)
   @inbounds @views for j = 1:p
    X0[:,:,j] = X[:,cross*j-(cross-1):cross*j]
   end

   return X0

end

function getB(H1par,p0::Int64,q,p,cross)
  #p0= size(Xnul,2)
    if (cross!=1)
       B=zeros(cross-1+p0,q,p)
    else
       B=zeros(cross+p0,q,p)
    end

      @inbounds @views for j=1:length(H1par)
           B[:,:,j]=H1par[j].B
       end
   return B
end

##########
"""

    mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;LogP::Bool=false,
              Xnul::Matrix{Float64}=ones(size(Y,1),1))
    mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;LogP::Bool=false,
              Xnul::Matrix{Float64}=ones(size(Y,1),1))

Implement 1d-genome scan.  The second `mlm1Scan()` is for `Z=I` case; 
one can also run the first by inserting an identity matrix (I) into `Z`.
```math
 vec(Y) \\sim MVN (  (Z \\otimes X)vec(B) (or XBZ'), \\Sigma \\otimes I),
 ```
where size(Y)=(n,m), size(X)=(n,p), size(Z)=(m,q).

# Arguments

- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Y` : A n x m matrix of response variables, i.e. n individuals (or lines) by m traits (or environments). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[:,1]` (a vector) ->`Y[:,[1]]` (a matrix) .
- `XX` : A type of [`Markers`](@ref). Be cautious when combining genotype infomation into the struct of `Markers`; `size(X) = (n,p)`.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
      If nothing to insert in `Z`, just exclude it or insert an identity matrix, `Matrix(1.0I,m,m)`.  m traits x q phenotypic covariates.
- `reml`: Boolean.  Default is fitting the model via mle. Resitricted MLE is implemented if `true`.

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(size(Y,1),1)`.  Adding covariates (C) is `Xnul= hcat(ones(n),C)` where `size(C)=(c,n)` for `n = size(Y,1)`.
- `LogP` : Boolean. Default is `false`.  Returns ``-\\log_{10}{P}`` instead of LOD scores if `true`.

# Output

- `LODs` (or `logP`) : LOD scores. Can change to ``- \\log_{10}{P}`` in [`lod2logP`](@ref) if `LogP = true`.
- `B` : A 3-d array of `B` (fixed effects) matrices under H1: existence of QTL.  If covariates are added to `Xnul` : `Xnul= [ones(size(Y,1)) Covariates]`,
        ex. For sex covariates in 4-way cross analysis, B[2,:,100], B[3:5,:,100] are effects for sex, the rest genotypes of the 100th QTL, respectively.
- `est0` : A type of `MLM.Estimat` including parameter estimates under H0: no QTL.              

"""
function mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;LogP::Bool=false,
    Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); q=size(Z,1)
    LODs = zeros(p); H1par=[]
    

     #nul scan
     est0=mGLM(Y,Xnul,Z,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
    else
        X= XX.X
    end
        LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,X,Z,reml)
        B= getB(H1par,size(Xnul,2),q,p,cross)
    
    if(LogP)
        df = prod(size(B[:,:,1]))-prod(size(est0.B))
        logP=lod2logP(LODs,df)
        return logP, B, est0
    else #print lods
        return LODs,B, est0
    end
       
end

#Z=I
function mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;LogP::Bool=false,
    Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), 
    p=Int(size(XX.X,2)/cross); m=size(Y,2)
    LODs = zeros(p); H1par=[]
    

     #null scan
     est0=mGLM(Y,Xnul,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
    else
        X = XX.X
    end

       LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,X,reml)
       B= getB(H1par,size(Xnul,2),m,p,cross)
    

    if(LogP)
        df = prod(size(B[:,:,1]))-prod(size(est0.B))
        logP=lod2logP(LODs,df)
        return logP, B, est0
    else #print lods
        return LODs,B, est0
    end
       
end

#######

"""

     mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))
     mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))

Implement 2d-genome scan.  The second `mlm2Scan()` is for `Z=I` case; 
one can also run the first by inserting an identity matrix (I) into `Z`.
```math
 vec(Y) \\sim MVN (  (Z \\otimes X)vec(B) (or XBZ'), \\Sigma \\otimes I),
 ```
where size(Y)=(n,m), size(X)=(n,p), size(Z)=(m,q).

# Arguments

- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Y` : A n x m matrix of response variables, i.e. n individuals (or lines) by m traits (or environments). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[:,1]` (a vector) ->`Y[:,[1]]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).  Be cautious when combining genotype infomation into the struct of `Markers`; `size(X) = (n,p)`.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
      If nothing to insert in `Z`, just exclude it or insert an identity matrix, `Matrix(1.0I,m,m)`.  m traits x q phenotypic covariates.
- `reml`: Boolean.  Default is fitting the model via mle. Resitricted MLE is implemented if `true`.

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's).  Unless adding covariates, just leave as it is.  See [`mlm1Scan`](@ref).

# Output

- `LODs` : LOD scores. Can change to ``- \\log_{10}{P}`` using [`lod2logP`](@ref).
- `est0` : A type of `MLM.Estimat` including parameter estimates under H0: no QTL.


"""
function mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); q=size(Z,1)
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);
 
    #nul scan
    est0=mGLM(Y,Xnul,Z,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
        for j=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[j])
            marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,X[:,:,maridx],Z,reml)
        end
       
     else #cross=1
        for j= eachindex(Chr)
            maridx=findall(XX.chr.==Chr[j])
         marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,XX.X[:,maridx],Z,reml) 
        end
    end
     
    
        return LODs, est0
    
    
end

#Z=I
function mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))

# size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); 
    LODs=zeros(p,p);  Chr=unique(XX.chr);

#nul scan
    est0=mGLM(Y,Xnul,reml)
  if(cross!=1)
    X = mat2Array(cross,p,XX.X)
    for j=eachindex(Chr)
        maridx=findall(XX.chr.==Chr[j])
        marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,X[:,:,maridx],reml)
    end
   
   else #cross=1
    for j= eachindex(Chr)
        maridx=findall(XX.chr.==Chr[j])
     marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,XX.X[:,maridx],reml) 
    end
  end
 
  return LODs, est0

end

#########

"""
  
     mlmTest(nperm::Int64,cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;
              Xnul::Matrix{Float64}=ones(size(Y,1),1),pval=[0.05 0.01])
     mlmTest(nperm::Int64,cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;
              Xnul::Matrix{Float64}=ones(size(Y,1),1),pval=[0.05 0.01])
            
Implement permutation test to get thresholds at the levels of type 1 error, `α`.  The second `mlmTest()` is for `Z=I` case; 
one can also run the first by inserting an identity matrix (I) into `Z`.
```math
 vec(Y) \\sim MVN (  (Z \\otimes X)vec(B) (or XBZ'), \\Sigma \\otimes I),
 ```
where size(Y)=(n,m), size(X)=(n,p), size(Z)=(m,q).


# Arguments

- `nperm` : An integer indicating the number of permutation to be implemented.
- `cross` : An integer indicating the number of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Y` : A n x m matrix of response variables, i.e. n individuals (or lines) by m traits (or environments). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[:,1]`  (a vector) -> `Y[:,[1]]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).  Be cautious when combining genotype infomation into the struct of `Markers`; `size(X) = (n,p)`.
- `reml`: Boolean.  Default is fitting the model via mle. Resitricted MLE is implemented if `true`.

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
        Default is an identity matrix for the dimension of m traits x q phenotypic covariates.

# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A type of struct, `MLM.Estimat(B,Σ,loglik)` including parameter estimates  
                for a MLM under H0: no QTL by permutation. 
- `cutoff` : A vector of thresholds corresponding to `pval`.

              
     

"""
function mlmTest(nperm::Int64,cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;
    Xnul::Matrix{Float64}=ones(size(Y,1),1),pval=[0.05 0.01])
      
     n= size(Y,1);p=Int(size(XX.X,2)/cross); maxlod=zeros(nperm)
     H1par= Vector{Any}(undef,nperm)
     
     if(cross!=1)
        X = mat2Array(cross,p,XX.X)
     else
        X= XX.X
     end
     mlmpermutation!(maxlod,H1par,nperm,cross,n,p,Y,X,Z,reml;Xnul=Xnul)

     cutoff = quantile(maxlod,1.0.-pval)
      
     return cutoff, maxlod, H1par
    
end

## Z=I
function mlmTest(nperm::Int64,cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;
    Xnul::Matrix{Float64}=ones(size(Y,1),1),pval=[0.05 0.01])
      
     n= size(Y,1);p=Int(size(XX.X,2)/cross); maxlod=zeros(nperm)
     H1par= Vector{Any}(undef,nperm)
     
     if(cross!=1)
        X = mat2Array(cross,p,XX.X)
     else
        X= XX.X
     end
     mlmpermutation!(maxlod,H1par,nperm,cross,n,p,Y,X,reml;Xnul=Xnul)
     cutoff = quantile(maxlod,1.0.-pval) 
     return cutoff, maxlod, H1par
    
end


function mlmpermutation!(maxlod,H1par,nperm,cross,n,p,Y,X,Z,reml;Xnul=Xnul)

    for l=1: nperm
        Y1= Y[shuffle(1:n),:]
        est0=mGLM(Y1,Xnul,Z,reml)
        LODs,H1par[l] = marker1Scan(p,cross,est0,Y,Xnul,X,Z,reml)
        maxlod[l]= maximum(LODs); 
        if (mod(l,100)==0)
            println("Scan for $(l)th permutation is done.")
        end
    end

end

#Z=I
function mlmpermutation!(maxlod,H1par,nperm,cross,n,p,Y,X,reml;Xnul=Xnul)

    for l=1: nperm
        Y1= Y[shuffle(1:n),:]
        est0=mGLM(Y1,Xnul,reml)
        LODs,H1par[l] = marker1Scan(p,cross,est0,Y,Xnul,X,reml)
        maxlod[l]= maximum(LODs); 
        if (mod(l,100)==0)
            println("Scan for $(l)th permutation is done.")
        end
    end

end