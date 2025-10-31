

struct TNull
Y::Matrix{Float64}
Xnul::Matrix{Float64}
Z::Matrix{Float64}
Σ::Matrix{Float64}
Ψ::Matrix{Float64}
end

function transByTrait(m,Tc,Y,Z,Prior,init::Result)

#  if (λc!= ones(m))
        if (Prior!= diagm(ones(m)) && Z!= diagm(ones(m)) )
            Z1, Σ1, Ψ =transForm(Tc,Z,init.Σ,Prior)
        elseif (Z != diagm(ones(m)))
            Z1,Σ1 =  transForm(Tc,Z,init.Σ,true)
            Ψ =Prior
        else #Z & prior =I 
            Σ1 =  transForm(Tc,init.Σ,Z)
            Z1=Z; Ψ = Prior
        end
        
        Y1= transForm(Tc,Y,init.Σ) # transform Y only by row (Tc)

    return Y1,Z1,Σ1,Ψ
end
#estimate Kc with prior
function nul1Scan(init::Init0,kmin,λg,Y,Xnul,Z,m,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)
       
      # n=size(Y,2); 

    if (Z!=diagm(ones(m)))   
        B0,Kc_0,Σ1,_=ecmLMM(Y,Xnul,Z,init.B,init.Vc,init.Σ,λg,ν₀,Ψ;tol=itol)
        nulpar=NestrvAG(kmin,Y,Xnul,Z,B0,Kc_0,Σ1,λg,ν₀,Ψ;tol=tol,ρ=ρ)
        
       else #Z=I
        nulpar = nulScan(init,kmin,λg,Y,Xnul,ν₀,Ψ;ρ=ρ,itol=itol,tol=tol)
     end
    return nulpar #Result
end

#H0 MVLMM for Kc estimation
function getKc(Y::Array{Float64,2},Tg::Matrix{Float64},λg::Vector{Float64},init::Init0;m=size(Y,1),Z=diagm(ones(m)), df_prior=m+1,
     Prior::Matrix{Float64}=cov(Y,dims=2)*3,
     Xnul::Array{Float64,2}=ones(1,size(Y,2)),itol=1e-2,tol::Float64=1e-3,ρ=0.001)
     
     Y1,Xnul_t = transForm(Tg,Y,Xnul,1) #null model transformation
     
 
     est0= nul1Scan(init,1,λg,Y1,Xnul_t,Z,m,df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
      Tc, λc = K2eig(est0.Vc)
     
      #trait-wise transformation
      Y2,Z1,Σ1,Ψ = transByTrait(m,Tc,Y1,Z,Prior,est0)
      τ² =mean(Diagonal(est0.Vc)./m)
    
    return λc, TNull(Y2,Xnul_t,Z1,Σ1,Ψ),InitKc(est0.Vc,est0.B,est0.Σ,τ²,est0.loglik)
 
 end
 
 

## estimating Kc + prior
"""


    geneScan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
             Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;m=size(Y,1),H0_up::Bool=false,
             Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
             LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    
    geneScan(cross,Tg,Λg,Y,XX,false;m=size(Y,1),H0_up::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y,2)),
             df_prior=m+1,Prior=cov(Y,dims=2)*3,LogP=false,itol=1e-3,tol0=1e-3,tol=1e-4,ρ=0.001)
    
    geneScan(Tg,Λg,Y,XX,cross,false;Xnul=ones(1,size(Y,2)),df_prior=m+1,Prior=cov(Y,dims=2)*3,LogP=false,
             itol=1e-3,tol0=1e-3,tol=1e-4,ρ=0.001)


Implement 1d-genome scan with/without LOCO (Leave One Chromosome Out).  The first two functions are FlxQTL models with/without `Z` that 
preestimate a null variance component matrix (``V_C``) under H0 : no QTL, followed by its adjustment by a scalar parameter under H1 : existing QTL.  
The third `geneScan()` is based on a conventional MLMM that estimate all parameters under `H_0/H_1`.  
The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), K \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `K` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is pre-estimated under the null model (`H_0`) of no QTL from the conventional MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  

# Arguments

- `cross` : An integer (Int64) indicating the occurrence of combination of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degrees of freedom for the effect size of the genetic marker when doing genome scan.
- `Tg` : A n x n matrix of eigenvectors, or a 3d-array of eigenvectors if `LOCO ` is true.  See also [`K2eig`](@ref). 
- `Λg` : A n x 1 vector of eigenvalues from kinship. Returns a matrix of eigenvalues if `LOCO` is true.
- `Y` : A m x n matrix of response variables, i.e. m traits by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[1,:]` (a vector) ->`Y[[1],:]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (wavelet, polynomials, B-splines, etc.).
      If no assumption among traits, insert an identity matrix, `Matrix(1.0I,m,m)`, or use the second `geneScan()`.  
- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out) if `true`.

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(1,size(Y,2))`.  Adding covariates (C) is `Xnul= vcat(ones(1,n),C)` where `size(C)=(c,n)`.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default. 
- `H0_up` : Default returns null estimates, `est0` from the conventional MLMM.  It is recommended setting `H0_up=true` for ``m \\ge 11`` to avoid negative LODs. 
- `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` :  A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.
- `LogP` : Boolean. Default is `false`.  Returns ``-\\log_{10}{P}`` instead of LOD scores if `true`.

!!! Note
- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`, or increase `df_prior`, such that 
   ``m+1 \\le`` `df_prior` ``< 2m``.  The easiest setting is `df_prior = Int64(ceil(1.9m))` for numerical stability.   


# Output

- `LODs` (or `logP`) : LOD scores. Can change to ``- \\log_{10}{P}`` in [`lod2logP`](@ref) if `LogP = true`.
- `B` : A 3-d array of `B` (fixed effects) matrices under H1: existence of QTL.  If sex covariates, e.g. size(C)=(1,n), are added to `Xnul` : `Xnul= [ones(1,size(Y,2)); C]` 
        in 4-way cross analysis, B[:,2,100], B[:,3:5,100] are effects for sex, the rest genotypes of the 100th QTL, respectively.
- `est0` : A type of `EcmNestrv.Result` including parameter estimates under H0: no QTL for both functions.  Returns `EcmNestrv.Approx` if `H0_up=true` for the FlxQTL model only.
"""
function geneScan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;m=size(Y,1),H0_up::Bool=false,
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
          LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
      p=Int(size(XX.X,1)/cross); q=size(Z,2);

     if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)
      else #Z0=I
         init0=initial(Xnul,Y,false)
      end
    
            
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end
  if (LOCO)
      LODs=zeros(p);
      Chr=unique(XX.chr);est0=[];H1par=[]

        for i=eachindex(Chr)
              maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
# estimate H0 model by MVLMM and Kc
  @time λc, T0,init = getKc(Y,Tg[:,:,i],Λg[:,i],init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
               if (cross!=1)
 @fastmath @inbounds X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                 else
@fastmath @inbounds X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
               end
              # τ² estimation only for the null to get a better initial value
            @time est00=nulScan(init,1,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,df_prior,T0.Ψ,H0_up;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(nmar,q,1,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1,T0.Z,df_prior,T0.Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              if (H0_up)
               est0 =[est0;est00] #high dimensional traits
              else
               est0=[est0;Result(init.B,init.Kc,init.Σ,init.loglik)];
              end
              
          end
        
      else #no LOCO
     @time  λc,T0,init = getKc(Y,Tg,Λg,init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
               if (cross!=1)
                 X1=transForm(Tg,X0,cross)
                 else
                 X1=transForm(Tg,XX.X,cross)
               end

             @time est0=nulScan(init,1,Λg,λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,df_prior,T0.Ψ,H0_up;itol=itol,tol=tol,ρ=ρ)
              LODs,H1par=marker1Scan(p,q,1,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1,T0.Z,df_prior,T0.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
      #   B = arrngB(H1par,size(Xnul,1),q,p,cross)
        if (!H0_up) # keep null estimates from getKc
           est0=Result(init.B,init.Kc,init.Σ,init.loglik)
        end
  end

  # Output choice
    # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),q,p,cross)

  if (LogP) # transform LOD to -log10(p-value)
          if(LOCO)
              df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
          else
              df= prod(size(B[:,:,1]))-prod(size(est0.B))
          end
           logP=lod2logP(LODs,df)

      return logP,B,est0
  else
       return LODs,B,est0
   end
end

#Z=I
function geneScan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;m=size(Y,1),H0_up::Bool=false,
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
          LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
      p=Int(size(XX.X,1)/cross); 
     
         init0=initial(Xnul,Y,false)
                  
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end
  if (LOCO)
      LODs=zeros(p);
      Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

        for i=1:nChr
              maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
# estimate H0 model by MVLMM and Kc
  @time λc, T0,init = getKc(Y,Tg[:,:,i],Λg[:,i],init0;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
               if (cross!=1)
 @fastmath @inbounds X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                 else
@fastmath @inbounds X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
               end
              # τ² estimation only for the null to get a better initial value
            @time est00=nulScan(init,1,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Σ,df_prior,T0.Ψ,H0_up;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(nmar,m,1,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1,df_prior,T0.Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              if (H0_up)
               est0 =[est0;est00] #high dimensional traits
              else
               est0=[est0;Result(init.B,init.Kc,init.Σ,init.loglik)];
              end
              
          end
        
      else #no LOCO
     @time  λc,T0,init = getKc(Y,Tg,Λg,init0;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
               if (cross!=1)
                 X1=transForm(Tg,X0,cross)
                 else
                 X1=transForm(Tg,XX.X,cross)
               end

             @time est0=nulScan(init,1,Λg,λc,T0.Y,T0.Xnul,T0.Σ,df_prior,T0.Ψ,H0_up;itol=itol,tol=tol,ρ=ρ)
              LODs,H1par=marker1Scan(p,m,1,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1,df_prior,T0.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
      #   B = arrngB(H1par,size(Xnul,1),q,p,cross)
        if (!H0_up) # keep null estimates from getKc
           est0=Result(init.B,init.Kc,init.Σ,init.loglik)
        end
  end

  # Output choice
    # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),m,p,cross)

  if (LogP) # transform LOD to -log10(p-value)
          if(LOCO)
              df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
          else
              df= prod(size(B[:,:,1]))-prod(size(est0.B))
          end
           logP=lod2logP(LODs,df)

      return logP,B,est0
  else
       return LODs,B,est0
   end

end

##MVLMM
function geneScan(Tg,Λg,Y::Array{Float64,2},XX::Markers,cross::Int64,LOCO::Bool=false;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    m=size(Y,1), df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,LogP::Bool=false,
    itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

   
    p=Int(size(XX.X,1)/cross);

    #check the prior
    if (!isposdef(Prior))
        println("Error! Plug in a postivie definite Prior!")
     end

     #initialization
       init=initial(Xnul,Y,false)
        if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p); Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

       for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
#                 Xnul_t=Xnul*Tg[:,:,i]';
              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
          @fastmath @inbounds Y1,X=transForm(Tg[:,:,i],Y,X0[:,:,maridx],cross)
                   else
           @fastmath @inbounds Y1,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                    est00=nulScan(init,1,Λg[:,i],Y1,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
                lods, H1par1=marker1Scan(nmar,m,1,cross,est00,Λg[:,i],Y1,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
                LODs[maridx].=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
            # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     else #no loco
#             Xnul_t=Xnul*Tg';
             Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y1,X=transForm(Tg,Y,X0,cross)
                   else
                   Y1,X=transForm(Tg,Y,XX.X,cross)
                 end


                  est0=nulScan(init,1,Λg,Y1,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(p,m,1,cross,est0,Λg,Y1,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     end

    
     if (LogP) # transform LOD to -log10(p-value)
          if(LOCO)
                df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
            else
                df= prod(size(B[:,:,1]))-prod(size(est0.B))
            end
               logP=lod2logP(LODs,df)

        return logP,B,est0
     else
         return LODs,B,est0
     end
end




function gene1Scan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2};m=size(Y,1),
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
          LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001,δ=0.01)

       p=Int(size(XX.X,1)/cross); q=size(Z,2);LODs=zeros(p);
      Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[];
    #   Xnul_t=Array{Array{Float64,2}}(undef,nChr);fill!(Xnul_t,zeros(p0,n))
    
 println("Estimating a no-loco kinship for null variance component preestimation only for high dimension of $(m) traits")
     if (cross!=1)
        K = kinshipLin(XX.X,cross)
     else
        K= kinshipStd(XX.X)
     end
    
     if (!isposdef(K))
        K= K+δ*I
     println("Positive definiteness of the kinship is corrected to be ", isposdef(K),".")
     end

    
     if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)
       
      else #Z0=I
         init0=initial(Xnul,Y,false)
      end
 println("Pre-estimating a null variance component.")
  @time λc, T0, init = getKc(Y,K,init0;m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    
            
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end

     
   println("Impelementing genescan (LOCO = true).")
         for i=1:nChr
              maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
# estimate H0 model by MVLMM and Kc
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,Tg[:,:,i])
               if (cross!=1)
 @fastmath @inbounds Y1, X1=transForm(Tg[:,:,i],T0.Y,X0[:,:,maridx],cross)
                 else
@fastmath @inbounds Y1,X1=transForm(Tg[:,:,i],T0.Y,XX.X[maridx,:],cross)
               end
              # τ² estimation only for the null to get a better initial value
            @time est00=nulScan(init,1,Λg[:,i],λc,Y1,Xnul_t,T0.Z,T0.Σ,df_prior,T0.Ψ,true;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(nmar,q,1,cross,est00,Λg[:,i],λc,Y1,Xnul_t,X1,T0.Z,df_prior,T0.Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              est0 =[est0;est00] #high dimensional traits
              
          end
        
      

  # Output choice
    # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),q,p,cross)

  if (LogP) # transform LOD to -log10(p-value)
        df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
        logP=lod2logP(LODs,df)

      return logP,B,est0
  else
       return LODs,B,est0
   end
end

