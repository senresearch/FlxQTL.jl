
## marker1Scan : CPU 1D-genome scanning under H1 only (with/without loco)
function marker1Scan(q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,q));

        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
            lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    else # cross>1
        ## scanning genotype probabilities

            B0=hcat(Nullpar.B,zeros(Float64,q,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                  lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end

#Z=I
function marker1Scan(m,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,m));
#      f= open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","w")
        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
#             f=open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","a")
#               writedlm(f,[loglik0 est1.loglik Nullpar.loglik])
#             close(f)
                 end

    else # cross>1
        ## scanning genotype probabilities

        #initialize B under the alternative hypothesis
        B0=hcat(Nullpar.B,zeros(Float64,m,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX=vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end



##MVLMM
function marker1Scan(m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4)

        nmar=size(X1,1);
    if (cross==1)
        B0=hcat(Nullpar.B,zeros(m))

             lod=@distributed (vcat) for j=1:nmar
               XX= vcat(Xnul_t,@view X1[[j],:])
               B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                     lod0= (loglik0-Nullpar.loglik)/log(10)
               est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;ρ=ρ,tol=tol1)
               [(est1.loglik-Nullpar.loglik)/log(10) est1]
                           end

    else #cross>1

        B0=hcat(Nullpar.B,zeros(m,cross-1))

        lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t,@view X1[j,2:end,:])
            B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                     [(est1.loglik-Nullpar.loglik)/log(10) est1]
                          end

    end
    return lod[:,1], lod[:,2]

end

######### actual two genescan versions including prior

"""


    geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},
             LOCO::Bool=false;m=size(Y,1),tdata::Bool=false,LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
                Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    geneScan(cross::Int64,Tg::Union{Array{Float64,3},Array{Float64,2}},Tc::Array{Float64,2},Λg::Union{Array{Float64,2},Array{Float64,1}},
             λc::Array{Float64,1},Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;m=size(Y,1),LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
        Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    geneScan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;m=size(Y,1),Xnul::Array{Float64,2}=ones(1,size(Y,2)),
                df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*5,tdata::Bool=false,LogP::Bool=false,
               itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    gene1Scan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;m=size(Y,1),
               Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*5,
                 tdata::Bool=false,LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)           


Implement 1d-genome scan with/without LOCO (Leave One Chromosome Out).  Note that `gene1Scan` includes [`getKc`](@ref) for 
    precomputing `Kc`-- no need of precomputing and doing eigen-decomposition to `Kc` separately.  The third `geneScan()` is based on a conventional MLMM:
```math
vec(Y) \\sim MVN((Z \\otimes X)vec(B) (or XBZ'),  K \\otimes \\Sigma_1 +I \\otimes \\Sigma_2),
```
where `K` is a genetic kinship,
``\\Sigma_1, \\Sigma_2`` are covariance matrices for
random and error terms, respectively.  `Z` can be replaced with an identity matrix.

# Arguments

- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Tg` : A n x n matrix of eigenvectors from [`K2eig`](@ref), or [`K2Eig`](@ref).
       Returns 3d-array of eigenvectors as many as Chromosomes if `LOCO` is true.
- `Tc` : A m x m matrix of eigenvectors from climatic relatedness matrix.
- `Λg` : A n x 1 vector of eigenvalues from kinship. Returns a matrix of eigenvalues if `LOCO` is true.
- `λc` : A m x 1 vector of eigenvalues from climatic relatedness matrix. Use `ones(m)` for no climatic information added.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[1,:]` (a vector) ->`Y[[1],:]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
      If nothing to insert in `Z`, just exclude it or insert an identity matrix, `Matrix(1.0I,m,m)`.  m traits x q phenotypic covariates.
- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out).

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(1,size(Y0))`.  Adding covariates (C) is `Xnul= vcat(ones(1,m),C)` where `size(C)=(c,m)` for `m = size(Y0,1)`.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           A large scaled covariance matrix (a weakly informative prior) is default.
- `df_prior`: degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` :  A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.
- `LogP` : Boolean. Default is `false`.  Returns ``-\\log_{10}{P}`` instead of LOD scores if `true`.

!!! Note
- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`. It works in most cases. If not,
    can reduce both `tol0` and `tol` to `1e-4` or further.


# Output

- `LODs` (or `logP`) : LOD scores. Can change to ``- \\log_{10}{P}`` in [`lod2logP`](@ref) if `LogP = true`.
- `B` : A 3-d array of `B` (fixed effects) matrices under H1: existence of QTL.  If covariates are added to `Xnul` : `Xnul= [ones(1,size(Y0)); Covariates]`,
        ex. For sex covariates in 4-way cross analysis, B[:,2,100], B[:,3:5,100] are effects for sex, the rest genotypes of the 100th QTL, respectively.
- `est0` : A type of `EcmNestrv.Approx` including parameter estimates under H0: no QTL.
"""
function geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y::Array{Float64,2},
        XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
                Xnul::Array{Float64,2}=ones(1,size(Y,2)),m=size(Y,1),df_prior=m+1,
                Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        
        q=size(Z,2);  p=Int(size(XX.X,1)/cross); 

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y,Z)
          if (λc!= ones(m))
            if (Prior!= diagm(ones(m)))
                Z1, Σ1, Ψ =transForm(Tc,Z,init.Σ,Prior)
             else # prior =I 
                Z1,Σ1 =  transForm(Tc,Z,init.Σ,true)
                Ψ =Prior
            end
            
            Y1= transForm(Tc,Y,init.Σ) # transform Y only by row (Tc)

           else
            Z1=Z; Σ1 = init.Σ
            Y1=Y;Ψ = Prior
         end
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

           for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
                 if (cross!=1)
   @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
  @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
           # rearrange B into 3-d array
           B = arrngB(H1par,size(Xnul,1),q,p,cross)

        else #no LOCO
#          Xnul_t=Xnul*Tg';
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   Y1,X1=transForm(Tg,Y1,X0,cross)
                   else
                   Y1,X1=transForm(Tg,Y1,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)
    end

    # Output choice
    if (tdata) # should use with no LOCO to do permutation
        return est0,Xnul_t,Y1,X1,Z1
    elseif (LogP) # transform LOD to -log10(p-value)
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
function geneScan(cross::Int64,Tg::Union{Array{Float64,3},Array{Float64,2}},Tc::Array{Float64,2},Λg::Union{Array{Float64,2},Array{Float64,1}},λc::Array{Float64,1},Y::Array{Float64,2},
        XX::Markers,LOCO::Bool=false;LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y,2)),m=size(Y,1),df_prior=m+1,
        Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        #  
         p=Int(size(XX.X,1)/cross);
         
         #check the prior
         if (!isposdef(Prior))
            println("Error! Plug in a postivie definite Prior!")
         end
        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y)

         if(λc!= ones(m))
            if (Prior!= diagm(ones(m)))
                Y1,Σ1,Ψ= transForm(Tc,Y,init.Σ,Prior) # transform Y only by row (Tc)
            else #prior =I
                Y1,Σ1 =  transForm(Tc,Y,init.Σ,true)
                Ψ =Prior
            end
           else
            Σ1 =init.Σ
            Y1=Y;Ψ =Prior
         end

         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

           for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,Tg[:,:,i])
                 if (cross!=1)
      @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
      @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
           # rearrange B into 3-d array
           B = arrngB(H1par,size(Xnul,1),m,p,cross)

        else #no LOCO
#          Xnul_t=Xnul*Tg';
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   Y1,X1=transForm(Tg,Y1,X0,cross)
                   else
                   Y1,X1=transForm(Tg,Y1,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
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


##MVLMM
function geneScan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    m=size(Y,1), df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*5,tdata::Bool=false,LogP::Bool=false,
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
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
          @fastmath @inbounds Y,X=transForm(Tg[:,:,i],Y,X0[maridx,:,:],cross)
                   else
           @fastmath @inbounds Y,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                    est00=nulScan(init,1,Λg[:,i],Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
                lods, H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
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
                   Y,X=transForm(Tg,Y,X0,cross)
                   else
                   Y,X=transForm(Tg,Y,XX.X,cross)
                 end


                  est0=nulScan(init,1,Λg,Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     end

    if (tdata) # should use with no LOCO
        return est0,Xnul_t,Y,X
    elseif (LogP) # transform LOD to -log10(p-value)
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


## estimating Kc + prior
function gene1Scan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;m=size(Y,1),
    Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*5,
    tdata::Bool=false,LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
    q=size(Z,2);  p=Int(size(XX.X,1)/cross);

    ## picking up initial values for parameter estimation under the null hypothesis
        init= getKc(Y;Z=Z, df_prior=df_prior, Prior=Prior,Xnul=Xnul,itol=itol,tol=tol0,ρ=ρ)
        Tc, λc = K2eig(init.Kc) 

        
          if (Prior!= diagm(ones(m)))
            Z1, Σ1, Ψ =transForm(Tc,Z,init.Σ,Prior)
           else # prior =I 
              Z1,Σ1 =  transForm(Tc,Z,init.Σ,true)
              Ψ =Prior
          end
          
          Y1= transForm(Tc,Y,init.Σ) # transform Y only by row (Tc)

         
       
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end
  if (LOCO)
      LODs=zeros(p);
      Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

         for i=1:nChr
              maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
 @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
               if (cross!=1)
 @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                 else
@fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
               end
              #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              est0=[est0;est00];
          end
         # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),q,p,cross)

      else #no LOCO
#          Xnul_t=Xnul*Tg';
          Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
               if (cross!=1)
                 Y1,X1=transForm(Tg,Y1,X0,cross)
                 else
                 Y1,X1=transForm(Tg,Y1,XX.X,cross)
               end

                est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
              LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
        B = arrngB(H1par,size(Xnul,1),q,p,cross)
  end

  # Output choice
  if (tdata) # should use with no LOCO to do permutation
      return est0,Xnul_t,Y1,X1,Z1
  elseif (LogP) # transform LOD to -log10(p-value)
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

#Z=I: estimating Kc + prior
function gene1Scan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;
    Xnul::Array{Float64,2}=ones(1,size(Y,2)),m=size(Y,1),df_prior=m+1,Prior=cov(Y,dims=2)*5,
    tdata::Bool=false,LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
      p=Int(size(XX.X,1)/cross);

    ## picking up initial values for parameter estimation under the null hypothesis
        init= getKc(Y; df_prior=df_prior, Prior=Prior,Xnul=Xnul,itol=itol,tol=tol0,ρ=ρ)
        Tc, λc = K2eig(init.Kc) 

        
          if (Prior!= diagm(ones(m)))
             Y1,Σ1,Ψ= transForm(Tc,Y,init.Σ,Prior) # transform Y only by row (Tc)
              
           else # prior =I 
              Y1,Σ1 =  transForm(Tc,Y,init.Σ,true)
              Ψ =Prior
          end
              
       
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end
  if (LOCO)
      LODs=zeros(p);
      Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

         for i=1:nChr
              maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
 @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
               if (cross!=1)
 @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                 else
@fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
               end
              #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              est0=[est0;est00];
          end
         # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),m,p,cross)

      else #no LOCO
#          Xnul_t=Xnul*Tg';
          Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
               if (cross!=1)
                 Y1,X1=transForm(Tg,Y1,X0,cross)
                 else
                 Y1,X1=transForm(Tg,Y1,XX.X,cross)
               end

                est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
              LODs,H1par=marker1Scan(m,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
        B = arrngB(H1par,size(Xnul,1),m,p,cross)
  end

  # Output choice
  if (tdata) # should use with no LOCO to do permutation
      return est0,Xnul_t,Y1,X1
  elseif (LogP) # transform LOD to -log10(p-value)
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




