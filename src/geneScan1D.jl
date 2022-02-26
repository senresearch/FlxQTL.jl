
## marker1Scan : CPU 1D-genome scanning under H1 only (with/without loco)
function marker1Scan(q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,q));

        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
            lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    else # cross>1
        ## scanning genotype probabilities

            B0=hcat(Nullpar.B,zeros(Float64,q,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                  lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end

#Z=I
function marker1Scan(m,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,m));
#      f= open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","w")
        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
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
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end

## upgraded version: 
function marker1Scan(m,q,kmin,cross,λg,Y1,Xnul_t,X1,Z1,Nullpar::Result,Σ1,τ2_0;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,q));

        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,τ2_0,Σ1,λg,ones(m);tol=tol0)
            lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,ones(m);ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    else # cross>1
        ## scanning genotype probabilities

            B0=hcat(Nullpar.B,zeros(Float64,q,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,τ2_0,Σ1,λg,ones(m);tol=tol0)
                  lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,ones(m);ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end



##MVLMM
function marker1Scan(m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1;ρ=0.001,tol0=1e-3,tol1=1e-4)

        nmar=size(X1,1);
    if (cross==1)
        B0=hcat(Nullpar.B,zeros(m))

             lod=@distributed (vcat) for j=1:nmar
               XX= vcat(Xnul_t,@view X1[[j],:])
               B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                     lod0= (loglik0-Nullpar.loglik)/log(10)
               est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;ρ=ρ,tol=tol1)
               [(est1.loglik-Nullpar.loglik)/log(10) est1]
                           end

    else #cross>1

        B0=hcat(Nullpar.B,zeros(m,cross-1))

        lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t,@view X1[j,2:end,:])
            B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;tol=tol1,ρ=ρ)
                     [(est1.loglik-Nullpar.loglik)/log(10) est1]
                          end

    end
    return lod[:,1], lod[:,2]

end



"""


    geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001,LogP::Bool=false)
    geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001,LogP::Bool=false)
    geneScan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;
        Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001,LogP::Bool=false)


Implement 1d-genome scan with/without LOCO (Leave One Chromosome Out).  Note that the third `geneScan()` is based on a conventional MLMM:
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
- `Y0` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[1,:]` (a vector) ->`Y[[1],:]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z0` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
      If nothing to insert in `Z0`, just exclude it or insert an identity matrix, `Matrix(1.0I,m,m)`.  m traits x q phenotypic covariates.
- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out).

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(1,size(Y0))`.  Adding covariates (C) is `Xnul= vcat(ones(1,m),C)` where `size(C)=(c,m)` for `m = size(Y0,1)`.
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
function geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y0::Array{Float64,2},
        XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        m=size(Y0,1);
        q=size(Z0,2);  p=Int(size(XX.X,1)/cross);

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0,Z0)
          if(λc!= ones(m))
            Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
            Y1= transForm(Tc,Y0,init.Σ) # transform Y only by row (Tc)
           else
            Z1=Z0; Σ1 = init.Σ
            Y1=Y0
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
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                 if (cross!=1)
                   Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
                   Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,Z1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
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

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)
    end

    # Output choice
    if (tdata) # should use with no LOCO to do permutation
        return LODs,B,est0,Y1,X1,Z1
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
function geneScan(cross::Int64,Tg::Union{Array{Float64,3},Array{Float64,2}},Tc::Array{Float64,2},Λg::Union{Array{Float64,2},Array{Float64,1}},λc::Array{Float64,1},Y0::Array{Float64,2},
        XX::Markers,LOCO::Bool=false;LogP::Bool=false,
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

         m=size(Y0,1);
         p=Int(size(XX.X,1)/cross);

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0)
         if(λc!= ones(m))
            Σ1=convert(Array{Float64,2},Symmetric(BLAS.symm('R','U',init.Σ,Tc)*Tc'))
            Y1= transForm(Tc,Y0,init.Σ) # transform Y only by row (Tc)
           else
            Σ1 =init.Σ
            Y1=Y0
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
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                 if (cross!=1)
                   Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
                   Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Σ1;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
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

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Σ1;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,λc,Y1,Xnul_t,X1;tol0=tol0,tol1=tol,ρ=ρ)
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
function geneScan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
        Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    m=size(Y0,1);
    p=Int(size(XX.X,1)/cross);
     #initialization
       init=initial(Xnul,Y0,false)
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
                   Y,X=transForm(Tg[:,:,i],Y0,X0[maridx,:,:],cross)
                   else
                   Y,X=transForm(Tg[:,:,i],Y0,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                    est00=nulScan(init,1,Λg[:,i],Y,Xnul_t;itol=itol,tol=tol,ρ=ρ)
                lods, H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],Y,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
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
                   Y,X=transForm(Tg,Y0,X0,cross)
                   else
                   Y,X=transForm(Tg,Y0,XX.X,cross)
                 end


                  est0=nulScan(init,1,Λg,Y,Xnul_t;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,Y,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     end

    if (tdata) # should use with no LOCO
        return LODs,B,est0,Y,X
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



## export functions
# export geneScan,marker1Scan
## upgraded version: H0:MLMM, H1:FlxQTL

function obtainKc(Kg::Array{Float64,2},Y0::Array{Float64,2},Z0::Array{Float64,2};
        Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    
      T,λ = K2eig(Kg)
      init=initial(Xnul,Y0,Z0,false)
      Y=BLAS.gemm('N','T',Y0,T)
      Xnul_t=BLAS.gemm('N','T',Xnul,T)
      est0=nulScan(init,1,λ,Y,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
      return est0.Vc
end


# function gene1Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},
#                 LOCO::Bool=false;LogP::Bool=false,
#                 Xnul::Array{Float64,2}=ones(1,size(Y0,2)),kmin::Int64=1,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
      
#      init=initial(Xnul,Y0,Z0,false)
          
#        if (LOCO)
#         #estimate Kc
          
          
#           @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,1])
#            Y=BLAS.gemm('N','T',Y0,Tg[:,:,1])
#            est=nulScan(init,kmin,Λg[:,1],Y,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
#             Tc,λc=K2eig(est.Vc)
#            res1,B,est0 = geneScan(cross,Tg,Tc,Λg,λc,Y0,XX,Z0,true;LogP=LogP,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)
        
#          else
          
        
#           @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,Tg)
#            Y=BLAS.gemm('N','T',Y0,Tg)
#            est=nulScan(init,kmin,Λg,Y,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
#             Tc,λc=K2eig(est.Vc)
#            res1,B,est0 =geneScan(cross,Tg,Tc,Λg,λc,Y0,XX,Z0;LogP=LogP,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)
        
        
#        end
    
  
#     return res1, B, est0
# end


## upgraded version1: H0:MLMM, H1:FlxQTL (develop further later)
# function gene0Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},
#                 LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
#                 Xnul::Array{Float64,2}=ones(1,size(Y0,2)),kmin::Int64=1,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

#         m=size(Y0,1);
#         q=size(Z0,2);  p=Int(size(XX.X,1)/cross);

#         ## picking up initial values for parameter estimation under the null hypothesis
#             init=initial(Xnul,Y0,Z0,false)
# #           if(λc!= ones(m))
# #             Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
# #             Y1= transForm(Tc,Y0,init.Σ) # transform Y only by row (Tc)
# #            else
# #             Z1=Z0; Σ1 = init.Σ
# #             Y1=Y0
# #          end
#          if (cross!=1)
#             X0=mat2array(cross,XX.X)
#          end
#     if (LOCO)
#         LODs=zeros(p);
#         Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

#            for i=1:nChr
#                 maridx=findall(XX.chr.==Chr[i])
# #                 Xnul_t=Xnul*Tg[:,:,i]';
#    @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
#                  if (cross!=1)
#                    Y1,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
#                    else
#                    Y1,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
#                  end
#                 #parameter estimation under the null using MLMM
#                 est00=nulScan(init,kmin,Λg[:,i],Y1,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
#                 Y2,Σ1,Z,τ1=transform(est00.Vc,Y1,Z0,est00.Σ)
#              lods,H1par1=marker1Scan(m,q,kmin,cross,Λg[:,i],Y2,Xnul_t,X1,Z,est00,Σ1,τ1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
#                 LODs[maridx]=lods
#                 H1par=[H1par;H1par1]
#                 est0=[est0;est00];
#             end
#            # rearrange B into 3-d array
#            B = arrngB(H1par,size(Xnul,1),q,p,cross)

#         else #no LOCO
# #          Xnul_t=Xnul*Tg';
#             Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
#                  if (cross!=1)
#                    Y1,X1=transForm(Tg,Y1,X0,cross)
#                    else
#                    Y1,X1=transForm(Tg,Y1,XX.X,cross)
#                  end

#                   est0=nulScan(init,kmin,Λg,λc,Y1,Xnul_t,Z1,Σ1;itol=itol,tol=tol,ρ=ρ)
#                 LODs,H1par=marker1Scan(q,kmin,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
#              # rearrange B into 3-d array
#           B = arrngB(H1par,size(Xnul,1),q,p,cross)
#     end

#     # Output choice
#     if (tdata) # should use with no LOCO to do permutation
#         return LODs,B,est0,Y1,X1,Z1
#     elseif (LogP) # transform LOD to -log10(p-value)
#             if(LOCO)
#                 df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
#             else
#                 df= prod(size(B[:,:,1]))-prod(size(est0.B))
#             end
#              logP=lod2logP(LODs,df)

#         return logP,B,est0
#     else
#          return LODs,B,est0
#      end
# end
