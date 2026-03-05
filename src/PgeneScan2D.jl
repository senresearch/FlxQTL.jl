


function marker2Scan!(LODs,mindex::Array{Int64,1},q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(q,2*(cross-1)))

              for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                      XX=@views vcat(Xnul_t,X1[2:end,:,j],X1[2:end,:,l])
                      B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                      est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                                   end
                 @views LODs[mindex[j+1:end],mindex[j]].=lod
               end

     else #cross=1
        B0=hcat(Nullpar.B,zeros(q,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                       XX=@views vcat(Xnul_t,X1[[j,l],:])
                       B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                       est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                               end
               @views LODs[mindex[j+1:end],mindex[j]].=lod
                end

   end #if cross
   # return LODs #[Lods, H1_parameters]
end

#MVLMM
function marker2Scan!(LODs,mindex::Array{Int64,1},m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1,ν₀,Ψ;tol0=1e-3,tol1=1e-4,ρ=0.001)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(m,2*(cross-1)))

               for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                           XX=@views vcat(Xnul_t,X1[2:end,:,j],X1[2:end,:,l])
                           B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                            lod0=(loglik0-Nullpar.loglik)/log(10)
                           est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                           (est1.loglik-Nullpar.loglik)/log(10)
                                       end
               @views LODs[mindex[j+1:end],mindex[j]] .=lod
               end

        else #cross=1
          B0=hcat(Nullpar.B,zeros(m,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                         XX=@views vcat(Xnul_t,X1[[j,l],:])
                         B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                             lod0=(loglik0-Nullpar.loglik)/log(10)
                         est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                        (est1.loglik-Nullpar.loglik)/log(10)
                #  println([j l])
                                       end
                 @views LODs[mindex[j+1:end],mindex[j]] .=lod
                end

    end #if cross

end

####

function geneScan2(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;m=size(Y,1),Z=diagm(ones(m)),
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
          itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);q=size(Z,2);kmin=1
    LODs=zeros(p,p);  Chr=unique(XX.chr); 
           ## initialization
           if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)
      else #Z0=I
         init0=initial(Xnul,Y,false)
      end
              
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
         
    if (LOCO)
        est0=[];
         for i= eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);
# estimate H0 model by MVLMM and Kc
   λc, T0,init = getKc(Y,Tg[:,:,i],Λg[:,i],init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
               if (cross!=1)
 @fastmath @inbounds X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                 else
 @fastmath @inbounds X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
               end
           
          est00=nulScan(init,kmin,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,df_prior,T0.Ψ,true;ρ=ρ,itol=itol,tol=tol)
          marker2Scan!(LODs,maridx,q,kmin,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1,T0.Z,df_prior,T0.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
          est0 =[est0;Result(est00.B,est00.τ2*init.Kc,est00.Σ,est00.loglik)] 
         end

     else #no LOCO

   λc,T0,init = getKc(Y,Tg,Λg,init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
                 if (cross!=1)
                 X1=transForm(Tg,X0,cross)
                 else
                 X1=transForm(Tg,XX.X,cross)
               end

         est0=nulScan(init,kmin,Λg,λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,df_prior,T0.Ψ,true;itol=itol,tol=tol,ρ=ρ)

             for i=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1,T0.Z,df_prior,T0.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
             end
            est0 = Result(est0.B,est0.τ2*init.Kc,est0.Σ,est0.loglik)
    end

    return LODs,est0
end

##MVLMM
function geneScan2(Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},Y::Array{Float64,2},XX::Markers,cross::Int64,LOCO::Bool=false;m=size(Y,1),
                   Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
                  Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);kmin=1
    Chr=unique(XX.chr);  LODs=zeros(p,p);est0=[];

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
            for i=eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
             @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   Y,X=transForm(Tg[:,:,i],Y,X0[:,:,maridx],cross)
                   else
                   Y,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end

                   
                 est=nulScan(init,kmin,Λg[:,i],Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,m,kmin,cross,est,Λg[:,i],Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

        else #no LOCO
#            Xnul_t=Xnul*Tg';
           Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y,X=transForm(Tg,Y,X0,cross)
                   else
                   Y,X=transForm(Tg,Y,XX.X,cross)
                 end


             est0=nulScan(init,kmin,Λg,Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
        for i=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,m,kmin,cross,est0,Λg,Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
        end

        end #LOCO
    return LODs,est0
end


"""

    gene2Scan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
               Y::Array{Float64,2},XX::Markers,LOCO::Bool=false,penalize::Bool=false;m=size(Y,1),Z=diagm(ones(m)),
               Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
               itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    gene2Scan(Tg,Λg,Y,XX,cross,LOCO,penalize;Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
                  Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)



Implement 2d-genome scan with/without LOCO (Leave One Chromosome Out).  The first function is a FlxQTL model with/without `Z` and with the option of penalization that 
preestimate a null variance component matrix (``V_C``) under H0 : no QTL, followed by its adjustment by a scalar parameter under H1 : existing QTL.  
The second type of `gene2Scan()` is fitted by the conventional MLMM that estimate all parameters under `H0/H1` with the option of penalization.  
The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), Kg \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `Kg` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is pre-estimated under the null model (`H0`) of no QTL from the conventional MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  

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

- `penalize` : Boolean. Default is `false` (no prior used for penalization).  For higher dimensional traits, i.e. `large m=size(Y,1)`, penalization is recommended; set `penalize=true` 
            with adjustment of `df_prior` or/and `Prior` if necessary.
- `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(1,size(Y,2))`.  Adding covariates (C) is `Xnul= vcat(ones(1,n),C)` where `size(C)=(c,n)`.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default. 
- `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` :  A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.

!!! Note
- If some LOD scores return negative values under penalization or no penalization, you may reduce tolerences for ECM to e.g., `tol0 = 1e-4` (no penalization), 
  or switch to penalization (`penalize=true`), follwed by adjusting `df_prior`, such that 
   ``m+1 \\le`` `df_prior` ``< 2m`` to avoid singluarity errors.  The last resort could be `df_prior = Int64(ceil(1.9m))` unless any of them would work.  
   Adjusting `df_prior` is more effective than doing `Prior`; we do not recommend this adjustment for lower dimensional traits (``m < 15 \\sim 20``), 
    depending on the data since this may slow the performance.  

# Output

- `LODs` : A matrix of LOD scores. Can change to ``- \\log_{10}{P}`` using [`lod2logP`](@ref).
- `H0est` : A type of `EcmNestrv.Result` including parameter estimates under H0: no QTL.

"""
function gene2Scan(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;penalize::Bool=false,m=size(Y,1),Z=diagm(ones(m)),
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
          itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
    
     if (!penalize)

        LODs, H0est = gene2Scan(Tg,Λg,Y,XX,LOCO,cross;m=m,Z=Z,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)

     else 
        LODs, H0est = geneScan2(cross,Tg,Λg,Y,XX,LOCO;m=m,Z=Z,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ,df_prior=df_prior,Prior=Prior)
          
     end

     return LODs, H0est

end

#MVLMM
function gene2Scan(Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},Y::Array{Float64,2},XX::Markers,cross::Int64,LOCO::Bool=false;penalize::Bool=false,m=size(Y,1),
                   Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
                  Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    if (penalize)

        LODs, H0est = geneScan2(Tg,Λg,Y,XX,cross,LOCO;m=m,Xnul=Xnul,df_prior=df_prior,Prior=Prior,itol=itol,tol0=tol0,tol=tol,ρ=ρ)
    else
        LODs, H0est = geneScan2(Tg,Λg,Y,XX,LOCO,cross;Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)
    end

    return LODs, H0est
end