

"""

       K2eig(K,LOCO::Bool=false)

Returns eigenvectors and eigenvalues of genetic relatedness, or 3-d array of these of a genetic relatedness if `LOCO` is `true`.

# Arguments

- `K` : A matrix of genetic relatedness (Default).  3-d array of genetic relatedness (`LOCO` sets to be true.)
- `LOCO` : Boolean. Default is `false` (no LOCO). (Leave One Chromosome Out).

# Output

- `T` : A matrix of eigenvectors, or a 3-d array of eigenvectors if `LOCO` sets to be `true`.
- `λ` : A vector of eigenvalues, or a matrix of eigenvalues if `LOCO` sets to be `true`.

# Examples

For a null variance component, or genetic relatedness for `LOCO =false`,
```
 T, λ = K2eig(K)

```
produces a matrix of `T` and a vector of `λ`.

For a genetic kinship calculated under `LOCO` (a 3-d array of kinship matrices),
```
 T, λ = K2eig(K,true)

```
produces a 3-d array of matrices `T` and a matrix of `λ`.

"""
function K2eig(K,LOCO::Bool=false)
    if(LOCO)
        nChr=size(K,3);
        n=size(K,1);
        T=zeros(n,n,nChr);
        λ=zeros(n,nChr);
#        @inbounds Threads.@threads for j=1:nChr
       @inbounds for j=1:nChr
            Λ=svd(K[:,:,j]);
            T[:,:,j],λ[:,j]=Λ.Vt,Λ.S
        end
        else #no loco
        Λ=svd(K);
        T,λ=Λ.Vt,Λ.S
    end
    return T,λ
end


# """

#       K2Eig(Kg,Vc::Array{Float64,2},LOCO::Bool=false)


# Returns two pairs of eigenvectors and eigenvalues for genetic relatedness matrices.

# # Arguments

# - `Kg` : A matrix of a genetic kinship, or 3-d array of that if `LOCO` sets to be `true`.
# - `Vc` : A matrix of a variance component preestimated by the null model of no QTL.
# - `LOCO` : Boolean. Default is `false` (no LOCO). (Leave One Chromosome Out). `LOCO` is only connected to the genetic kinship (`Kg`).

# # Output

# - `Tg` : A matrix of eigenvectors for `Kg`, or 3-d array of eigenvectors if `LOCO` sets to be `true`.
# - `λg` : A vector of eigenvalues for `Kg`, or matrix of eigenvalues if `LOCO` sets to be `true`.
# - `Tc` : A matrix of eigenvectors for `Vc`.
# - `λc` : A vector of eigenvalues for `Vc`

# See [`K2eig`](@ref).

# # Examples

# For a genetic kinship calculated under `LOCO` (a 3-d array of kinship matrices),

# ```
#  Tg,λg,Tc,λc = K2Eig(Kg,Vc,true)

# ```
# produces a 3-d array of matrices `Tg`, matrices of `λg`, `Tc`, respectively, and a vector of `λc`.

# """
# function K2Eig(Kg,Vc::Array{Float64,2},LOCO::Bool=false)
#     Tc, λc=K2eig(Vc)
#     if (LOCO)
#         Tg, λg =K2eig(Kg,true)
#         else #no loco
#         Tg, λg=K2eig(Kg)
#     end
#     return Tg,λg,Tc,λc
# end




function transZ!(Z::Array{Float64,2},Tc::Array{Float64,2},Z0::Array{Float64,2})

    mul!(Z,Tc,Z0)
end

# rotate Σ₀ or Ψ₀ (prior parameter for Σ)
function transPD(Tc::Array{Float64,2},Σ₀::Array{Float64,2})

    Σ=Symmetric(BLAS.symm('R','U',Σ₀,Tc)*Tc')

    return convert(Array{Float64,2},Σ)
end


#rotate by row (trait(or site)-wise)
function transForm(Tc::Array{Float64,2},Z0::Array{Float64,2},Σ_0::Array{Float64,2},both::Bool=false)

       Z=similar(Z0)
    if (both)
#          Z=Tc*Z0
#          Σ=Symmetric((Tc*Σ_0)*Tc')
          transZ!(Z,Tc,Z0)
          Σ= transPD(Tc,Σ_0)
         return Z,Σ
     else
         return  transZ!(Z,Tc,Z0)
    end
end

function transForm(Tc::Array{Float64,2},Z0::Array{Float64,2},Σ₀::Array{Float64,2},Ψ₀::Array{Float64,2})

    if (isposdef(Ψ₀))
    Z, Σ = transForm(Tc,Z0,Σ₀,true)
    Ψ = transPD(Tc,Ψ₀)
    else 
        println("Error! Plug in a postivie definite Prior!")
    end

    return Z, Σ, Ψ
end
      
 


# rotate by column (individual-wise)
function transForm(Tg::Array{Float64,2},Y0::Array{Float64,2},X0,cross::Int64)


          Y=BLAS.gemm('N','T',Y0,Tg)
          X=transForm(Tg,X0,cross)

    return Y,X
end

function transForm(Tg::Array{Float64,2},X0,cross::Int64)
   
    if (cross==1)
          X=BLAS.gemm('N','T',X0,Tg)

        else #cross>1 size(X0)= (cross,n,p)
          p=size(X0,3); n=size(X0,2)
            X=zeros(cross,n,p)
       @fastmath @inbounds for j=1:cross
#             X[:,j,:] = X0[:,j,:]*Tg'
             X[j,:,:]= BLAS.gemm('N','N',Tg,X0[j,:,:])
                     end
    end
    
    return X
    
end


struct InitKc
    Kc::Matrix{Float64} 
    B::Matrix{Float64}
    Σ::Matrix{Float64}
    τ2::Float64
    loglik::Float64
 end
 
struct Init
B::Array{Float64,2}
τ2::Float64
Σ::Array{Float64,2}
end

struct Init0
B::Array{Float64,2}
Vc::Array{Float64,2}
Σ::Array{Float64,2}
end


# including MLMM
function initial(Xnul,Y0,Z0,incl_τ2::Bool=true)
     m=size(Y0,1);
    init_val=MLM.mGLM(convert(Array{Float64,2},Y0'),convert(Array{Float64,2},Xnul'),Z0)
         
       if (incl_τ2)
#         Σ0= init_val.Σ*sqrt(1/m);  τ2 =mean(Diagonal(Σ0));
          lmul!(sqrt(1/m),init_val.Σ)
          τ2 =mean(Diagonal(init_val.Σ))

            return Init(init_val.B',τ2,init_val.Σ)
        else #H0 for MLMM
        
        lmul!(sqrt(1/m),init_val.Σ)
        return Init0(init_val.B',0.5*init_val.Σ,0.5*init_val.Σ)
        end
        
end

#Z=I (including MLMM)
function initial(Xnul,Y0,incl_τ2::Bool=true)
     m=size(Y0,1);
    init_val=MLM.mGLM(convert(Array{Float64,2},Y0'),convert(Array{Float64,2},Xnul'))

      if (incl_τ2)
        lmul!(sqrt(1/m),init_val.Σ)
        τ2 =mean(Diagonal(init_val.Σ))

          return Init(init_val.B',τ2,init_val.Σ)
        else
#             c=rand(1)[1]
#            Vc=c*init_val.Σ;    Σ0=(1-c)*init_val.Σ
        lmul!(sqrt(1/m),init_val.Σ)
          return Init0(init_val.B',0.5*init_val.Σ,0.5*init_val.Σ)
       end

end

## nulScan : scan w/o qtl to obtain null parameter estimates
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Z1,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
            nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)
         
    return nulpar
end

#including prior
function nulScan(init::InitKc,kmin,λg,λc,Y1,Xnul_t,Z1,Σt,ν₀,Ψ,H0_up::Bool;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
    nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)
    
 if (H0_up) # null model update for high dimensional traits
    return nulpar
 else
    return Approx(nulpar.B,nulpar.τ2,nulpar.Σ,init.loglik)
 end

end
#Z=I:including prior
function nulScan(init::InitKc,kmin,λg,λc,Y1,Xnul_t,Σt,ν₀,Ψ,H0_up::Bool;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
    nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)
 
 if (H0_up) # null model update for high dimensional traits
    return nulpar
  else
    return Approx(nulpar.B,nulpar.τ2,nulpar.Σ,init.loglik)
  end

end


#Z=I
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
            nulpar= NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)

    return nulpar
end




#MVLMM :Z=I
# function nulScan(init::Init0,kmin,λg,Y1,Xnul_t;ρ=0.001,itol=1e-3,tol=1e-4)

#         B0,Vc_0,Σ1,_ = ecmLMM(Y1,Xnul_t,init.B,init.Vc,init.Σ,λg;tol=itol)
#         nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,Vc_0,Σ1,λg;tol=tol,ρ=ρ)

#        return nulpar
# end

#including prior
function nulScan(init::Init0,kmin,λg,Y1,Xnul_t,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

        B0,Vc_0,Σ1,_= ecmLMM(Y1,Xnul_t,init.B,init.Vc,init.Σ,λg,ν₀,Ψ;tol=itol)
        nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,Vc_0,Σ1,λg,ν₀,Ψ;tol=tol,ρ=ρ)

       return nulpar
end


##rearrange Bs estimated under H1 into 3-d array
## dimensions are from Z & X, size(Z)=(m,q), size(X)=(p,n)
## p1= size(Xnul,1) : Xnul may or may not include covariates. default is ones(1,n)
## H1par : 1-d array including parameter estimates under H1 enclosed by EcmNestrv.Approx
function arrngB(H1par,p1::Int64,q,p,cross)

     if (cross!=1)
        B=zeros(q,cross-1+p1,p)
     else
        B=zeros(q,cross+p1,p)
     end

       @inbounds @views for j=1:length(H1par)
            B[:,:,j]=H1par[j].B
        end
    return B
end

#  """

#     getKc(Y::Array{Float64,2};m=size(Y,1),Z=diagm(ones(m)), df_prior=m+1,
#            Prior::Matrix{Float64}=cov(Y,dims=2)*5,Xnul::Array{Float64,2}=ones(1,size(Y,2)),
#            itol=1e-2,tol::Float64=1e-3,ρ=0.001)

# Pre-estimate `Kc` by regressing `Y` on `Xnul`, i.e. estimating environmental covariates under `H0: no QTL`.

# # Argument

# - `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
#         i.e. `Y0[1,:]` (a vector) ->`Y[[1],:]` (a matrix) .

# ## Keyword Arguments

# - `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
#         An identity matrix, ``I_m``, is default. 
# - `Xnul` :  A matrix of covariates. Default is intercepts (1's): `Xnul= ones(1,size(Y0))`.  Adding covariates (C) is `Xnul= vcat(ones(1,m),C)` where `size(C)=(c,m)` for `m = size(Y0,1)`.
# - `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
#            A large scaled covariance matrix (a weakly informative prior) is default.
# - `df_prior`: degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
# - `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
# - `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
# - `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.

# # Output

# - `InitKc` :  A type of struct of arrays, including pre-estimated `Kc`,`and null estimates of B`, `Σ`,`τ2`used as initial values inside 
#      `gene1Scan`, one of [`geneScan`](@ref) functions, or [`gene2Scan`](@ref).

# # Examples

# ```
# julia> K0 = getKc(Y)  
# julia> K0.Kc  # for Kc
# julia> K0.B # for B under H0

# ```

# """
#  function getKc(Y::Array{Float64,2};m=size(Y,1),Z=diagm(ones(m)), df_prior=m+1,
#      Prior::Matrix{Float64}=cov(Y,dims=2)*3,
#      Xnul::Array{Float64,2}=ones(1,size(Y,2)),itol=1e-2,tol::Float64=1e-3,ρ=0.001)
     
#      if(Z!=diagm(ones(m)))
#          init0=initial(Xnul,Y,Z,false)
#       else #Z0=I
#          init0=initial(Xnul,Y,false)
#       end
 
#      est0= nul1Scan(init0,1,Y,Xnul,Z,m,df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
#        τ² =mean(Diagonal(est0.Vc)./m)
#      return InitKc(est0.Vc, est0.B, est0.Σ, τ²,est0.loglik)
 
#  end
 
# ###########
# #estimate Kc with prior
# function nul1Scan(init::Init0,kmin,Y,Xnul,Z,m,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)
       
#       n=size(Y,2); λg=ones(n)

#     if (Z!=diagm(ones(m)))   
#         B0,Kc_0,Σ1,_ =ecmLMM(Y,Xnul,Z,init.B,init.Vc,init.Σ,λg,ν₀,Ψ;tol=itol)
#         nulpar=NestrvAG(kmin,Y,Xnul,Z,B0,Kc_0,Σ1,λg,ν₀,Ψ;tol=tol,ρ=ρ)
        
#        else #Z=I
#         nulpar = nulScan(init,kmin,λg,Y,Xnul,ν₀,Ψ;ρ=ρ,itol=itol,tol=tol)
#      end
#     return nulpar
# end
