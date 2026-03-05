

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

## nulScan : scan w/o qtl to obtain null parameter estimates after getKc 
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Z1,Σt;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc;tol=itol)
            nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)
         
    return nulpar
end

#including prior
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Z1,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
            nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)
         
    return nulpar
end

## no prior with estimated Kc
function nulScan(init::InitKc,kmin,λg,λc,Y1,Xnul_t,Z1,Σt,H0_up::Bool;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc;tol=itol)
    nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)  #type: Approx
    
 if (H0_up) # null model update for high dimensional traits
    return nulpar
 else
    return Approx(nulpar.B,nulpar.τ2,nulpar.Σ,init.loglik) # replacing loglik only
 end

end
#Z=I
function nulScan(init::InitKc,kmin,λg,λc,Y1,Xnul_t,Σt,H0_up::Bool;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc;tol=itol)
    nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)
 
 if (H0_up) # null model update for high dimensional traits
    return nulpar
  else
    return Approx(nulpar.B,nulpar.τ2,nulpar.Σ,init.loglik) # replacing loglik only
  end

end

#including prior with estimated Kc
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
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Σt;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc;tol=itol)
            nulpar= NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)

    return nulpar
end

#including prior
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,_ =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
            nulpar= NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)

    return nulpar
end




#MVLMM :Z=I
function nulScan(init::Init0,kmin,λg,Y1,Xnul_t;ρ=0.001,itol=1e-3,tol=1e-4)

        B0,Vc_0,Σ1,_ = ecmLMM(Y1,Xnul_t,init.B,init.Vc,init.Σ,λg;tol=itol)
        nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,Vc_0,Σ1,λg;tol=tol,ρ=ρ)

       return nulpar
end

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

