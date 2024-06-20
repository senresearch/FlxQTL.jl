# export  K2eig, K2Eig, transForm

# K2eig , K2Eig : functions to decompose kinship (genetic :Kg, climate:Kc) matrices into
#corresponding eigenvectors(orthogonal matrices) and eigen values. K2Eig is a function that returns eigenvectors&values for two kinship matrices
#  See a shrinkgLoco kinship.jl
# Synopsis: (T,λ)=K2eig(K); (T,λ)=K2eig(Kc,true), (Tg,Tc,λg,λc)=K2Eig(Kg,Kc),(Tg,Tc,λg,λc)=K2Eig(Kg,Kc,true)
#Input:
# Kg (or K) : a 2-d or 3-d(LOCO=true) array genetic kinship matrix. dim(Kg)=(n,n) or dim(Kg)=(n,n,nChr)
# Kc(or K) : a environment related matrix using climate factors. dim(Kc)=(m,m)
# LOCO : boolean. dafault is false, i.e. no LOCO scheme used. When true, it compute 3-d array of orthogonal matrices
#Output:
# Tg (or T) : an orthogonal matrix (or 3-d array) by eigen decomposition to Kg
# Tc : an orthogonal matrix by eigen decompostion to Kc
# λg (or λ) : an eigenvalue vector from Kg
# λc : an eigenvalue vector from Kc

"""

       K2eig(K,LOCO::Bool=false)

Returns eigenvectors and eigenvalues of a (genetic, climatic) relatedness, or 3-d array of these of a genetic relatedness if `LOCO` is `true`.

# Arguments

- `K` : A matrix of (genetic or climatic) relatedness (Default).  3-d array of genetic relatedness (`LOCO` sets to be true.)
- `LOCO` : Boolean. Default is `false` (no LOCO). (Leave One Chromosome Out).

# Output

- `T` : A matrix of eigenvectors, or 3-d array of eigenvectors if `LOCO` sets to be `true`.
- `λ` : A vector of eigenvalues, or matrix of eigenvalues if `LOCO` sets to be `true`.


See also [`K2Eig`](@ref).

# Examples

For a (climatic) relatedness, or genetic relatedness for `LOCO =false`,
```
 T, λ = K2eig(K)

```
produces a matrix of `T` and a vector of `λ`.

For a genetic kinship calculated under `LOCO` (3-d array of kinship),
```
 T, λ = K2eig(K,true)

```
produces a 3-d array of `T` and a matrix of `λ`.

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


"""

      K2Eig(Kg,Kc::Array{Float64,2},LOCO::Bool=false)


Returns a two pairs of eigenvectors and eigenvalues for genetic and climatic relatedness matrices.

# Arguments

- `Kg` : A matrix of a genetic kinship, or 3-d array of that if `LOCO` sets to be `true`.
- `Kc` : A matrix of a climatic relatedness.
- `LOCO` : Boolean. Default is `false` (no LOCO). (Leave One Chromosome Out). `LOCO` is only connected to the genetic kinship (`Kg`).

# Output

- `Tg` : A matrix of eigenvectors for `Kg`, or 3-d array of eigenvectors if `LOCO` sets to be `true`.
- `λg` : A vector of eigenvalues for `Kg`, or matrix of eigenvalues if `LOCO` sets to be `true`.
- `Tc` : A matrix of eigenvectors for `Kc`.
- `λc` : A vector of eigenvalues for `Kc`

See [`K2eig`](@ref).

# Examples

For a genetic kinship calculated under `LOCO` (3-d array of kinship),

```
 Tg,λg,Tc,λc = K2Eig(Kg,Kc,true)

```
produces a 3-d array of `Tg`, matrices of `λg`, `Tc`, and a vector of `λc`.

"""
function K2Eig(Kg,Kc::Array{Float64,2},LOCO::Bool=false)
    Tc, λc=K2eig(Kc)
    if (LOCO)
        Tg, λg =K2eig(Kg,true)
        else #no loco
        Tg, λg=K2eig(Kg)
    end
    return Tg,λg,Tc,λc
end




# transForm: a function to rotate data to columnwise and rowwise
# Input:
# Tg, Tc : orthogonal matrices for rotating data column-wise(individual wise) and row-wise(variable wise), respectively
# Y0 (m x n), X0(p x n), Z0 (m x q) : raw data : phenotypes, genotypes, row-covariates, respectively
# Σ_0 : an initial variance-covariance matrix obtained from the multivarLm module.
#Output:
# Y,X,Z,Σ : transformed matrices
# See also: K2eig, K2Eig

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

        else #cross>1 size(X0)= (p,cross,n)
          p=size(X0,1); n=size(X0,3)
            X=zeros(p,cross,n)
       @fastmath @inbounds @views for j=1:cross
#             X[:,j,:] = X0[:,j,:]*Tg'
             X[:,j,:]= BLAS.gemm('N','T',X0[:,j,:],Tg)
                     end
    end
    
    return X
    
end






##initialize parameters(B,τ2 (or Vc),Σ) (H0:no qtl case)
## Xnul=ones(1,n) or Xnul=vcat(ones(1,n), covariates), where size(covariates,2)=n
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

#initialize parameters after computing Kc
struct Init1
B::Array{Float64,2}
τ2::Float64
Σ1::Array{Float64,2} #trait-wise transformed
Kc::Array{Float64,2}
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
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Z1,Σt;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc;tol=itol)
            nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)
         
    return nulpar
end

#including prior
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Z1,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
    nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)
 
return nulpar
end
#######
# pre-computed Kc included 
function nulScan(init::Init1,kmin,λg,λc,Y1,Xnul_t,Z1;ρ=0.001,itol=1e-3,tol=1e-4)
    
            B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z1,init.B,init.τ2,init.Σ1,λg,λc;tol=itol)
            nulpar=NestrvAG(kmin,Y1,Xnul_t,Z1,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)
        
    return nulpar
    
end
######

#Z=I
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Σt;ρ=0.001,itol=1e-3,tol=1e-4)

            B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc;tol=itol)
            nulpar= NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc;ρ=ρ,tol=tol)

    return nulpar
end

#including prior
function nulScan(init::Init,kmin,λg,λc,Y1,Xnul_t,Σt,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

    B0,τ2_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,init.B,init.τ2,Σt,λg,λc,ν₀,Ψ;tol=itol)
    nulpar= NestrvAG(kmin,Y1,Xnul_t,B0,τ2_0,Σ1,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol)

return nulpar
end


##########
#new version to estimate Kc
function nulScan1(init::Union{Init1,Init0},kmin,λg,Y1,Xnul_t,Z;ρ=0.001,itol=1e-3,tol=1e-4)
       
       if (typeof(init)==Init1)   
           B0,Kc_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z,init.B,init.Kc,init.Σ1,λg;tol=itol)
           nulpar=NestrvAG(kmin,Y1,Xnul_t,Z,B0,Kc_0,Σ1,λg;tol=tol,ρ=ρ)
        else #typeof(Init)==Init0)
           B0,Kc_0,Σ1,loglik0 =ecmLMM(Y1,Xnul_t,Z,init.B,init.Vc,init.Σ,λg;tol=itol)
           nulpar=NestrvAG(kmin,Y1,Xnul_t,Z,B0,Kc_0,Σ1,λg;tol=tol,ρ=ρ)
        end
       return nulpar
end
###########

#MVLMM :Z=I
function nulScan(init::Init0,kmin,λg,Y1,Xnul_t;ρ=0.001,itol=1e-3,tol=1e-4)

        B0,Vc_0,Σ1,loglik0 = ecmLMM(Y1,Xnul_t,init.B,init.Vc,init.Σ,λg;tol=itol)
        nulpar=NestrvAG(kmin,Y1,Xnul_t,B0,Vc_0,Σ1,λg;tol=tol,ρ=ρ)

       return nulpar
end

#including prior
function nulScan(init::Init0,kmin,λg,Y1,Xnul_t,ν₀,Ψ;ρ=0.001,itol=1e-3,tol=1e-4)

        B0,Vc_0,Σ1,loglik0 = ecmLMM(Y1,Xnul_t,init.B,init.Vc,init.Σ,λg,ν₀,Ψ;tol=itol)
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
