"""

      MLM 

A module for fitting general Multivariate Linear Models motivated by functional data analysis via mle or reml.
The default fitting method is mle. ( i.e. reml=false)

The model: 
``Y=XBZ'+E``, where ``E(Y)=XBZ'`` (or ``E(vec(Y))= (Z \\otimes X)vec(B)`` ),  ``var(vec(E))=\\Sigma \\otimes I.``
size(Y)=(n,m), size(X)=(n,p), size(Z)=(m,q).

"""
module MLM


using LinearAlgebra

struct EstZ
     pX::Array{Float64,2}
     Σ::Array{Float64,2}
     B::Array{Float64,2}
     loglik::Float64
end


# mGLMind :: fitting a multivariate linear model (initial computation)
# Synopsis: EstZ = mGLMInd(Y,X,Z,true)

# Input:
# Y: response variables (or phenotypes) 
# X: row  covariate fixed effects(independent variables or genotypes)
# Z: column covariate fixed effects (or basis functions or contrasts)
# reml:restricted mle (the default is mle )
# Output:
# EstZ.pX : computing inv(X'X)X'
# EstZ.Σ : parameter estimation of Σ
# EstZ.B : parameter estimation of B (or β)

# Example:
# See also mGLM




function mGLMind(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},reml::Bool=false)

    #number of individuals; number of covariates
    n,m=size(Y);
    p=size(X,2)
     q=size(Z,2)

    F=qr(X)
  
    Xquad=Symmetric(F.R'*F.R)  # Force the inverse of  (X'X =R'R) to be symmetric explicitly
    pX= Xquad\X'  #  inv(R'R)*X'
   
    pZ=Symmetric(BLAS.syrk('U','T',1.0,Z))\Z' # inv(Z'Z)*Z'
    B=BLAS.gemm('N','T',(pX*Y),pZ)
    
    # mle for B and Σ=I 
    Ŷ=BLAS.gemm('N','T',(X*B),Z)  # Yhat= XBZ'
    ESS=Symmetric(BLAS.syrk('U','T',1.0,(Y-Ŷ)))
if (reml) 
    Σ=ESS/(n-p)  
else
   Σ=ESS/n
end
   
     loglik=-0.5*(n*(m*log(2π)+logdet(Σ))+tr(ESS))
    if (reml)
        loglik += 0.5*p*q*log(2π)
    end
    
    return EstZ(pX,Σ,B,loglik)

end


"""

  mGLM(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},reml::Bool=false)
  mGLM(Y::Array{Float64,2},X::Array{Float64,2},reml::Bool=false)

Fitting multivariate General Linear Models via MLE (or REML) and returns a type of a struct `Estimat`.  


# Arguments

- `Y` : A matrix of response variables, i.e. traits. size(Y)=(n,m) for n individuals x m traits
- `X` : A matrix of independent variables, i.e. genotypes or genotype probabilities including intercept or/and covariates. size(X)=(n,p) for n individuals x p markers 
      including intercept or/and covariates 
- `Z` : An optional matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.). 
      If nothing to insert in `Z`, just exclude it or insert `Matrix(1.0I,m,m) `. size(Z)=(m,q) for m traits x q phenotypic covariates.
- `reml` : Boolean. Default is fitting the model via mle. Resitricted MLE is implemented if `true`. 


# Output

Returns [`Estimat`](@ref) .

"""
function mGLM(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},reml::Bool=false)
    #number of individuals; number of traits
    n,m=size(Y)
    #number of covariates in X, rank(X)=p
    p=size(X,2)
    #number of secondary (e.g. environment factors, grouping, etc.) variables in Z: rank(Z)=q
    q=size(Z,2)
    
    estInd=mGLMInd(Y,X,Z,reml)
#     Σ_0=estInd.Σ
#     pX=estInd.pX
    

     #transformed by Σ^(-1/2)    
    sqrtsig=sqrt(estInd.Σ)
 
    Y_t=(sqrtsig\Y')'
    Z_t=sqrtsig\Z
    
#     pZ= convert(Array{Float64,2},Symmetric(BLAS.gemm('T','N',Z,Σ\Z))\(Σ\Z)')
  
    pZ_t=Symmetric(BLAS.syrk('U','T',1.0,Z_t))\Z_t' # (Z'Σ^(-1)Z)^(-1)Z'Σ^(-1/2)
    
    B=BLAS.gemm('N','T',(estInd.pX*Y_t),pZ_t) 
    
    Ŷ_t= BLAS.gemm('N','T',(X*B),Z_t)
   
    ESS_t= Symmetric(BLAS.syrk('U','T',1.0,(Y_t-Ŷ_t)))# (Y_tran-Yhat_tran)'(Y_tran-Yhat_tran)
    
    if (reml)
        Σ_t = ESS_t/(n-p)
    else
        Σ_t = ESS_t/n
    end
    
    ## estimate loglikelihood 
    
    lg_mult=-0.5*(n*(m*log(2π)+logdet(Σ_t))+tr(ESS_t))
    if (reml)
        lg_mult += 0.5*p*q*log(2π)
    end
    
    ## Variance-Covariance matrix
    # Σ=A_mul_Bt(Σ_0,Σ_tran)
    # Σ_s=(Σ+Σ')/2

    return Estimat(B,estInd.Σ,lg_mult)
            
end




function mGLM(Y::Array{Float64,2},X::Array{Float64,2},reml::Bool=false)

    #number of individuals; number of covariates
    n,m=size(Y);
    p=size(X,2)
   

    F=qr(X)
  
    Xquad=Symmetric(F.R'*F.R)  # Force the inverse of  (X'X =R'R) to be symmetric explicitly
    pX= Xquad\X'  #  inv(R'R)*X'
   
    B=BLAS.gemm('N','N',pX,Y)
    
    # mle for B and Σ=I 
    Ŷ=BLAS.gemm('N','N',X,B)  # Yhat= XB
    ESS=Symmetric(BLAS.syrk('U','T',1.0,(Y-Ŷ)))
if (reml) 
    Σ=ESS/(n-p)  
else
   Σ=ESS/n
end
    
    loglik=-0.5*(n*(m*log(2π)+logdet(Σ))+tr(ESS))
    if (reml)
        loglik += 0.5*p*m*log(2π)
    end
    
    return Estimat(Σ,B,loglik)
end

"""
  
     Estimat(B::Array{Float64,2},Σ::Array{Float64,2},loglik::Float64)

A struct of arrays for results by fitting a multivariate linear model,  `mGLM()`.  
The results are `B`(fixed effects), `Σ` (m x m covariance matrix), `loglik`(a value of log-likelihood by mle or reml).

"""
struct Estimat
    B::Array{Float64,2}
    Σ::Array{Float64,2}
#    Σ_t::Array{Float64,2}
    loglik::Float64
end
    





# export mGLMind, EstI, EstZ, mGLM, Estimate




end