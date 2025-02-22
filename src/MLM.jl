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
using ..EcmNestrv:symSq, fixZ


#compute Σ⁻¹Z(Z'Σ⁻¹Z)⁻¹
function pinvZ(Z::Matrix{Float64},Σ::Matrix{Float64})
             
    return  Σ\fixZ(Z,Σ)'

end

#compute (X'X)⁻¹X'
function fiX(X::Matrix{Float64})

    return symSq(X,'T')\X'
    
end


struct Estz
     B0::Array{Float64,2}
     Σ::Array{Float64,2}
end


function mGLMind(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},n,m,p,q,reml::Bool=false)

    #number of individuals; number of covariates
    # n,m=size(Y);p=size(X,2); q=size(Z,2)
    
    B = zeros(p,q); ESS=zeros(m,m);B0=zeros(p,m)
    # F=qr(X)
  
    # Xquad=Symmetric(F.R'*F.R)  # Force the inverse of  (X'X =R'R) to be symmetric explicitly
    # pX= Xquad\X'  #  inv(R'R)*X'
    
    pivZ = fiX(Z) # inv(Z'Z)*Z'
   
    # pZ=Symmetric(BLAS.syrk('U','T',1.0,Z))\Z' # inv(Z'Z)*Z'
    # B=BLAS.gemm('N','T',(pX*Y),pivZ)
        
    # mle for B and Σ=I 
    # Ŷ=BLAS.gemm('N','T',(X*B),Z)  # Yhat= XBZ'
    # ESS=Symmetric(BLAS.syrk('U','T',1.0,(Y-Ŷ)))
    MLE!(B,ESS,B0,Y,X,Z,pivZ)

   if(reml) 
        Σ=ESS./(n-p)  
    else
        Σ=ESS./n
   end
        
    return Estz(B0,Σ)

end

#Z=I
function mGLMind(Y::Array{Float64,2},X::Array{Float64,2},n,m,p,reml::Bool=false)
 
    B = zeros(p,m); ESS=zeros(m,m);
    
    MLE!(B,ESS,pXY,Y,X)

   if(reml) 
        Σ=ESS./(n-p)  
    else
        Σ=ESS./n
   end
        
    return Estz(B,Σ)

end


function MLE!(B::Matrix{Float64},ESS::Matrix{Float64},pXY::Matrix{Float64},Y::Matrix{Float64},X::Matrix{Float64},Z::Matrix{Float64},pivZ::Matrix{Float64})

    pXY[:,:]= BLAS.gemm('N','N',fiX(X),Y)
    B[:,:] = BLAS.gemm('N','T',pXY,pivZ)
    dev= Y- BLAS.gemm('N','T',(X*B),Z)
    ESS[:,:]= Symmetric(BLAS.syrk('U','T',1.0,dev))

end

#Z=I
function MLE!(B::Matrix{Float64},ESS::Matrix{Float64},Y::Matrix{Float64},X::Matrix{Float64})

    B[:,:]= BLAS.gemm('N','N',fiX(X),Y)
    dev= Y- BLAS.gemm('N','N',X,B)
    ESS[:,:]= Symmetric(BLAS.syrk('U','T',1.0,dev))

end

function MLEs!(B::Matrix{Float64},ESS::Matrix{Float64},Y::Matrix{Float64},X::Matrix{Float64},Z::Matrix{Float64},pivZ::Matrix{Float64},pXY::Matrix{Float64})
   B[:,:] = BLAS.gemm('N','T',pXY,pivZ)
   dev = Y- BLAS.gemm('N','T',(X*B),Z)
   ESS[:,:] = Symmetric(BLAS.syrk('U','T',1.0,dev)) 

end


#Z=I
function MLEs!(ESS::Matrix{Float64},Y::Matrix{Float64},X::Matrix{Float64},pXY::Matrix{Float64})
#    B[:,:] = BLAS.gemm('N','T',pXY,pivZ)
   dev = Y- BLAS.gemm('N','N',X,pXY)
   ESS[:,:] = Symmetric(BLAS.syrk('U','T',1.0,dev)) 

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
    B = zeros(p,q); ESS=zeros(m,m);
    # estInd=mGLMind(Y,X,Z,reml)
#     Σ_0=estInd.Σ
#     pX=estInd.pX
    

     #transformed by Σ^(-1/2)    
    # sqrtsig=sqrt(estInd.Σ)
 
    # Y_t=(sqrtsig\Y')'
    # Z_t=sqrtsig\Z
    Est0= mGLMind(Y,X,Z,n,m,p,q,reml)
    pivZ= pinvZ(Z,Est0.Σ)
    MLEs!(B,ESS,Y,X,Z,pivZ,Est0.B0)
    # pZ_t=Symmetric(BLAS.syrk('U','T',1.0,Z_t))\Z_t' # (Z'Σ^(-1)Z)^(-1)Z'Σ^(-1/2)
    # B=BLAS.gemm('N','T',(estInd.pX*Y_t),pZ_t) 
    # Ŷ_t= BLAS.gemm('N','T',(X*B),Z_t)
    # ESS_t= Symmetric(BLAS.syrk('U','T',1.0,(Y_t-Ŷ_t)))# (Y_tran-Yhat_tran)'(Y_tran-Yhat_tran)
    
    if(reml)
        Σ = ESS/(n-p)
     else
        Σ = ESS/n
    end
    
    ## estimate loglikelihood 
    
    loglik=-0.5*(n*(m*log(2π)+logdet(Σ))+tr(ESS))
    if(reml)
        loglik += 0.5*p*q*log(2π)
    end
    
    return Estimat(B,Σ,loglik)
            
end



#Z=I
function mGLM(Y::Array{Float64,2},X::Array{Float64,2},reml::Bool=false)

    #number of individuals; number of covariates
    n,m=size(Y);
    p=size(X,2)
   
  Est0 = mGLMind(Y,X,n,m,p,reml)
  MLEs!(ESS,Y,X,Est0.B0)
   if(reml) 
      Σ=ESS/(n-p)  
     else
      Σ=ESS/n
    end
    
    loglik=-0.5*(n*(m*log(2π)+logdet(Σ))+tr(ESS))
    if(reml)
        loglik += 0.5*p*m*log(2π)
    end
    
    return Estimat(Est0.B0,Σ,loglik)
end

"""
  
     Estimat(B::Array{Float64,2},Σ::Array{Float64,2},loglik::Float64)

A struct of arrays for results by fitting a multivariate linear model,  `mGLM()`.  
The results are `B`(fixed effects), `Σ` (m x m covariance matrix), `loglik`(a value of log-likelihood by mle or reml).

"""
struct Estimat
    B::Array{Float64,2}
    Σ::Array{Float64,2}
    loglik::Float64
end
    










end