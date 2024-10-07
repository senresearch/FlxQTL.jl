

###Permutation test : all permutations are implemented without loco

#trans2iid : Transforming columnwise (individuals) correlated Y1 to iid Y_t, or transforming back to the correlated Y1
### Use a multivariate phenotype matrix to permute data and run an Ecm+Nesterov algorithm per phenotype to obtain
### the sum of lod's for univariate phenotypes.
##Input:
# Y : a matrix of phenotypes already transformed by orthogonal matrices corresponding eigenvalues λg,λc
# τ2_nul,Σ_nul : estimates from the multivariate model under the null.
## trnsback : default = false, i.e. perform an iid transformation, and if true, transforming back from the iid Y
function trans2iid(Y::Array{Float64,2},τ2_nul::Float64,Σ_nul::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1}
        ,trnsback::Bool=false)

    Y_t=similar(Y);
   # Std=Array{Float64}(undef,m,m)
    Λc=Diagonal(τ2_nul*λc)

    @fastmath @inbounds @views for j in eachindex(λg)
        Std=sqrt(λg[j]*Λc+Σ_nul)
        if (trnsback)
            Y_t[:,j]=Std*Y[:,j]
        else
            Y_t[:,j]=Std\Y[:,j]
        end
    end
    return Y_t
end
##MVLMM
function trans2iid(Y::Array{Float64,2},Vc_nul::Array{Float64,2},Σ_nul::Array{Float64,2},λg::Array{Float64,1},trnsback::Bool=false)

    Y_t=similar(Y);
   # Std=Array{Float64}(undef,m,m)
   @fastmath @inbounds @views for j in eachindex(λg)
         Std=sqrt(Vc_nul*λg[j]+Σ_nul)
        if (trnsback)
            Y_t[:,j]=Std*Y[:,j]
        else
            Y_t[:,j]=Std\Y[:,j]
        end
    end
    return Y_t
end

## permuteY : shuffling a Y matrix columnwise (by individuals)
function permutY(Y::Array{Float64,2},τ2_nul::Float64,Σ_nul::Array{Float64,2}
        ,λg::Array{Float64,1},λc::Array{Float64,1})
   n=size(Y,2);
    ### permutation by individuals
    #transforming to iid Y1
    Y_t=trans2iid(Y,τ2_nul,Σ_nul,λg,λc);
    rng=shuffle(1:n);
    # transforming back to correlated Y1 after permuting
    Y2=trans2iid(Y_t[:,rng],τ2_nul,Σ_nul,λg,λc,true);
    return Y2
end

#MVLMM
function permutY(Y::Array{Float64,2},Vc_nul::Array{Float64,2},Σ_nul::Array{Float64,2},λg::Array{Float64,1})
    n=size(Y,2);
    ### permutation by individuals
    #transforming to iid Y1
    Y_t=trans2iid(Y,Vc_nul,Σ_nul,λg);
    rng=shuffle(1:n);
    # transforming back to correlated Y1 after permuting
    Y2=trans2iid(Y_t[:,rng],Vc_nul,Σ_nul,λg,true);
    return Y2
end



## permutation: finding distribution of max lod's and B's for all markers under H1 for a multivariate model
##by permutation (no LOCO)
## Input : Y,X,Z are all transformed
## Output: lod : a column vector of max lod's
##         H1par : a matrix of parameter estimates by permutation.
##              %%%   rows: # of markers(B and loglik), columns: # of permutations

## finding distribution of max lod's for a multivariate model by permutation for 4waycross/intercross
function permutation(nperm::Int64,cross::Int64,Y::Array{Float64,2},X::Union{Array{Float64,2},Array{Float64,3}},
        Z::Array{Float64,2},Nullpar::Approx,λg::Array{Float64,1},λc::Array{Float64,1},Xnul_t::Array{Float64,2},ν₀,Ψ,ρ,τ₀
        ;tol0=1e-3,tol::Float64=1e-4)

#     n=size(Y,2); p=size(X,1);
    q=size(Z,2);
    kmin=1; lod=zeros(nperm);H1par=[]

    init=Init(Nullpar.B,Nullpar.τ2,Nullpar.Σ)
    
    for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        Y2=permutY(Y,init.τ2,init.Σ,λg,λc);
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(init,kmin,λg,λc,Y2,Xnul_t,Z,Nullpar.Σ,ν₀,Ψ,ρ,τ₀;itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(q,kmin,cross,perm_est0,λg,λc,Y2,Xnul_t,X,Z,ν₀,Ψ,ρ,τ₀;tol0=tol0,tol1=tol)
               
          lod[l]= maximum(LODs);  H1par=[H1par; H1par0]
            if (mod(l,50)==0)
              println("Scan for $(l)th permutation is done.")
            end
        end

    return lod, H1par
end


#MVLMM
function permutation(nperm::Int64,cross::Int64,Y::Array{Float64,2},X::Union{Array{Float64,2},Array{Float64,3}},
        Nullpar::Result,λg::Array{Float64,1},Xnul_t,ν₀,Ψ,ν,Ψ₀;tol0=1e-3,tol::Float64=1e-4)

     m=size(Y,1);
    kmin=1;lod=zeros(nperm);H1par=[]
   
    init=Init0(Nullpar.B,Nullpar.Vc,Nullpar.Σ)
    
     for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        Y2=permutY(Y,Nullpar.Vc,Nullpar.Σ,λg);

        #initial parameter values for permutations are from genome scanning under the null hypothesis.
         perm_est0=nulScan(init,kmin,λg,Y2,Xnul_t,ν₀,Ψ,ν,Ψ₀;itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(m,kmin,cross,perm_est0,λg,Y2,Xnul_t,X,ν₀,Ψ,ν,Ψ₀;tol0=tol0,tol1=tol)
    
         lod[l]= maximum(LODs);  H1par=[H1par; H1par0]
             if (mod(l,50)==0)
              println("Scan for $(l)th permutation is done.")
             end
        end


    return lod, H1par
end


"""

      permTest(nperm::Int64,cross,Kg,Kc,Y,XX::Markers,Z;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
             Prior::Matrix{Float64}=cov(Y,dims=2),df_prior_τ2=1,τ2_Pr::Float64=1.0,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4)
      permTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2),df_Rprior=m+1,Rprior=diagm(ones(m)),Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4)
   
Implement permutation test to get thresholds at the levels of type 1 error, `α`.  Note that the last `permTest()` 
is for the conventional MLMM: 
```math
vec(Y)\\sim MVN((I \\otimes X)vec(B) (or BX), K \\otimes \\Sigma_1 +I \\otimes \\Sigma_2),
``` 
where `K` is a genetic kinship, ``\\Sigma_1, \\Sigma_2`` are covariance matrices for random and error terms, respectively.

# Arguments

- `nperm` : An integer indicating the number of permutation to be implemented.
- `cross` : An integer indicating the number of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Kg` : A n x n genetic kinship matrix. Should be symmetric positive definite.
- `Kc` : A m x m climatic relatedness matrix. Should be symmetric positive definite.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z` : An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.). 
      If nothing to insert in `Z`, just insert an identity matrix, `Matrix(1.0I,m,m)`.  `m` traits x `q` phenotypic covariates. 

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of Inverse-Wishart prior distributon for the residual error matrix, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           ``cov(Y,dims=2)`` (empirical scale matrix) is default.
- `df_prior`: degrees of freedom, ``\\nu_0`` of Inverse-Wishart prior distributon for the residual error matrix.  `m+1` (non-informative) is default.
- `df_prior_τ2`: degree of freedom, ``\\rho`` of scaled Inverse-``\\Chi^2`` prior distribution for ``\\tau^2``. `1` is default.
- `τ2_Pr`: a positive scaled parameter of scaled Inverse-``\\Chi^2`` prior distribution for ``\\tau^2``, i.e., ``\\tau^2 \\sim Scale-inv \\Chi^2(\\rho, \\tau_0)``. ``1.0`` is default.           
- `Rprior`: A positive definite scale matrix, ``\\Psi_0``, of Inverse-Wishart prior distribution for the random effect matrix, i.e. ``\\Sigma_1 \\sim W^{-1}_m (\\Psi_0, \\nu)``.  
           ``I_m`` (for non-informative prior) is default.
- `df_Rprior`: degrees of freedom, ``\\nu`` of Inverse-Wishart prior distributon for \\Sigma_1.  `m+1` (non-informative) is default.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.

# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A type of struct, `EcmNestrv.Approx(B,τ2,Σ,loglik)` including parameter estimates  or `EcmNestrv.Result(B,Vc,Σ,loglik)` 
                for a conventional MLMM under H0: no QTL by permutation. 
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function permTest(nperm::Int64,cross,Kg,Kc,Y,XX::Markers,Z;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
        Prior::Matrix{Float64}=cov(Y,dims=2),Xnul=ones(1,size(Y,2)),df_prior_τ2=1,τ2_Pr::Float64=1.0,itol=1e-4,tol0=1e-3,tol=1e-4)
    #permutation without LOCO
       Tg,λg,Tc,λc=K2Eig(Kg,Kc)
       est0,Xnul_t,Y1,X1,Z1 = geneScan(cross,Tg,Tc,λg,λc,Y,XX,Z;tdata=true,Xnul=Xnul,m=m,
                                       df_prior=df_prior,Prior=Prior,df_prior_τ2=df_prior_τ2,τ2_Pr=τ2_Pr,itol=itol,tol0=tol0,tol=tol)
       maxLODs, H1par_perm= permutation(nperm,cross,Y1,X1,Z1,est0,λg,λc,Xnul_t,df_prior,Prior,df_prior_τ2,τ2_Pr;tol0=tol0,tol=tol)
       maxLODs=convert(Array{Float64,1},maxLODs)
       cutoff= quantile(maxLODs,1.0.-pval)
    return maxLODs, H1par_perm, cutoff

end


#new version
# function permTest(nperm::Int64,cross,Kg,Y,XX::Markers,Z;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
#                Prior::Matrix{Float64}=diagm(ones(m)),Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
#     #permutation without LOCO
#        Tg,λg=K2eig(Kg)
#        est0,Xnul_t,Y1,X1,Z1  = gene1Scan(cross,Tg,λg,Y,XX,Z;tdata=true,m=m,df_prior=df_prior,
#                                           Prior=Prior,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)
#        maxLODs, H1par_perm= permutation(nperm,cross,Y1,X1,Z1,est0,λg,λc,Xnul_t,df_prior,Prior;tol0=tol0,tol=tol,ρ=ρ)
#        maxLODs=convert(Array{Float64,1},maxLODs)
#        cutoff= quantile(maxLODs,1.0.-pval)
#     return maxLODs, H1par_perm, cutoff

# end


#MVLMM
function permTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2),
                 df_Rprior=m+1,Rprior=diagm(ones(m)),Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4)
    #permutation without LOCO
       Tg,λg=K2eig(Kg)
       est0,Xnul_t,Y1,X1 = geneScan(cross,Tg,λg,Y,XX;tdata=true,m=m,df_prior=df_prior,Prior=Prior,df_Rprior=df_Rprior,
                                    Rprior=Rprior,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol)
       maxLODs, H1par_perm= permutation(nperm,cross,Y1,X1,est0,λg,Xnul_t,df_prior,Prior,df_Rprior,Rprior;tol0=tol0,tol=tol)
       maxLODs=convert(Array{Float64,1},maxLODs)
       cutoff= quantile(maxLODs,1.0.-pval)
    return maxLODs, H1par_perm, cutoff

end


# export trans2iid, permutY, permutation,permTest




