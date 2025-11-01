

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
function permutY!(Y1::Array{Float64},Y::Array{Float64,2},τ2_nul::Float64,Σ_nul::Array{Float64,2}
        ,λg::Array{Float64,1},λc::Array{Float64,1})
#    n=size(Y,2);
    ### permutation by individuals
    #transforming to iid Y1
    # Y_t=trans2iid(Y,τ2_nul,Σ_nul,λg,λc);
    rng=shuffle(axes(Y,2));
    # transforming back to correlated Y1 after permuting
    Y1[:,:]=trans2iid(Y[:,rng],τ2_nul,Σ_nul,λg,λc,true);

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

#MVLMM
function permutY!(Y2::Matrix{Float64},Y::Array{Float64,2},Vc_nul::Array{Float64,2},Σ_nul::Array{Float64,2},λg::Array{Float64,1})
    #  n=size(Y,2);
    ### permutation by individuals
    #transforming to iid Y1
    # Y_t=trans2iid(Y,Vc_nul,Σ_nul,λg);
    rng=shuffle(axes(Y,2));
    # transforming back to correlated Y1 after permuting
    Y2[:,:]=trans2iid(Y[:,rng],Vc_nul,Σ_nul,λg,true);

end


## finding distribution of max lod's for a multivariate model by permutation for 4waycross/intercross
function permutation(nperm::Int64,cross::Int64,p::Int64,q::Int64,Y::Array{Float64,2},X::Union{Array{Float64,2},Array{Float64,3}},
        tNul::TNull,Nullpar::InitKc,λg::Array{Float64,1},λc::Array{Float64,1},ν₀
        ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    # n=size(Y,2);    q=size(Z,2);
    lod=zeros(nperm);H1par=[]; Y2=similar(Y)

    init=Init(Nullpar.B,Nullpar.τ2,Nullpar.Σ)
    
    for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        Y2=permutY!(Y2,Y,1.0,tNul.Σ,λg,λc)
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(init,1,λg,λc,Y2,tNul.Xnul,tNul.Z,tNul.Σ,ν₀,tNul.Ψ;ρ=ρ,itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(p,q,1,cross,perm_est0,λg,λc,Y2,tNul.Xnul,X,tNul.Z,ν₀,tNul.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
         lod[l]= maximum(LODs);  H1par=[H1par; H1par0]

            if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
            end
        end

    return lod, H1par
end


#MVLMM
function permutation(nperm::Int64,cross::Int64,p::Int64,Y::Array{Float64,2},X::Union{Array{Float64,2},Array{Float64,3}},
        Nullpar::Result,λg::Array{Float64,1},Xnul_t,ν₀,Ψ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

     m=size(Y,1);
    lod=zeros(nperm);H1par=[]
   
    init=Init0(Nullpar.B,Nullpar.Vc,Nullpar.Σ)
    
     for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        Y2=permutY(Y,Nullpar.Vc,Nullpar.Σ,λg);

        #initial parameter values for permutations are from genome scanning under the null hypothesis.
         perm_est0=nulScan(init,1,λg,Y2,Xnul_t,ν₀,Ψ;itol=tol0,tol=tol,ρ=ρ)
        LODs,H1par0=marker1Scan(p,m,1,cross,perm_est0,λg,Y2,Xnul_t,X,ν₀,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
    
         lod[l]= maximum(LODs);  H1par=[H1par; H1par0]
             if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
             end
        end


    return lod, H1par
end



#no loco-null scan for permutation
function Scan0(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2};
            Xnul::Array{Float64,2}=ones(1,size(Y,2)),m=size(Y,1),df_prior=m+1,
            Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

    
    # q=size(Z,2);  
    # p=Int(size(XX.X,1)/cross); 

    ## picking up initial values for parameter estimation under the null hypothesis
    if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)  
       else #Z0=I
         init0=initial(Xnul,Y,false)
      end

   λc, tNul, NulKc = getKc(Y,Tg,Λg,init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)  
    
     if (cross!=1)
        X0=mat2array(cross,XX.X)
     end
#        
             if (cross!=1)
               X1=transForm(Tg,X0,cross)
               else
               X1=transForm(Tg,XX.X,cross)
             end
             
        Y0 = trans2iid(tNul.Y,1.0,tNul.Σ,Λg,λc)
        
   
    return λc, tNul, NulKc, Y0, X1
end

##MVLMM
function Scan0(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    m=size(Y,1), df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

   
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
    
#             Xnul_t=Xnul*Tg';
             Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y,X=transForm(Tg,Y,X0,cross)
                   else
                   Y,X=transForm(Tg,Y,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
      
        return p,est0,Xnul_t,Y,X

end



"""

      permTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),Z=diagm(ones(m)),df_prior=m+1,
              Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
      
      mlmmTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],df_prior=m+1,
               Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
   
Implement permutation test to get thresholds at the levels of type 1 error, `α`.  Note that `mlmmTest()` 
is based on the conventional MLMM (`Z=I`).
The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), K \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `K` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is pre-estimated under the null model (`H0`) of no QTL from the conventional MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  

!!! NOTE
- `permutationTest()` is implemented by `geneScan` with LOCO.  

# Arguments

- `nperm` : An integer indicating the number of permutation to be implemented.
- `cross` : An integer indicating the number of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Kg` : A n x n genetic kinship matrix. Should be symmetric positive definite.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
         If the data does not assume any particular trait relation, just use `Z = I` (default).  
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.  


!!! Note
- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`, or increase `df_prior`, such that 
   ``m+1 \\le`` `df_prior` ``< 2m``.  The easiest setting is `df_prior = Int64(ceil(1.9m))` for numerical stability.   

# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A type of struct, `EcmNestrv.Approx(B,τ2,Σ,loglik)` including parameter estimates  or `EcmNestrv.Result(B,Vc,Σ,loglik)` 
                for a conventional MLMM under H0: no QTL by permutation. 
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function permTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),Z=diagm(ones(m)),df_prior=m+1,
        Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
    #permutation without LOCO
    p=Int(size(XX.X,1)/cross);  q=size(Z,2)
    Tg,λg=K2eig(Kg)
    λc, tNul, NulKc, Y0, X1 = Scan0(cross,Tg,λg,Y,XX,Z;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    maxLODs, H1par_perm= permutation(nperm,cross,p,q,Y0,X1,tNul,NulKc,λg,λc,df_prior;tol0=tol0,tol=tol,ρ=ρ)
    maxLODs=convert(Array{Float64,1},maxLODs)
    cutoff= quantile(maxLODs,1.0.-pval)

    return maxLODs, H1par_perm, cutoff

end



#MVLMM
function mlmmTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
    #permutation without LOCO
     Tg,λg=K2eig(Kg)
     p,est0,Xnul_t,Y1,X1 = Scan0(cross,Tg,λg,Y,XX;Xnul=Xnul,m=m, df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
     maxLODs, H1par_perm= permutation(nperm,cross,p,Y1,X1,est0,λg,Xnul_t,df_prior,Prior;tol0=tol0,tol=tol,ρ=ρ)
     maxLODs=convert(Array{Float64,1},maxLODs)
     cutoff= quantile(maxLODs,1.0.-pval)
     
    return maxLODs, H1par_perm, cutoff

end




include("PpermutationTest1.jl")


