

###Permutation test : all permutations are implemented without loco
struct TbyT
Y::Matrix{Float64}
Z::Matrix{Float64}
Ψ::Matrix{Float64}
Σ::Matrix{Float64}
end


function transByTrait(m,Tc,λc,Y,Z,Xnul,Prior)

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

    return TbyT(Y1,Z1,Ψ,Σ1),init
end


## permuteY : shuffling a Y matrix columnwise (by individuals)
function permutY!(Y2::Matrix{Float64},rng::Vector{Int64},Y::Array{Float64,2},τ2_nul::Float64,Σ_nul::Array{Float64,2}
        ,λg::Array{Float64,1},λc::Array{Float64,1})
   
    ### permutation by individuals
    #transforming to iid Y1
    Y_t=trans2iid(Y,τ2_nul,Σ_nul,λg,λc);
    # rng=shuffle(1:n);
    # transforming back to correlated Y1 after permuting
    Y2[:,:]=trans2iid(Y_t[:,rng],τ2_nul,Σ_nul,λg,λc,true);
  
end

#MVLMM
function permutY!(Y2::Matrix{Float64},rng::Vector{Int64},Y::Array{Float64,2},Vc_nul::Array{Float64,2},Σ_nul::Array{Float64,2},λg::Array{Float64,1})
   
    ### permutation by individuals
    #transforming to iid Y1
    Y_t=trans2iid(Y,Vc_nul,Σ_nul,λg);
    # rng=shuffle(1:n);
    # transforming back to correlated Y1 after permuting
    Y2[:,:]=trans2iid(Y_t[:,rng],Vc_nul,Σ_nul,λg,true);

end

function scan0loco(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y::Array{Float64,2},XX::Markers,
        Z::Array{Float64,2};Xnul::Array{Float64,2}=ones(1,size(Y,2)),m=size(Y,1),df_prior=m+1,
                Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

         # LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[]
        p0,n=size(Xnul)
        Xnul_t=Array{Array{Float64,2}}(undef,nChr);fill!(Xnul_t,zeros(p0,n))
        Y2 = Array{Array{Float64,2}}(undef,nChr);fill!(Y2,zeros(m,n))

           tbyt, init= transByTrait(m,Tc,λc,Y,Z,Xnul,Prior)
        if (cross!=1)
            X1=mat2array(cross,XX.X)
         else
            X1=similar(XX.X) #pre-assigned 
         end
       
        
            # rng=shuffle(1:n);
     for i=1:nChr
         maridx=findall(XX.chr.==Chr[i]);
            @fastmath @inbounds Xnul_t[i]=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
        if (cross!=1)
           @fastmath @inbounds Y2[i],X1[:,:,maridx]=transForm(Tg[:,:,i],tbyt.Y,X1[:,:,maridx],cross)
           else
           @fastmath @inbounds Y2[i],X1[maridx,:]=transForm(Tg[:,:,i],tbyt.Y,XX.X[maridx,:],cross)
        end
                #parameter estimation under the null
            est00=nulScan(init,1,Λg[:,i],λc,Y2[i],Xnul_t[i],tbyt.Z,tbyt.Σ,df_prior,tbyt.Ψ;ρ=ρ,itol=itol,tol=tol)
            est0=[est0;est00];

     end
        
         return est0, Xnul_t,Y2,X1,tbyt
    
end

##MVLMM
function scan0loco(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    m=size(Y,1), df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*5,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

   
        Chr=unique(XX.chr);nChr=length(Chr);est0=[]
        p0,n=size(Xnul)
        Xnul_t=Array{Array{Float64,2}}(undef,nChr);fill!(Xnul_t,zeros(p0,n))
        Y2 = Array{Array{Float64,2}}(undef,nChr);fill!(Y2,zeros(m,n))

    #check the prior
    if (!isposdef(Prior))
        println("Error! Plug in a postivie definite Prior!")
     end

     #initialization
       init=initial(Xnul,Y,false)
        if (cross!=1)
            X1=mat2array(cross,XX.X)
            else
            X1=similar(XX.X) #pre-assigned 
         end

    for i=1:nChr
         maridx=findall(XX.chr.==Chr[i]);
            @fastmath @inbounds Xnul_t[i]=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
        if (cross!=1)
           @fastmath @inbounds Y2[i],X1[:,:,maridx]=transForm(Tg[:,:,i],Y,X1[:,:,maridx],cross)
           else
           @fastmath @inbounds Y2[i],X1[maridx,:]=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
        end
                #parameter estimation under the null
            est00=nulScan(init,1,Λg[:,i],Y2[i],Xnul_t[i],df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
            est0=[est0;est00];

     end
        
         return est0, Xnul_t,Y2,X1
    
end


## finding distribution of max lod's for a multivariate model by permutation for 4waycross/intercross
function locoPermutation(cross::Int64,p::Int64,Y::Array{Array{Float64,2},1},X::Union{Array{Float64,2},Array{Float64,3}},chr::Array{Any,1},
       Nullpar,TT::TbyT,Λg::Array{Float64,2},λc::Array{Float64,1},Xnul_t::Array{Array{Float64,2},1},ν₀
        ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

#     n=size(Y,2); p=size(X,1);
    q=size(TT.Z,2);m,n=size(Y[1]);Chr=unique(chr);nChr=length(Chr);
    lod=zeros(p);H1par=[]

       # for l= 1:nperm
       #generating a new dataset by shuffling
        rng=shuffle(1:n);
        Y1= Matrix{Float64}(undef,m,n)
       for j=1:nChr
        maridx=findall(chr.==Chr[j]);nmar=length(maridx)
        init=Init(Nullpar[j].B,Nullpar[j].τ2,Nullpar[j].Σ)
        permutY!(Y1,rng,Y[j],init.τ2,init.Σ,Λg[:,j],λc);
         
           if (cross!=1)
             X1=X[:,:,maridx]
            else 
             X1=X[maridx,:]
           end         
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(init,1,Λg[:,j],λc,Y1,Xnul_t[j],TT.Z,init.Σ,ν₀,TT.Ψ;ρ=ρ,itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(nmar,q,1,cross,perm_est0,Λg[:,j],λc,Y1,Xnul_t[j],X1,TT.Z,ν₀,TT.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
         lod[maridx]= LODs;  H1par=[H1par; H1par0]

            # if (mod(l,100)==0)
            #   println("Scan for $(l)th permutation is done.")
            # end
        end

    return maximum(lod), H1par
end


#MVLMM
function locoPermutation(cross::Int64,p::Int64,Y::Array{Array{Float64,2},1},X::Union{Array{Float64,2},Array{Float64,3}},chr::Array{Any,1},
        Nullpar,Λg::Array{Float64,2},Xnul_t::Array{Array{Float64,2},1},ν₀,Ψ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    m,n=size(Y[1]);Chr=unique(chr);nChr=length(Chr);
    lod=zeros(p);H1par=[]
    
    # init=Init0(Nullpar.B,Nullpar.Vc,Nullpar.Σ)
    
    #  for l= 1:nperm
       #generating a new dataset by shuffling
         rng=shuffle(1:n);
        Y1= Matrix{Float64}(undef,m,n)
       for j=1:nChr
        maridx=findall(chr.==Chr[j]);nmar=length(maridx)
        init=Init0(Nullpar[j].B,Nullpar[j].Vc,Nullpar[j].Σ)
        permutY!(Y1,rng,Y[j],init.Vc,init.Σ,Λg[:,j]);
         
           if (cross!=1)
             X1=X[:,:,maridx]
            else 
             X1=X[maridx,:]
           end         
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(init,1,Λg[:,j],Y1,Xnul_t[j],ν₀,Ψ;ρ=ρ,itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(nmar,m,1,cross,perm_est0,Λg[:,j],Y1,Xnul_t[j],X1,ν₀,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
         lod[maridx]= LODs;  H1par=[H1par; H1par0]

    
        end

    return maximum(lod), H1par

end





"""

      permutationTest(nperm::Int64,cross,Kg,Kc,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),Z=diagm(ones(m)),df_prior=m+1,
          Prior::Matrix{Float64}=cov(Y,dims=2)*5,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
      permutationTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2)*5,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
   
Implement permutation test by geneScan with LOCO  to get thresholds at the levels of type 1 error, `α`.  Note that the last `permutationTest()` 
is for the conventional MLMM: 
```math
vec(Y)\\sim MVN((I \\otimes X)vec(B) (or BX), K \\otimes \\Sigma_1 +I \\otimes \\Sigma_2),
``` 
where `K` is a genetic kinship, ``\\Sigma_1, \\Sigma_2`` are covariance matrices for random and error terms, respectively.

!!! NOTE
- `permTest()` is implemented by geneScan without LOCO.  

# Arguments

- `nperm` : An integer indicating the number of permutation to be implemented.
- `cross` : An integer indicating the number of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Kg` : A n x n genetic kinship matrix. Should be symmetric positive definite.
- `Kc` : A m x m precomputed covariance matrix of `Kc` under the null model of no QTL. Should be symmetric positive definite.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
        Default is an identity matrix for the dimension of m traits x q phenotypic covariates.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           A large scaled covariance matrix (a weakly informative prior) is default.
- `df_prior`: degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.  

# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A vector of structs, `EcmNestrv.Approx(B,τ2,Σ,loglik)` including parameter estimates  or `EcmNestrv.Result(B,Vc,Σ,loglik)` 
                for a conventional MLMM under H0: no QTL by permutation. 
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function permutationTest(nperm::Int64,cross,Kg,Kc,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),Z=diagm(ones(m)),df_prior=m+1,
        Prior::Matrix{Float64}=cov(Y,dims=2)*5,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
    #permutation without LOCO
    p=Int(size(XX.X,1)/cross); maxLODs = zeros(nperm);H1par=[]
    Tg,Λg,Tc,λc=K2Eig(Kg,Kc,true)
    est0,Xnul_t,Y2,X1,TTr= scan0loco(cross,Tg,Tc,Λg,λc,Y,XX,Z;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    
    for l= 1:nperm
    maxlod, H1par_perm=locoPermutation(cross,p,Y2,X1,XX.chr,est0,TTr,Λg,λc,Xnul_t,df_prior;tol0=tol0,tol=tol,ρ=ρ)
    maxLODs[l]=maxlod; H1par=[H1par;H1par_perm]
             if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
             end
    end
    maxLODs=convert(Array{Float64,1},maxLODs)
    cutoff= quantile(maxLODs,1.0.-pval)

    return maxLODs, H1par, cutoff

end



#MVLMM
function permutationTest(nperm::Int64,cross,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2)*5,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
    #permutation without LOCO
      p=Int(size(XX.X,1)/cross); maxLODs = zeros(nperm);H1par=[]
     Tg,Λg=K2eig(Kg,true)
     
      est0,Xnul_t,Y2,X1= scan0loco(cross,Tg,Λg,Y,XX;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    
    for l= 1:nperm
    maxlod, H1par_perm=locoPermutation(cross,p,Y2,X1,XX.chr,est0,Λg,Xnul_t,df_prior,Prior;tol0=tol0,tol=tol,ρ=ρ)
    maxLODs[l]=maxlod; H1par=[H1par;H1par_perm]
             if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
             end
    end
    maxLODs=convert(Array{Float64,1},maxLODs)
    cutoff= quantile(maxLODs,1.0.-pval)
     
    return maxLODs, H1par, cutoff

end







