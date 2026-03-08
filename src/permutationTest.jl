##### permutation for no prior 
#actual null parameters by genescan from the data (not permuted) to permute 
function scan0loco(cross::Int64,n::Int64,m::Int64,Tg::Array{Float64,3},Λg::Matrix{Float64},Y::Array{Float64,2},XX::Markers
        ;Xnul::Array{Float64,2}=ones(1,n),Z=diagm(ones(m)),itol=1e-3,tol::Float64=1e-4,ρ=0.001)

         # LODs=zeros(p);
        Chr=unique(XX.chr); nChr=length(Chr);NulKc=[];tNuls=[];
        Λc= Array{Array{Float64,1}}(undef,nChr);fill!(Λc,zeros(m))
        Y0= similar(Y)

        #    tbyt, init= transByTrait(m,Tc,λc,Y,Z,Xnul,Prior)
        if (cross!=1)
            X1=mat2array(cross,XX.X)
         else
            X1=similar(XX.X) #pre-assigned 
         end
       
        
      if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)  
       else #Z0=I
         init0=initial(Xnul,Y,false)
      end
  
     for i= eachindex(Chr)
         maridx=findall(XX.chr.==Chr[i]);
            Λc[i], tt,init = getKc(init0,Y,Tg[:,:,i],Λg[:,i];Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)
        if (cross!=1)
           @fastmath @inbounds X1[:,:,maridx]=transForm(Tg[:,:,i],X1[:,:,maridx],cross)
         else
           @fastmath @inbounds X1[maridx,:]=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
        end
          Y0[:,:]= trans2iid(tt.Y,1.0,tt.Σ,Λg[:,i],Λc[i]) # preparing transformed Y to be iid for permutation
            
          tNuls=[tNuls;TNul(Y0,tt.Xnul,tt.Z,tt.Σ)];NulKc=[NulKc;init]

     end
        
         return Λc, tNuls, NulKc, X1
 
end

#no loco-null scan for permutation
function Scan0(cross::Int64,n::Int64,m::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers;Z=diagm(ones(m)),
            Xnul::Array{Float64,2}=ones(1,n),itol=1e-3,tol::Float64=1e-4,ρ=0.001)

    
    # q=size(Z,2);  
    # p=Int(size(XX.X,1)/cross); 

    ## picking up initial values for parameter estimation under the null hypothesis
    if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)  
       else #Z0=I
         init0=initial(Xnul,Y,false)
      end

   λc, tNul, NulKc = getKc(init0,Y,Tg,Λg;Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)  
    
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
      
   
    return λc,TNul(Y0,tNul.Xnul,tNul.Z,tNul.Σ), NulKc, X1
end


## finding distribution of max lod's for a multivariate model by permutation for 4waycross/intercross
function locoPermutation(cross::Int64,p::Int64,q::Int64,n::Int64,m::Int64,X::Union{Array{Float64,2},Array{Float64,3}},
        chr::Array{Any,1},tnuls,nulKc,λg::Array{Float64,2},λc::Array{Array{Float64,1},1}
        ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

#     n=size(Y,2); p=size(X,1);
    Chr=unique(chr)
    kmin=1; lod=zeros(p);H1par=[]

    rng=shuffle(1:n);
    Y1= Matrix{Float64}(undef,m,n)
    
    for l=eachindex(Chr)
         maridx=findall(chr.==Chr[j]);nmar=length(maridx)
       #transformation back to correlated Y after permutation
         Y1 = trans2iid(tnuls[l].Y[:,rng],1.0,tnuls[l].Σ,λg[:,l],λc[l],true) 
        
         if (cross!=1)
             X1=X[:,:,maridx]
            else 
             X1=X[maridx,:]
           end    
       
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(nulKc[l],kmin,λg[:,l],λc[l],Y1,tnuls[l].Xnul,tnuls[l].Z,tnuls[l].Σ,true;itol=tol0,tol=tol,ρ=ρ)    
        LODs,H1par0=marker1Scan(nmar,q,kmin,cross,perm_est0,λg[:,l],λc[l],Y1,tnuls[l].Xnul,X1,tnuls[l].Z;tol0=tol0,tol1=tol,ρ=ρ)
         
          lod[maridx]= LODs;  H1par=[H1par; H1par0]
           
        end

    return maximum(lod), H1par
end

function permutation(nperm::Int64,cross::Int64,p::Int64,q::Int64,X::Union{Array{Float64,2},Array{Float64,3}},
        tNul::TNul,Nullpar::InitKc,λg::Array{Float64,1},λc::Array{Float64,1}
        ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    # n=size(Y,2);    q=size(Z,2);
    lod=zeros(nperm);H1par=[]; Y2=similar(Y)

    # init=Init(Nullpar.B,Nullpar.τ2,Nullpar.Σ)
    
    for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        permutY!(Y2,tNul.Y,1.0,tNul.Σ,λg,λc)
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(Nullpar,1,λg,λc,Y2,tNul.Xnul,tNul.Z,tNul.Σ;ρ=ρ,itol=tol0,tol=tol)
        LODs,H1par0=marker1Scan(p,q,1,cross,perm_est0,λg,λc,Y2,tNul.Xnul,X,tNul.Z,true;tol0=tol0,tol1=tol,ρ=ρ)
         lod[l]= maximum(LODs);  H1par=[H1par; H1par0]

            if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
            end
        end

    return lod, H1par
end

#MVLMM
function Scan0(Tg,Λg,Y::Array{Float64,2},XX::Markers,cross::Int64;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    itol=1e-3,tol::Float64=1e-4,ρ=0.001)

   
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

                  est0=nulScan(init,1,Λg,Y,Xnul_t;itol=itol,tol=tol,ρ=ρ)
      
        return est0,Xnul_t,Y,X

end

function permutation(nperm::Int64,cross::Int64,p::Int64,Y::Array{Float64,2},X::Union{Array{Float64,2},Array{Float64,3}},
        Nullpar::Result,λg::Array{Float64,1},Xnul_t;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

     m=size(Y,1);
    kmin=1;lod=zeros(nperm);H1par=[]
   
    init=Init0(Nullpar.B,Nullpar.Vc,Nullpar.Σ)
    
     for l= 1:nperm
        ### permuting a phenotype matrix by individuals
        Y2=permutY(Y,Nullpar.Vc,Nullpar.Σ,λg);

        #initial parameter values for permutations are from genome scanning under the null hypothesis.
         perm_est0=nulScan(init,kmin,λg,Y2,Xnul_t;itol=tol0,tol=tol,ρ=ρ)
        LODs,H1par0=marker1Scan(p,m,kmin,cross,perm_est0,λg,Y2,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
    
         lod[l]= maximum(LODs);  H1par=[H1par; H1par0]
             if (mod(l,50)==0)
              println("Scan for $(l)th permutation is done.")
             end
        end


    return lod, H1par
end



function permutationTest(Kg::Union{Matrix{Float64},Array{Float64,3}},Y,XX::Markers,nperm::Int64,cross::Int64,LOCO::Bool=false;
       pval=[0.05 0.01],m=size(Y,1),Z=diagm(ones(m)),Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-4,tol=1e-4,ρ=0.001)
       
       p=Int(size(XX.X,1)/cross); n=size(Y,2); q=size(Z,2)
        if (!LOCO)
    #permutation without LOCO
          Tg,λg=K2Eig(Kg)
          λc,tNul, NulKc, X1=Scan0(cross,n,m,Tg,Λg,Y,XX;Z=Z,Xnul=Xnul,itol=itol,tol=tol,ρ=ρ)
          maxLODs, H1par= permutation(nperm,cross,p,q,X1,tNul,NulKc,λg,λc;tol0=tol0,tol=tol,ρ=ρ)
          maxLODs=convert(Array{Float64,1},maxLODs)
            
        else #loco test
            maxLODs = zeros(nperm);H1par=[]
            Tg,Λg=K2Eig(Kg,true)
            Λc, tNuls, NulKc, X1 = scan0loco(cross,n,m,Tg,Λg,Y,XX;Xnul=Xnul,Z=Z,itol=itol,tol=tol,ρ=ρ)
            for l=1:nperm
                maxLODs[l],H1par0 = locoPermutation(cross,p,q,n,m,X1,XX.chr,tNuls,NulKc,Λg,Λc;tol0=tol0,tol=tol,ρ=ρ)
                H1par=[H1par;H1par0]
                if (mod(l,100)==0)
                    println("Scan for $(l)th permutation is done.")
                end
            end
        end
            
    cutoff= quantile(maxLODs,1.0.-pval)

    return maxLODs, H1par, cutoff

end

"""

     permutationTest(nperm::Int64,cross::Int64,Kg::Array{Float64,3},Y::Array{Float64,2},XX::Markers,LOCO::Bool=false,penalize::Bool=false;pval=[0.05 0.01],
                     Z=diagm(ones(m)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),
                     itol=1e-4,tol0=1e-4,tol=1e-4,ρ=0.001)

     permTest(nperm,cross,Kg::Array{Float64,3},Y,XX;pval=[0.05 0.01],Z=diagm(ones(m)),df_prior=m+1,Prior=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),
              itol=1e-4,tol0=1e-4,tol=1e-4,ρ=0.001,δ=0.01)
                     
      
Implement permutation test by estimating the distribution of maximum LOD scores to get thresholds at the levels of type 1 error, `α`.  
The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), Kg \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `Kg` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is estimated under the null model (H0) of no QTL from the conventional MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  

!!! NOTE
- `permutationTest()` has options of penalization as well as LOCO.  Depending on the data size and computer capacity, one can select any combination.  
   There may exist slight to moderate differences in the choice of LOCO and penalization.
- `permTest()` is implemented by permutation test with LOCO, but ``\\Omega`` is estimated with a genetic kinship by [`kinshipLin`](@ref) without LOCO.
- [`mlmmTest`](@ref) is implemented by the conventional MLMM without LOCO.  

# Arguments

- `nperm` : An integer (Int64) indicating the number of permutation to be implemented.
- `cross` : An integer (Int64) indicating the number of combination of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degrees of freedom for the effect size of a genetic marker when doing genome scan.
- `Kg` : A 3d-array of n x n genetic kinship matrices (`LOCO = true` or only for `permTest()`) or a matrix of genetic kinship for the default option of `LOCO = false`. Should be symmetric positive definite.  
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out) if `true`.
- `penalize` : Boolean. Default is `false` (no penalization).  For higher dimensional traits, i.e. large `m=size(Y,1)`, penalization is recommended, i.e. set `penalize=true` for numerical 
           stability with adjustment of `df_prior` or/and `Prior` if necessary.

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (wavelet, polynomials, B-splines, etc.).
         If the data does not assume any particular trait relation, just use `Z = diagm(ones(m)) or Matrix(1.0I,m,m)` (default).  
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
            An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.  
- `δ` : A tuning parameter in `permTest()` to correct a non-positive definite kinship without LOCO to pre-estimate a null variance component. 
        This `no-LOCO` kinship is computed inside the function for efficient computation.

!!! Note
- When some LOD scores return negative values, you may reduce tolerences for ECM to `tol0 = 1e-4`, or increase `df_prior`, where ``m+1 \\le`` `df_prior` ``< 2m``.
   The last resort could be `df_prior = Int64(ceil(1.9m))` to avoid sigularity errors unless any of them works.  
    Adjusting `df_prior` works better than doing `'Prior`; we do recommend this adjustment for higher dimensional traits with genotype probability data, depending on the data--we have tested no penalization option witout error up to ``m = 16`` with 
   genotype probabilities or `m = 30` with genotypes.  Lower dimensional traits with penalization slow performance. 
- This LOCO version of permutation test is desirable to be implemented by high-performance computers.


# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A vector of structs, `EcmNestrv.Approx(B,τ2,Σ,loglik)` for each Chromosome per permutation, i.e. `# of Chromosomes` x `nperm`.
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function permutationTest(nperm::Int64,cross::Int64,Kg::Array{Float64,3},Y::Array{Float64,2},XX::Markers,LOCO::Bool=false,penalize::Bool=false;pval=[0.05 0.01],m=size(Y,1),
             Z=diagm(ones(m)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),
             itol=1e-4,tol0=1e-4,tol=1e-4,ρ=0.001)


             if (!penalize) #no penalization

               maxLODs, H1par, cutoff =  permutationTest(Kg,Y,XX,nperm,cross,LOCO;pval=pval,m=m,Z=Z,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ)

             else

               maxLODs, H1par, cutoff =  permutationTest(nperm,cross,Kg,Y,XX,LOCO;pval=pval,m=m,Z=Z,Xnul=Xnul,itol=itol,tol0=tol0,tol=tol,ρ=ρ,df_prior=df_prior,Prior=Prior)
             end

      return maxLODs, H1par, cutoff

end


#MVLMM
"""

     mlmmTest(nperm::Int64,cross::Int64,Kg,Y,XX::Markers;pval=[0.05 0.01],df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
   
Implement permutation test without LOCO to get thresholds at the levels of type 1 error, `α` by 
the conventional MLMM (`Z=I`) with an option of penalization. See also [`permutationTest`](@ref).

# Arguments

- `nperm` : An integer indicating the number of permutation to be implemented.
- `cross` : An integer indicating the number of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Kg` : A n x n genetic kinship matrix. Should be symmetric positive definite.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `penalize` : Boolean. Default is `false` (no penalization).  For higher dimensional traits, i.e. `large m=size(Y,1)`, penalization is recommended; set `penalize=true` 
           with adjustment of `df_prior` or/and `Prior` if necessary.

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
           An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-4`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.  


!!! Note
- When some LOD scores return negative values, you may reduce tolerence for ECM (`tol0`) to run longer, or increase `df_prior`, where
   ``m+1 \\le`` `df_prior` ``< 2m`` to avoid singularity errors.  The last resort could be `df_prior = Int64(ceil(1.9m))` unless any of them would works.   

# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A type of struct, `EcmNestrv.Approx(B,τ2,Σ,loglik)` including parameter estimates  or `EcmNestrv.Result(B,Vc,Σ,loglik)` 
                for a conventional MLMM under H0: no QTL by permutation. 
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function mlmmTest(nperm::Int64,cross::Int64,Kg,Y,XX::Markers,penalize::Bool=false;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
                 Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
    #permutation without LOCO
     p=Int(size(XX.X,1)/cross);
     Tg,λg=K2eig(Kg)

     if (!penalize) #no penalization
        est0,Xnul_t,Y1,X1 = Scan0(Tg,λg,Y,XX,cross;Xnul=Xnul,itol=itol,tol=tol,ρ=ρ)
        maxLODs, H1par_perm= permutation(nperm,cross,p,Y1,X1,est0,λg,Xnul_t;tol0=tol0,tol=tol,ρ=ρ)
     else  #penalization
         est0,Xnul_t,Y1,X1 = Scan0(cross,Tg,λg,Y,XX;Xnul=Xnul,m=m, df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
         maxLODs, H1par_perm= permutation(nperm,cross,p,Y1,X1,est0,λg,Xnul_t,df_prior,Prior;tol0=tol0,tol=tol,ρ=ρ)
     end

     maxLODs=convert(Array{Float64,1},maxLODs)
     cutoff= quantile(maxLODs,1.0.-pval)
     
    return maxLODs, H1par_perm, cutoff

end





