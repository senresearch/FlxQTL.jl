

###Permutation test : all permutations are implemented by loco
struct TbyT
Xnul::Matrix{Float64}
Z::Matrix{Float64}
Σ::Matrix{Float64}
Ψ::Matrix{Float64}
end

struct Tbyt
Y::Matrix{Float64}
Z::Matrix{Float64}
Σ::Matrix{Float64}
Ψ::Matrix{Float64}
end


#actual null parameters by genescan from the data (not permuted) to permute (m>16 or20)
function scan0loco(cross::Int64,Tg::Array{Float64,3},Λg::Matrix{Float64},Y::Array{Float64,2},XX::Markers,
        Z::Array{Float64,2},n::Int64,m::Int64;Xnul::Array{Float64,2}=ones(1,n),df_prior=m+1,
                Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

         # LODs=zeros(p);
        Chr=unique(XX.chr); nChr=length(Chr);NulKc=[];tNuls=[];
        Λc= Array{Array{Float64,1}}(undef,nChr);fill!(Λc,zeros(m))
        Y0 = Array{Array{Float64,2}}(undef,nChr);fill!(Y0,zeros(m,n))

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
            Λc[i], tt,init = getKc(Y,Tg[:,:,i],Λg[:,i],init0;Xnul=Xnul,m=m,Z=Z,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
        if (cross!=1)
           @fastmath @inbounds X1[:,:,maridx]=transForm(Tg[:,:,i],X1[:,:,maridx],cross)
         else
           @fastmath @inbounds X1[maridx,:]=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
        end
          Y0[i]= trans2iid(tt.Y,1.0,tt.Σ,Λg[:,i],Λc[i]) # preparing transformed Y to be iid for permutation
            
          tNuls=[tNuls;TbyT(tt.Xnul,tt.Z,tt.Σ,tt.Ψ)];NulKc=[NulKc;init]

     end
        
         return Λc, tNuls, NulKc, Y0, X1
 
end


#estimating Kc w/o loco for permutation
function getKc(Y::Array{Float64,2},Kg::Array{Float64,2},init::Init0;m=size(Y,1),Z=diagm(ones(m)), df_prior=m+1,
     Prior::Matrix{Float64}=cov(Y,dims=2)*3,
     Xnul::Array{Float64,2}=ones(1,size(Y,2)),itol=1e-2,tol::Float64=1e-3,ρ=0.001)

      Tg,λg=K2eig(Kg)
      Y1,Xnul_t = transForm(Tg,Y,Xnul,1) #null model transformation

     est0= nul1Scan(init,1,λg,Y1,Xnul_t,Z,m,df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
      Tc, λc = K2eig(est0.Vc)
     Y1,Z1,Σ1,Ψ = transByTrait(m,Tc,Y,Z,Prior,est0)
      τ² =mean(Diagonal(est0.Vc)./m)
     
    #   Y1 = trans2iid(Y1,1.0,Σ1,λg,λc) #tranform to iid

   return λc, Tbyt(Y1,Z1,Σ1,Ψ),InitKc(est0.Vc,est0.B,est0.Σ,τ²,est0.loglik)

end

# Kc (no loco) (m<16 or 20)
function scan0loco(cross::Int64,Kg::Matrix{Float64},Tg::Array{Float64,3},Λg::Matrix{Float64},Y::Array{Float64,2},XX::Markers,
        Z::Array{Float64,2},n::Int64,m::Int64;Xnul::Array{Float64,2}=ones(1,n),df_prior=m+1,
                Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

       Chr=unique(XX.chr); nChr=length(Chr); p0,n=size(Xnul)
       Xnul_t=Array{Array{Float64,2}}(undef,nChr);fill!(Xnul_t,zeros(p0,n))
       Y0 = Array{Array{Float64,2}}(undef,nChr);fill!(Y0,zeros(m,n))

       
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
       
       λc, tNul, initKc = getKc(Y,Kg,init0;m=m,Z=Z,df_prior=df_prior,Prior=Prior,Xnul=Xnul,itol=itol,tol=tol,ρ=ρ)
       
       for i = eachindex(Chr)
        maridx=findall(XX.chr.==Chr[i]);
         @fastmath @inbounds Xnul_t[i]=BLAS.gemm('N','T',Xnul,Tg[:,:,i])
        if (cross!=1)
           @fastmath @inbounds Y0[i], X1[:,:,maridx]=transForm(Tg[:,:,i],tNul.Y,X1[:,:,maridx],cross)
         else
           @fastmath @inbounds Y0[i],X1[maridx,:]=transForm(Tg[:,:,i],tNul.Y,XX.X[maridx,:],cross)
        end

         Y0[i]= trans2iid(Y0[i],1.0,tNul.Σ,Λg[:,i],λc)
         
       end

        return λc, tNul, initKc, Y0, X1, Xnul_t

end


##MVLMM
function scan0loco(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    m=size(Y,1), df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,itol=1e-3,tol::Float64=1e-4,ρ=0.001)

   
        Chr=unique(XX.chr);est0=[]
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

    for i=eachindex(Chr)
         maridx=findall(XX.chr.==Chr[i]);
            @fastmath @inbounds Xnul_t[i]=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
        if (cross!=1)
           @fastmath @inbounds Y2[i],X1[:,:,maridx]=transForm(Tg[:,:,i],Y,X1[:,:,maridx],cross)
           else
           @fastmath @inbounds Y2[i],X1[maridx,:]=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
        end
                #parameter estimation under the null
            est00=nulScan(init,1,Λg[:,i],Y2[i],Xnul_t[i],df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
            # transformation to iid for permutation later
            Y2[i]=trans2iid(Y2[i],est00.Vc,est00.Σ,Λg[:,i])
            est0=[est0;est00];

     end
        
         return est0, Xnul_t,Y2,X1
    
end


## finding distribution of max lod's for a multivariate model by permutation 
function locoPermutation(cross::Int64,p::Int64,q::Int64,m::Int64,n::Int64,Y0::Array{Array{Float64,2},1},
               X::Union{Array{Float64,2},Array{Float64,3}},chr::Array{Any,1},NulKc,TT,
             Λg::Array{Float64,2},λc::Array{Array{Float64,1},1},ν₀;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    Chr=unique(chr)
    lod=zeros(p);H1par=[]

       #generating a new dataset by shuffling
        rng=shuffle(1:n);
        Y1= Matrix{Float64}(undef,m,n)
        
       for j=eachindex(Chr)
        maridx=findall(chr.==Chr[j]);nmar=length(maridx)
        #transformation back to correlated Y after permutation
         Y1 = trans2iid(Y0[j][:,rng],1.0,TT[j].Σ,Λg[:,j],λc[j],true) 

          if (cross!=1)
             X1=X[:,:,maridx]
            else 
             X1=X[maridx,:]
           end         
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(NulKc[j],1,Λg[:,j],λc[j],Y1,TT[j].Xnul,TT[j].Z,TT[j].Σ,ν₀,TT[j].Ψ,true;ρ=ρ,itol=tol0,tol=tol)
        # print(typeof(perm_est0))
        LODs,H1par0=marker1Scan(nmar,q,1,cross,perm_est0,Λg[:,j],λc[j],Y1,TT[j].Xnul,X1,TT[j].Z,ν₀,TT[j].Ψ;tol0=tol0,tol1=tol,ρ=ρ)
         lod[maridx]= LODs;  H1par=[H1par; H1par0]

        end

    return maximum(lod), H1par
end

# Kc(no loco) (m<16 or 20)
function locoPermutation(cross::Int64,p::Int64,q::Int64,m::Int64,n::Int64,Y0::Array{Array{Float64,2},1},
    Xnul_t::Array{Array{Float64,2},1},X::Union{Array{Float64,2},Array{Float64,3}},chr::Array{Any,1},
    NulKc,TT,Λg::Array{Float64,2},λc::Vector{Float64},ν₀;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

     Chr=unique(chr)
    lod=zeros(p);H1par=[]
   
       #generating a new dataset by shuffling
        rng=shuffle(1:n);
        Y1= Matrix{Float64}(undef,m,n)
        
       for j=eachindex(Chr)
        maridx=findall(chr.==Chr[j]);nmar=length(maridx)
        #transformation back to correlated Y after permutation
         Y1 = trans2iid(Y0[j][:,rng],1.0,TT.Σ,Λg[:,j],λc,true) 

          if (cross!=1)
             X1=X[:,:,maridx]
            else 
             X1=X[maridx,:]
           end         
        #initial parameter values for permutations are from genome scanning under the null hypothesis.
        perm_est0=nulScan(NulKc,1,Λg[:,j],λc,Y1,Xnul_t[j],TT.Z,TT.Σ,ν₀,TT.Ψ,true;ρ=ρ,itol=tol0,tol=tol)
        # print(typeof(perm_est0))
        LODs,H1par0=marker1Scan(nmar,q,1,cross,perm_est0,Λg[:,j],λc,Y1,Xnul_t[j],X1,TT.Z,ν₀,TT.Ψ;tol0=tol0,tol1=tol,ρ=ρ)
         lod[maridx]= LODs;  H1par=[H1par; H1par0]

        end

        return maximum(lod), H1par
end

#MVLMM
function locoPermutation(cross::Int64,p::Int64,m::Int64,n::Int64,Y::Array{Array{Float64,2},1},X::Union{Array{Float64,2},Array{Float64,3}},chr::Array{Any,1},
        Nullpar,Λg::Array{Float64,2},Xnul_t::Array{Array{Float64,2},1},ν₀,Ψ;tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    Chr=unique(chr);
    lod=zeros(p);H1par=[]
    
    # init=Init0(Nullpar.B,Nullpar.Vc,Nullpar.Σ)
    
    #  for l= 1:nperm
       #generating a new dataset by shuffling
         rng=shuffle(1:n);
         Y1= Matrix{Float64}(undef,m,n)
       for j=eachindex(Chr)
        maridx=findall(chr.==Chr[j]);nmar=length(maridx)
        init=Init0(Nullpar[j].B,Nullpar[j].Vc,Nullpar[j].Σ)
        # transformation back to the correlated Y
         Y1=trans2iid(Y[j][:,rng],Nullpar[j].Vc,Nullpar[j].Σ,Λg[:,j],true)
         
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

      permutationTest(nperm,cross,Kg,Y,XX;pval=[0.05 0.01],Z=diagm(ones(m)),Xnul=ones(1,size(Y,2)),
                     df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,
                     LOCO_all::Bool=false,itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001,δ =0.01)
      
Implement permutation test from the distribution of maximum LOD scores by LOCO to get thresholds at the levels of type 1 error, `α`.  
The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), K \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `K` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is pre-estimated under the null model (H0) of no QTL from the conventiona MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  

!!! NOTE
- `permTest()` and `mlmmTest()` are implemented by `geneScan` without LOCO.  

# Arguments

- `nperm` : An integer (Int64) indicating the number of permutation to be implemented.
- `cross` : An integer (Int64) indicating the number of combination of alleles or genotypes. Ex. `2` for RIF, `4` for four-way cross, `8` for HS mouse (allele probabilities), etc.
          This value is related to degrees of freedom for the effect size of a genetic marker when doing genome scan.
- `Kg` : A 3d-array of n x n genetic kinship matrices. Should be symmetric positive definite.
- `Y` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y[1,:]`  (a vector) -> `Y[[1],:]`  (a matrix) .
- `XX` : A type of [`Markers`](@ref).

## Keyword Arguments 

- `pval` : A vector of p-values to get their quantiles. Default is `[0.05  0.01]` (without comma).
- `Xnul` : A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `Z` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (wavelet, polynomials, B-splines, etc.).
         If the data does not assume any particular trait relation, just use `Z = diagm(ones(m)) or Matrix(1.0I,m,m)` (default).  
- `Prior`: A positive definite scale matrix, ``\\Psi``, of prior Inverse-Wishart distributon, i.e. ``\\Sigma \\sim W^{-1}_m (\\Psi, \\nu_0)``.  
            An amplified empirical covariance matrix is default.
- `df_prior`: Degrees of freedom, ``\\nu_0`` for Inverse-Wishart distributon.  `m+1` (weakly informative) is default.
- `LOCO_all` : Boolean. Default is `false`, which implements `geneScan(LOCO=true)` with permuted data but a null variance component (`Vc`) preestimated only once
               with a kinship ([`kinshipLin`](@ref) for genotype (or allele) probabilities, or [`kinshipStd`](@ref) for genotypes) by `LOCO=false` implicitly.  
               It is recommended setting `true` for higher-dimensional traits for faster convergence and decent accuracy, i.e. approximately ``m > 15`` depending on the data.
- `itol` : A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` : A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.  
- `δ` : A tuning parameter to correct a non-positive definite kinship without LOCO to pre-estimate a null variance component for low- to medium-dimensional
        traits (``m \\le 10 \\sim 15``) only.  This `no-LOCO` kinship is computed inside the function for efficient computation.

!!! Note
- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`, or increase `df_prior`, such that 
   ``m+1 \\le`` `df_prior` ``< 2m``.  The easiest setting is `df_prior = Int64(ceil(1.9m))` for numerical stability.   


# Output

- `maxLODs` : A nperm x 1 vector of maximal LOD scores by permutation. 
- `H1par_perm` : A vector of structs, `EcmNestrv.Approx(B,τ2,Σ,loglik)` for each Chromosome per permutation, i.e. `# of Chromosomes` x `nperm`.
- `cutoff` : A vector of thresholds corresponding to `pval`.


"""
function permutationTest(nperm::Int64,cross::Int64,Kg::Array{Float64,3},Y::Array{Float64,2},XX::Markers;pval=[0.05 0.01],m=size(Y,1),
             Z=diagm(ones(m)),df_prior=m+1,Prior::Matrix{Float64}=cov(Y,dims=2)*3,Xnul=ones(1,size(Y,2)),LOCO_all::Bool=false,
             itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001,δ=0.01)
    #permutation without LOCO
    p=Int(size(XX.X,1)/cross); n=size(Y,2); q=size(Z,2)
    maxLODs = zeros(nperm);H1par=[]
    Tg,Λg=K2eig(Kg,true)
 if (LOCO_all) #m >16 or 18
   println("Start estimating null parameters by the LOCO scheme for permutations.")
    λc, tNuls, NulKc, Y0, X1 = scan0loco(cross,Tg,Λg,Y,XX,Z,n,m;Xnul=Xnul,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
   println("Start scanning genetic markers by the LOCO scheme with $(m)-trait permuted data") 
    for l= 1:nperm
    maxlod, H1par_perm=locoPermutation(cross,p,q,m,n,Y0,X1,XX.chr,NulKc,tNuls,Λg,λc,df_prior;tol0=tol0,tol=tol,ρ=ρ)
    maxLODs[l]=maxlod; H1par=[H1par;H1par_perm]
             if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
             end
    end

   else # m <= 16 or 20
    println("Estimate a no-LOCO kinship for null variance component preestimation only for $(m) traits.")
     if (cross!=1)
        K = kinshipLin(XX.X,cross)
     else
        K= kinshipStd(XX.X)
     end
    
     if (!isposdef(K))
        K= K+δ*I
     println("Positive definiteness of the kinship is corrected to be ", isposdef(K),".")
     end
     println("Start estimating null parameters without LOCO for permutations.")
     λc, tNul, initKc, Y0, X1, Xnul_t = scan0loco(cross,K,Tg,Λg,Y,XX,Z,n,m;Xnul=Xnul,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    println("Start scanning genetic markers by the LOCO scheme with $(m)-trait permuted data") 
     for l = 1:nperm
        maxlod, H1par_perm=locoPermutation(cross,p,q,m,n,Y0,Xnul_t,X1,XX.chr,initKc,tNul,Λg,λc,df_prior;tol0=tol0,tol=tol,ρ=ρ)
        maxLODs[l]=maxlod; H1par=[H1par;H1par_perm]
             if (mod(l,100)==0)
              println("Scan for $(l)th permutation is done.")
             end

     end

end
    maxLODs=convert(Array{Float64,1},maxLODs)
    cutoff= quantile(maxLODs,1.0.-pval)

    return maxLODs, H1par, cutoff

end



# #MVLMM
# function permutationTest(nperm::Int64,cross::Int64,Kg,Y,XX::Markers;pval=[0.05 0.01],m=size(Y,1),df_prior=m+1,
#                  Prior::Matrix{Float64}=cov(Y,dims=2)*3,n=size(Y,2),Xnul=ones(1,n),itol=1e-4,tol0=1e-3,tol=1e-4,ρ=0.001)
#     #permutation without LOCO
#       p=Int(size(XX.X,1)/cross); 
#       maxLODs = zeros(nperm);H1par=[]
#      Tg,Λg=K2eig(Kg,true)
     
#       est0,Xnul_t,Y2,X1= scan0loco(cross,Tg,Λg,Y,XX;Xnul=Xnul,m=m,df_prior=df_prior,Prior=Prior,itol=itol,tol=tol,ρ=ρ)
    
#     for l= 1:nperm
#     maxLODs[l], H1par_perm=locoPermutation(cross,p,m,n,Y2,X1,XX.chr,est0,Λg,Xnul_t,df_prior,Prior;tol0=tol0,tol=tol,ρ=ρ)
#     H1par=[H1par;H1par_perm]
#              if (mod(l,100)==0)
#               println("Scan for $(l)th permutation is done.")
#              end
#     end
#     maxLODs=convert(Array{Float64,1},maxLODs)
#     cutoff= quantile(maxLODs,1.0.-pval)
     
#     return maxLODs, H1par, cutoff

# end







