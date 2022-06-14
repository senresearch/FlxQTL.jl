

## export functions
# export gene2Scan, marker2Scan!



function marker2Scan!(LODs,mindex::Array{Int64,1},q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1;ρ=0.001,tol0=1e-3,tol1=1e-4)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(q,2*(cross-1)))

              for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                      XX=@views vcat(Xnul_t,X1[j,2:end,:],X1[l,2:end,:])
                      B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                      est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                                   end
                 @views LODs[mindex[j+1:end],mindex[j]].=lod
               end

     else #cross=1
        B0=hcat(Nullpar.B,zeros(q,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                       XX=@views vcat(Xnul_t,X1[[j,l],:])
                       B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                       est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                               end
               @views LODs[mindex[j+1:end],mindex[j]].=lod
                end

   end #if cross
   # return LODs #[Lods, H1_parameters]
end

#MVLMM
function marker2Scan!(LODs,mindex::Array{Int64,1},m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1;tol0=1e-3,tol1=1e-4,ρ=0.001)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(m,2*(cross-1)))

               for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                           XX=@views vcat(Xnul_t,X1[j,2:end,:],X1[l,2:end,:])
                           B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                            lod0=(loglik0-Nullpar.loglik)/log(10)
                           est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;tol=tol1,ρ=ρ)
                           (est1.loglik-Nullpar.loglik)/log(10)
                                       end
               @views LODs[mindex[j+1:end],mindex[j]] .=lod
               end

        else #cross=1
          B0=hcat(Nullpar.B,zeros(m,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                         XX=@views vcat(Xnul_t,X1[[j,l],:])
                         B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                             lod0=(loglik0-Nullpar.loglik)/log(10)
                         est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;tol=tol1,ρ=ρ)
                        (est1.loglik-Nullpar.loglik)/log(10)
                #  println([j l])
                                       end
                 @views LODs[mindex[j+1:end],mindex[j]] .=lod
                end

    end #if cross

end



"""

    gene2Scan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},
        Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;ρ=0.001,Xnul::Array{Float64,2}=ones(1,size(Y0,2))
        ,itol=1e-4,tol0=1e-3,tol::Float64=1e-4)
    gene2Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;
                      Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)


Implement 2d-genome scan with/without LOCO (Leave One Chromosome Out). Note that the second `gene2Scan()` is based on a conventional MLMM:
```math
vec(Y) \\sim MVN((Z \\otimes X)vec(B) (or XBZ') , K \\otimes \\Sigma_1 +I \\otimes \\Sigma_2),
```
 where `K` is a genetic kinship, ``\\Sigma_1, \\Sigma_2`` are covariance matrices for
random and error terms, respectively.  `Z` can be replaced with an identity matrix.

# Arguments

- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Tg` : A n x n matrix of eigenvectors from [`K2eig`](@ref), or [`K2Eig`](@ref).
       Returns 3d-array of eigenvectors as many as Chromosomes if `LOCO` is true.
- `Tc` : A m x m matrix of eigenvectors from climatic relatedness matrix.
- `Λg` : A n x 1 vector of eigenvalues from kinship. Returns a matrix of eigenvalues if `LOCO` is true.
- `λc` : A m x 1 vector of eigenvalues from climatic relatedness matrix. Use `ones(m)` for no climatic information added.
- `Y0` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[1,:]` (a vector) -> `Y[[1],:]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z0` :  An optional m x q matrix of low-dimensional phenotypic covariates, i.e. contrasts, basis functions (fourier, wavelet, polynomials, B-splines, etc.).
      If nothing to insert in `Z0`, just insert an identity matrix, `Matrix(1.0I,m,m)`.  m traits x q phenotypic covariates.
- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out).

## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's).  Unless adding covariates, just leave as it is.  See [`geneScan`](@ref).
- `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` :  A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.

!!! Note

- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`. It works in most cases. If not,
    can reduce both `tol0` and `tol` to `1e-4` or further.


# Output

- `LODs` : LOD scores. Can change to ``- \\log_{10}{P}`` using [`lod2logP`](@ref).
- `est0` : A type of `EcmNestrv.Approx` including parameter estimates under H0: no QTL.

"""
function gene2Scan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},
        Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;ρ=0.001,Xnul::Array{Float64,2}=ones(1,size(Y0,2))
        ,kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4)

    p=Int(size(XX.X,1)/cross);q=size(Z0,2);
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);
           ## initialization
            init=initial(Xnul,Y0,Z0)
            Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
            Y1= transForm(Tc,Y0,init.Σ,false) # transform Y only by row (Tc)
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        est0=[];
         for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
                   Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end

                  est=nulScan(init,kmin,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,q,kmin,cross,est,Λg[:,i],λc,Y2,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

     else #no LOCO
#             Xnul_t=Xnul*Tg';
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   Y1,X1=transForm(Tg,Y1,X0,cross)
                   else
                   Y1,X1=transForm(Tg,Y1,XX.X,cross)
                 end

                  est0=nulScan(init,kmin,Λg,λc,Y1,Xnul_t,Z1,Σ1;itol=itol,tol=tol,ρ=ρ)
             for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             end
    end
    return LODs,est0
end

##MVLMM
function gene2Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;
                      Xnul::Array{Float64,2}=ones(1,size(Y0,2)),kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);m=size(Y0,1);
    Chr=unique(XX.chr); nChr=length(Chr); LODs=zeros(p,p);est0=[];
      #initialization
       init=initial(Xnul,Y0,false)
       if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
            for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
             @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   Y,X=transForm(Tg[:,:,i],Y0,X0[maridx,:,:],cross)
                   else
                   Y,X=transForm(Tg[:,:,i],Y0,XX.X[maridx,:],cross)
                 end

                   est=nulScan(init,kmin,Λg[:,i],Y,Xnul_t;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,m,kmin,cross,est,Λg[:,i],Y,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

        else #no LOCO
#            Xnul_t=Xnul*Tg';
           Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y,X=transForm(Tg,Y0,X0,cross)
                   else
                   Y,X=transForm(Tg,Y0,XX.X,cross)
                 end


             est0=nulScan(init,kmin,Λg,Y,Xnul_t;itol=itol,tol=tol,ρ=ρ)
        for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,m,kmin,cross,est0,Λg,Y,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
        end

        end #LOCO
    return LODs,est0
end


#new version adding estimating Kc inside
function gene2Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
        ρ=0.001,Xnul::Array{Float64,2}=ones(1,size(Y0,2))
        ,kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4)

    p=Int(size(XX.X,1)/cross);q=size(Z0,2);m=size(Y0,1);
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);
           ## initialization
            init=initial(Xnul,Y0,Z0,false)#type: Init0

         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        est0=[];
         for i=1:nChr
             λc, Y1, Xnul_t,Z1,init=updateKc(m,init,Tg[:,:,i],Λg[:,i],Y0,Z0,Xnul;itol=itol,tol=tol,ρ=ρ) #type: Init1 for H0 estimates
                maridx=findall(XX.chr.==Chr[i]);
                
                 if (cross!=1) #individual-wise tranformation 
                   X1=transForm(Tg[:,:,i],X0[maridx,:,:],cross)
                   else
                   X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end

                est=nulScan(init,kmin,Λg[:,i],λc,Y1,Xnul_t,Z1;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,q,kmin,cross,est,Λg[:,i],λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

     else #no LOCO
         λc, Y1, Xnul_t,Z1,init=updateKc(m,init,Tg,Λg,Y0,Z0,Xnul;itol=itol,tol=tol,ρ=ρ)
                 if (cross!=1) #individual-wise tranformation 
                   X1=transForm(Tg,X0,cross)
                   else
                   X1=transForm(Tg,XX.X,cross)
                 end

                  est0=nulScan(init,kmin,Λg,λc,Y1,Xnul_t,Z1;itol=itol,tol=tol,ρ=ρ)
             for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             end
    end
    return LODs,est0
end

 