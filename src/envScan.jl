
#pick up a marker to be included for envrionmental scan
function pikGeno(midx::Array{Int64,1},XX::Markers,cross::Int64)

    if (cross!=1)
        X0=mat2array(cross,XX.X)
        X0=X0[midx,:,:]
    else
        X0=XX.X[midx,:]
    end
    return X0
end



"""


      envScan(Midx::Array{Int64,1},cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},
        Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)


Implement environment scan conditional on a genetic marker of interest (QTL).  Each of `q` trait covariate data is scanned
(regressed) given a major QTL selected from genome scan, `geneScan` to obtain LOD scores.

# Arguments

- `Midx` : A vector of genetic marker indices (major QTL) selected based on LOD scores from [`geneScan`](@ref).
- `cross` : An integer indicating the number of alleles or genotypes. Ex. 2 for RIF, 4 for four-way cross, 8 for HS mouse (allele probabilities), etc.
          This value is related to degree of freedom when doing genome scan.
- `Tg` : A n x n matrix of eigenvectors from [`K2eig`](@ref), or [`K2Eig`](@ref).
       Returns 3d-array of eigenvectors as many as Chromosomes if `LOCO` is true.
- `Tc` : A m x m matrix of eigenvectors from the precomputed covariance matrix of `Kc` under the null model of no QTL.
- `Λg` : A n x 1 vector of eigenvalues from kinship. Returns a matrix of eigenvalues if `LOCO` is true.
- `λc` : A m x 1 vector of eigenvalues from `Kc`. 
- `Y0` : A m x n matrix of response variables, i.e. m traits (or environments) by n individuals (or lines). For univariate phenotypes, use square brackets in arguement.
        i.e. `Y0[1,:]` (a vector) ->`Y[[1],:]` (a matrix) .
- `XX` : A type of [`Markers`](@ref).
- `Z0` :  A m x q matrix of low-dimensional trait covariate data for environment scan, i.e. minimum or maximum monthly temperature data, monthly photoperiod data, etc.

- `LOCO` : Boolean. Default is `false` (no LOCO). Runs genome scan using LOCO (Leave One Chromosome Out).


## Keyword Arguments

- `Xnul` :  A matrix of covariates. Default is intercepts (1's).  Unless plugging in particular covariates, just leave as it is.
- `itol` :  A tolerance controlling ECM (Expectation Conditional Maximization) under H0: no QTL. Default is `1e-3`.
- `tol0` :  A tolerance controlling ECM under H1: existence of QTL. Default is `1e-3`.
- `tol` : A tolerance of controlling Nesterov Acceleration Gradient method under both H0 and H1. Default is `1e-4`.
- `ρ` : A tunning parameter controlling ``\\tau^2``. Default is `0.001`.

!!! Note

- When some LOD scores return negative values, reduce tolerences for ECM to `tol0 = 1e-4`. It works in most cases. If not,
    can reduce both `tol0` and `tol` to `1e-4` or further.


# Output

- `LODs` : A vector of LOD scores by envrionment scan when including each major QTL.
- `B` : A 3-d array of `B` (fixed effects) matrices under H1: existence of an environment factor (covariate)
         conditional on a major QTL.
- `est0` : A type of `EcmNestrv.Approx` including parameter estimates under H0: no environment factor
           conditional on a major QTL.

"""
function envScan(Midx::Array{Int64,1},cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},
        Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        m,n=size(Y0); M=length(Midx)
        q=size(Z0,2);   LODs=zeros(q,M); est0=[];H1par=[]
        ##pick up markers to be included for environmental scan
          Xp= pikGeno(Midx,XX,cross)

        # transformation by environments (Tc)
            Z1=  transForm(Tc,hcat(ones(m,1),Z0),Matrix(1.0I,m,m))
            Y1= transForm(Tc,Y0,Matrix(1.0I,m,m))
 if (LOCO)
             Chr=unique(XX.chr)
             chr= XX.chr[Midx]
         for j in eachindex(Midx)
              cidx=findall(Chr.==chr[j])
     # getting initial values for estimation under H0 (Z=ones(m,1)) & transformation by individuals (Tg)
             if (cross!=1)
                   init=initial(vcat(Xnul,@view Xp[j,2:end,:]),Y0,ones(m,1))
                   Σ1=convert(Array{Float64,2},Symmetric(BLAS.symm('R','U',init.Σ,Tc)*Tc'))
                   Y1,X1=transForm(Tg[:,:,cidx[1]],Y1,vcat(Xnul,@view Xp[j,2:end,:]),1)

                  else
                   init=initial(vcat(Xnul,@view Xp[[j],:]),Y0,ones(m,1))
                   Σ1=convert(Array{Float64,2},Symmetric(BLAS.symm('R','U',init.Σ,Tc)*Tc'))
                   Y1,X1=transForm(Tg[:,:,cidx[1]],Y1,vcat(Xnul,@view Xp[[j],:]),1)
             end

        # environmental scan
              est00=nulScan(init,kmin,Λg[:,cidx[1]],λc,Y1,X1,Z1[:,[1]],Σ1;itol=itol,tol=tol,ρ=ρ)
              LODs[:,j],H1par1=env1Scan(q,kmin,est00,Λg[:,cidx[1]],λc,Y1,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
              est0=[est0;est00]; H1par=[H1par;H1par1]
        end #for


     else   # no LOCO
             #pre-allocation
               if(cross!=1)
                   X=zeros(cross,n)
                 else
                   X=zeros(cross+1,n)
               end

                Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                Y1,X1=transForm(Tg,Y1,Xp,cross)
         for j in eachindex(Midx)
             if (cross!=1)
                    X[:,:] = vcat(Xnul_t,@view X1[j,2:end,:])
                    init=initial(X,Y0,ones(m,1))
                    Σ1=convert(Array{Float64,2},Symmetric(BLAS.symm('R','U',init.Σ,Tc)*Tc'))
                    est00=nulScan(init,kmin,Λg,λc,Y1,X,Z1[:,[1]],Σ1;itol=itol,tol=tol,ρ=ρ)
                    LODs[:,j], H1par1=env1Scan(q,kmin,est00,Λg,λc,Y1,X,Z1;tol0=tol0,tol1=tol,ρ=ρ)
                 else
                    X[:,:]= vcat(Xnul_t,@view X1[[j],:])
                    init=initial(vcat(Xnul,@view Xp[[j],:]),Y0,ones(m,1))
                    Σ1=convert(Array{Float64,2},Symmetric(BLAS.symm('R','U',init.Σ,Tc)*Tc'))
                    est00=nulScan(init,kmin,Λg,λc,Y1,X,Z1[:,[1]],Σ1;itol=itol,tol=tol,ρ=ρ)
                    LODs[:,j], H1par1=env1Scan(q,kmin,est00,Λg,λc,Y1,X,Z1;tol0=tol0,tol1=tol,ρ=ρ)
            end #end cross
                   est0=[est0;est00]; H1par=[H1par;H1par1]
        end #for
 end #LOCO

             # rearrange B into 3-d array
          B = arrngB(H1par)

        return LODs,B,est0

end



## env1Scan : CPU environmental scan per significant locus under H1 only
function env1Scan(q,kmin,Nullpar::Approx,λg,λc,Y1,X1,Z1;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)


         p=size(X1,1);
        B0=vcat(Nullpar.B,zeros(1,p));

        lod=@distributed (vcat)  for j=1:q
         B0,τ2,Σ,loglik0 =ecmLMM(Y1,X1,Z1[:,[1,j+1]],B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                  lod0=(loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,X1,Z1[:,[1,j+1]],B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    return lod[:,1],lod[:,2]
end


##rearrange Bs estimated under H1 into 3-d array
## dimensions are from Z & X, size(Z)=(m,q), size(X)=(p,n)
## p1= size(Xnul,1) : Xnul may or may not include covariates. default is ones(1,n)
## H1par : 1-d array including parameter estimates under H1 enclosed by EcmNestrv.Approx
function arrngB(H1par)


    Q=length(H1par); q,p = size(H1par[1].B)
    B=zeros(q,p,Q)

    @inbounds @views for j=1:Q
            B[:,:,j]=H1par[j].B
        end
    return B
end
# export pikGeno,env1Scan,envScan,arrngB
