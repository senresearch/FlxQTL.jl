
## marker1Scan : CPU 1D-genome scanning under H1 only (with/without loco)
function marker1Scan(nmar,q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        # nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,q));

        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
            lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    else # cross>1
        ## scanning genotype probabilities

            B0=hcat(Nullpar.B,zeros(Float64,q,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t, @view X1[2:end,:,j])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                  lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end

#Z=I
function marker1Scan(nmar,m,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        # nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,m));
#      f= open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","w")
        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
#             f=open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","a")
#               writedlm(f,[loglik0 est1.loglik Nullpar.loglik])
#             close(f)
                 end

    else # cross>1
        ## scanning genotype probabilities

        #initialize B under the alternative hypothesis
        B0=hcat(Nullpar.B,zeros(Float64,m,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX=vcat(Xnul_t, @view X1[2:end,:,j])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end


##MVLMM
function marker1Scan(nmar,m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1;ρ=0.001,tol0=1e-3,tol1=1e-4)

        # nmar=size(X1,1);
    if (cross==1)
        B0=hcat(Nullpar.B,zeros(m))

             lod=@distributed (vcat) for j=1:nmar
               XX= vcat(Xnul_t,@view X1[[j],:])
               B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                     lod0= (loglik0-Nullpar.loglik)/log(10)
               est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;ρ=ρ,tol=tol1)
               [(est1.loglik-Nullpar.loglik)/log(10) est1]
                           end

    else #cross>1

        B0=hcat(Nullpar.B,zeros(m,cross-1))

        lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t,@view X1[2:end,:,j])
            B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg;tol=tol1,ρ=ρ)
                     [(est1.loglik-Nullpar.loglik)/log(10) est1]
                          end

    end
    return lod[:,1], lod[:,2]

end

########## Kc estimation

function transByTrait(m,Tc,Y,Z,init::Result)

#  if (λc!= ones(m))
        
        if (Z != diagm(ones(m)))
            Z1,Σ1 =  transForm(Tc,Z,init.Σ,true)
        else #Z =I 
            Σ1 =  transForm(Tc,init.Σ,Z)
            Z1=Z
        end
        
        Y1= transForm(Tc,Y,init.Σ) # transform Y only by row (Tc)

    return Y1,Z1,Σ1
end

function nul1Scan(init::Init0,kmin,λg,Y,Xnul,Z,m;ρ=0.001,itol=1e-3,tol=1e-4)
       
      # n=size(Y,2); 

    if (Z!=diagm(ones(m)))   
        B0,Kc_0,Σ1,_=ecmLMM(Y,Xnul,Z,init.B,init.Vc,init.Σ,λg;tol=itol)
        nulpar=NestrvAG(kmin,Y,Xnul,Z,B0,Kc_0,Σ1,λg;tol=tol,ρ=ρ)
        
       else #Z=I
        nulpar = nulScan(init,kmin,λg,Y,Xnul;ρ=ρ,itol=itol,tol=tol)
     end
    return nulpar #Result
end

struct TNul
Y::Matrix{Float64}
Xnul::Matrix{Float64}
Z::Matrix{Float64}
Σ::Matrix{Float64}
end

#H0 MVLMM for Kc estimation w/o prior
function getKc(init::Init0,Y::Array{Float64,2},Tg::Matrix{Float64},λg::Vector{Float64};m=size(Y,1),Z=diagm(ones(m)),
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),itol=1e-2,tol::Float64=1e-3,ρ=0.001)
     
     Y1,Xnul_t = transForm(Tg,Y,Xnul,1) #null model transformation
     
 
     est0= nul1Scan(init,1,λg,Y1,Xnul_t,Z,m;ρ=ρ,itol=itol,tol=tol)
      Tc, λc = K2eig(est0.Vc)
     
      #trait-wise transformation
      Y2,Z1,Σ1 = transByTrait(m,Tc,Y1,Z,est0)
      τ² = 1.0
    
    return λc, TNul(Y2,Xnul_t,Z1,Σ1),InitKc(est0.Vc,est0.B,est0.Σ,τ²,est0.loglik)
 
 end
 


function gene1Scan(Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},cross::Int64,LOCO::Bool=false;H0_up::Bool=false,
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
      
        q=size(Z,2);  p=Int(size(XX.X,1)/cross);m=size(Y,1)

        ## picking up initial values for parameter estimation under the null hypothesis
           if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)
      else #Z0=I
         init0=initial(Xnul,Y,false)
      end
    
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end

    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);est0=[];H1par=[]

           for i=eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
        λc, T0,init = getKc(init0,Y,Tg[:,:,i],Λg[:,i];Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
                 if (cross!=1)
   @fastmath @inbounds X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                   else
  @fastmath @inbounds X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,H0_up;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(nmar,q,1,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1,T0.Z;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]

                if (H0_up)
               est0 =[est0;Result(est00.B,est00.τ2*init.Kc,est00.Σ,est00.loglik)] #high dimensional traits
              else
               est0=[est0;Result(init.B,init.Kc,init.Σ,init.loglik)];
              end
            end

        else #no LOCO
            λc,T0,init = getKc(init0,Y,Tg,Λg;Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)
           
                 if (cross!=1)
                   X1=transForm(Tg,X0,cross)
                   else
                   X1=transForm(Tg,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,H0_up;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(p,q,1,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1,T0.Z;tol0=tol0,tol1=tol,ρ=ρ)
                
                if (H0_up)
                    est0 = Result(est0.B,est0.τ2*init.Kc,est0.Σ,est0.loglik)
                else
                    est0= Result(init.B,init.Kc,init.Σ,init.loglik)
                end
        
    end

         # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)

    # Output choice
   
    if (LogP) # transform LOD to -log10(p-value)
            if(LOCO)
                df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
            else
                df= prod(size(B[:,:,1]))-prod(size(est0.B))
            end
             logP=lod2logP(LODs,df)

        return logP,B,est0
     else
         return LODs,B,est0
     end
end

#Z=I
function geneScan1(cross::Int64,Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;H0_up::Bool=false,
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)
      
         p=Int(size(XX.X,1)/cross);m=size(Y,1)

        
         init0=initial(Xnul,Y,false)
     
    
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end

    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);est0=[];H1par=[]

           for i=eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)
        λc, T0,init = getKc(init0,Y,Tg[:,:,i],Λg[:,i];Xnul=Xnul,m=m,itol=itol,tol=tol,ρ=ρ)
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
                 if (cross!=1)
   @fastmath @inbounds X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                   else
  @fastmath @inbounds X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Σ,H0_up;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(nmar,m,1,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]

                if (H0_up)
               est0 =[est0;Result(est00.B,est00.τ2*init.Kc,est00.Σ,est00.loglik)] #high dimensional traits
              else
               est0=[est0;Result(init.B,init.Kc,init.Σ,init.loglik)];
              end
            end

        else #no LOCO
            λc,T0,init = getKc(init0,Y,Tg,Λg;Xnul=Xnul,m=m,itol=itol,tol=tol,ρ=ρ)
           
                 if (cross!=1)
                   X1=transForm(Tg,X0,cross)
                   else
                   X1=transForm(Tg,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,T0.Y,T0.Xnul,T0.Σ,H0_up;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(p,m,1,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1;tol0=tol0,tol1=tol,ρ=ρ)
                 
                if (H0_up)
                    est0 = Result(est0.B,est0.τ2*init.Kc,est0.Σ,est0.loglik)
                else
                    est0= Result(init.B,init.Kc,init.Σ,init.loglik)
                end
        
    end

         # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),m,p,cross)

    # Output choice
   
    if (LogP) # transform LOD to -log10(p-value)
            if(LOCO)
                df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
            else
                df= prod(size(B[:,:,1]))-prod(size(est0.B))
            end
             logP=lod2logP(LODs,df)

        return logP,B,est0
     else
         return LODs,B,est0
     end
end

##MVLMM
function geneScan1(Tg,Λg,Y::Array{Float64,2},XX::Markers,cross::Int64,LOCO::Bool=false;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
                LogP::Bool=false,itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

   
    p=Int(size(XX.X,1)/cross);m=size(Y,1)

     #initialization
       init=initial(Xnul,Y,false)
        if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p); Chr=unique(XX.chr);est0=[];H1par=[]

       for i=eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);nmar=length(maridx)

              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
          @fastmath @inbounds Y1,X=transForm(Tg[:,:,i],Y,X0[:,:,maridx],cross)
                   else
           @fastmath @inbounds Y1,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                    est00=nulScan(init,1,Λg[:,i],Y1,Xnul_t;itol=itol,tol=tol,ρ=ρ)
                lods, H1par1=marker1Scan(nmar,m,1,cross,est00,Λg[:,i],Y1,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
                LODs[maridx].=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
         
     else #no loco

             Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y1,X=transForm(Tg,Y,X0,cross)
                   else
                   Y1,X=transForm(Tg,Y,XX.X,cross)
                 end


                  est0=nulScan(init,1,Λg,Y1,Xnul_t;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(p,m,1,cross,est0,Λg,Y1,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
           
     end

    # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     if (LogP) # transform LOD to -log10(p-value)
          if(LOCO)
                df= prod(size(B[:,:,1]))-prod(size(est0[1].B))
            else
                df= prod(size(B[:,:,1]))-prod(size(est0.B))
            end
               logP=lod2logP(LODs,df)

        return logP,B,est0
     else
         return LODs,B,est0
     end
end







