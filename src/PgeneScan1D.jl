
## marker1Scan : CPU 1D-genome scanning under H1 only (with/without loco)
function marker1Scan(q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,q));

        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
            lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                 end

    else # cross>1
        ## scanning genotype probabilities

            B0=hcat(Nullpar.B,zeros(Float64,q,cross-1))

          lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                  lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end

#Z=I
function marker1Scan(m,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4,nchr=0)

        nmar=size(X1,1);
    if (cross==1) ## scanning genotypes
        B0=hcat(Nullpar.B,zeros(Float64,m));
#      f= open(homedir()*"/GIT/fmulti-lmm/result/test_ecmlmm.txt","w")
        lod=@distributed (vcat) for j=1:nmar
            XX=vcat(Xnul_t,@view X1[[j],:])
        B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                lod0= (loglik0-Nullpar.loglik)/log(10)
        est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1,numChr=nchr,nuMarker=j)
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
                XX=vcat(Xnul_t, @view X1[j,2:end,:])
                B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
            [(est1.loglik-Nullpar.loglik)/log(10) est1]
                                  end

     end

    return lod[:,1],lod[:,2]
end



##MVLMM
function marker1Scan(m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4)

        nmar=size(X1,1);
    if (cross==1)
        B0=hcat(Nullpar.B,zeros(m))

             lod=@distributed (vcat) for j=1:nmar
               XX= vcat(Xnul_t,@view X1[[j],:])
               B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                     lod0= (loglik0-Nullpar.loglik)/log(10)
               est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;ρ=ρ,tol=tol1)
               [(est1.loglik-Nullpar.loglik)/log(10) est1]
                           end

    else #cross>1

        B0=hcat(Nullpar.B,zeros(m,cross-1))

        lod=@distributed (vcat) for j=1:nmar
                XX= vcat(Xnul_t,@view X1[j,2:end,:])
            B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                 lod0= (loglik0-Nullpar.loglik)/log(10)
                est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                     [(est1.loglik-Nullpar.loglik)/log(10) est1]
                          end

    end
    return lod[:,1], lod[:,2]

end


######### actual two genescan versions including prior
function geneScan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},Y0::Array{Float64,2},
        XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
                Xnul::Array{Float64,2}=ones(1,size(Y0,2)),df_prior=length(λc)+1,
                Prior::Matrix{Float64}=diagm(ones(df_prior-1)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        m=size(Y0,1);
        q=size(Z0,2);  p=Int(size(XX.X,1)/cross); 

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0,Z0)
          if (λc!= ones(m))
            if (Prior!= diagm(ones(m)))
                Z1, Σ1, Ψ =transForm(Tc,Z0,init.Σ,Prior)
             else # prior =I 
                Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
                Ψ =Prior
            end
            
            Y1= transForm(Tc,Y0,init.Σ) # transform Y only by row (Tc)

           else
            Z1=Z0; Σ1 = init.Σ
            Y1=Y0;Ψ = Prior
         end
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

           for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
                 if (cross!=1)
   @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
  @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
           # rearrange B into 3-d array
           B = arrngB(H1par,size(Xnul,1),q,p,cross)

        else #no LOCO
#          Xnul_t=Xnul*Tg';
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   Y1,X1=transForm(Tg,Y1,X0,cross)
                   else
                   Y1,X1=transForm(Tg,Y1,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)
    end

    # Output choice
    if (tdata) # should use with no LOCO to do permutation
        return est0,Xnul_t,Y1,X1,Z1
    elseif (LogP) # transform LOD to -log10(p-value)
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
function geneScan(cross::Int64,Tg::Union{Array{Float64,3},Array{Float64,2}},Tc::Array{Float64,2},Λg::Union{Array{Float64,2},Array{Float64,1}},λc::Array{Float64,1},Y0::Array{Float64,2},
        XX::Markers,LOCO::Bool=false;LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y0,2)),df_prior=length(λc)+1,
        Prior::Matrix{Float64}=diagm(ones(df_prior-1)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

         m=size(Y0,1);
         p=Int(size(XX.X,1)/cross);
         
         #check the prior
         if (!isposdef(Prior))
            println("Error! Plug in a postivie definite Prior!")
         end
        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0)

         if(λc!= ones(m))
            if (Prior!= diagm(ones(m)))
                Y1,Σ1,Ψ= transForm(Tc,Y0,init.Σ,Prior) # transform Y only by row (Tc)
            else #prior =I
                Y1,Σ1 =  transForm(Tc,Y0,init.Σ,true)
                Ψ =Prior
            end
           else
            Σ1 =init.Σ
            Y1=Y0;Ψ =Prior
         end

         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

           for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
   @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,Tg[:,:,i])
                 if (cross!=1)
      @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
      @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
           # rearrange B into 3-d array
           B = arrngB(H1par,size(Xnul,1),m,p,cross)

        else #no LOCO
#          Xnul_t=Xnul*Tg';
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   Y1,X1=transForm(Tg,Y1,X0,cross)
                   else
                   Y1,X1=transForm(Tg,Y1,XX.X,cross)
                 end

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),m,p,cross)
    end


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
function geneScan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,LOCO::Bool=false;tdata::Bool=false,LogP::Bool=false,
        Xnul::Array{Float64,2}=ones(1,size(Y0,2)),df_prior=length(λc)+1,
        Prior::Matrix{Float64}=diagm(ones(df_prior-1)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    m=size(Y0,1);
    p=Int(size(XX.X,1)/cross);

    #check the prior
    if (!isposdef(Prior))
        println("Error! Plug in a postivie definite Prior!")
     end

     #initialization
       init=initial(Xnul,Y0,false)
        if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p); Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

       for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
          @fastmath @inbounds Y,X=transForm(Tg[:,:,i],Y0,X0[maridx,:,:],cross)
                   else
           @fastmath @inbounds Y,X=transForm(Tg[:,:,i],Y0,XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                    est00=nulScan(init,1,Λg[:,i],Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
                lods, H1par1=marker1Scan(m,1,cross,est00,Λg[:,i],Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
                LODs[maridx].=lods
                H1par=[H1par;H1par1]
                est0=[est0;est00];
            end
            # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     else #no loco
#             Xnul_t=Xnul*Tg';
             Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y,X=transForm(Tg,Y0,X0,cross)
                   else
                   Y,X=transForm(Tg,Y0,XX.X,cross)
                 end


                  est0=nulScan(init,1,Λg,Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
            LODs,H1par=marker1Scan(m,1,cross,est0,Λg,Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
             B = arrngB(H1par,size(Xnul,1),m,p,cross)
     end

    if (tdata) # should use with no LOCO
        return est0,Xnul_t,Y,X
    elseif (LogP) # transform LOD to -log10(p-value)
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


## estimating Kc + prior
function geneScan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
    tdata::Bool=false,LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y0,2)),m=size(Y0,1),df_prior=m+1,
    Prior::Matrix{Float64}=diagm(ones(df_prior-1)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    
    q=size(Z0,2);  p=Int(size(XX.X,1)/cross);

    ## picking up initial values for parameter estimation under the null hypothesis
        init= getKc(Y0;m=m,Z0=diagm(ones(m)), df_prior=df_prior, Prior=Prior,Xnul=Xnul,itol=1e-2,tol=1e-3,ρ=ρ)
        Tc, λc = K2eig(init.Kc) 

        
          if (Prior!= diagm(ones(m)))
              Z1, Σ1, Ψ =transForm(Tc,Z0,init.Σ,Prior)
           else # prior =I 
              Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
              Ψ =Prior
          end
          
          Y1= transForm(Tc,Y0,init.Σ) # transform Y only by row (Tc)

         
       
       if (cross!=1)
          X0=mat2array(cross,XX.X)
       end
  if (LOCO)
      LODs=zeros(p);
      Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

         for i=1:nChr
              maridx=findall(XX.chr.==Chr[i])
#                 Xnul_t=Xnul*Tg[:,:,i]';
 @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
               if (cross!=1)
 @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                 else
@fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
               end
              #parameter estimation under the null
                est00=nulScan(init,1,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;ρ=ρ,itol=itol,tol=tol)
              lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
              LODs[maridx]=lods
              H1par=[H1par;H1par1]
              est0=[est0;est00];
          end
         # rearrange B into 3-d array
         B = arrngB(H1par,size(Xnul,1),q,p,cross)

      else #no LOCO
#          Xnul_t=Xnul*Tg';
          Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
               if (cross!=1)
                 Y1,X1=transForm(Tg,Y1,X0,cross)
                 else
                 Y1,X1=transForm(Tg,Y1,XX.X,cross)
               end

                est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
              LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
           # rearrange B into 3-d array
        B = arrngB(H1par,size(Xnul,1),q,p,cross)
  end

  # Output choice
  if (tdata) # should use with no LOCO to do permutation
      return est0,Xnul_t,Y1,X1,Z1
  elseif (LogP) # transform LOD to -log10(p-value)
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


#pre-estimate Kc 
struct InitKc
   Kc::Matrix{Float64} 
   B::Matrix{Float64}
   Σ::Matrix{Float64}
   τ2::Float64
end


function getKc(Y0::Array{Float64,2};m=size(Y0,1),Z0=diagm(ones(m)), df_prior=m+1,
    Prior::Matrix{Float64}=diagm(ones(df_prior-1)),
    Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-2,tol::Float64=1e-3,ρ=0.001)
    
    if(Z0!=diagm(ones(m)))
        init0=initial(Xnul,Y0,Z0,false)
     else #Z0=I
        init0=initial(Xnul,Y0,false)
     end

    est0= nul1Scan(init0,1,Y0,Xnul,Z0,m,df_prior,Prior;ρ=ρ,itol=itol,tol=tol)
      τ² mean(Diagonal(est0.Σ))
    return InitKc(est0.Vc, est0.B, est0.Σ, τ²)

end




