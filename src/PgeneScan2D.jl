

## export functions
# export gene2Scan, marker2Scan!



function marker2Scan!(LODs,mindex::Array{Int64,1},q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1,ν₀,Ψ;ρ=0.001,tol0=1e-3,tol1=1e-4)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(q,2*(cross-1)))

              for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                      XX=@views vcat(Xnul_t,X1[j,2:end,:],X1[l,2:end,:])
                      B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                      est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                                   end
                 @views LODs[mindex[j+1:end],mindex[j]].=lod
               end

     else #cross=1
        B0=hcat(Nullpar.B,zeros(q,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                       XX=@views vcat(Xnul_t,X1[[j,l],:])
                       B0,τ2,Σ,loglik0 =ecmLMM(Y1,XX,Z1,B0,Nullpar.τ2,Nullpar.Σ,λg,λc,ν₀,Ψ;tol=tol0)
                       lod0=(loglik0-Nullpar.loglik)/log(10)
                       est1=ecmNestrvAG(lod0,kmin,Y1,XX,Z1,B0,τ2,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol1)
                      (est1.loglik-Nullpar.loglik)/log(10)
                               end
               @views LODs[mindex[j+1:end],mindex[j]].=lod
                end

   end #if cross
   # return LODs #[Lods, H1_parameters]
end

#MVLMM
function marker2Scan!(LODs,mindex::Array{Int64,1},m,kmin,cross,Nullpar::Result,λg,Y1,Xnul_t,X1,ν₀,Ψ;tol0=1e-3,tol1=1e-4,ρ=0.001)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(m,2*(cross-1)))

               for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                           XX=@views vcat(Xnul_t,X1[j,2:end,:],X1[l,2:end,:])
                           B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                            lod0=(loglik0-Nullpar.loglik)/log(10)
                           est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                           (est1.loglik-Nullpar.loglik)/log(10)
                                       end
               @views LODs[mindex[j+1:end],mindex[j]] .=lod
               end

        else #cross=1
          B0=hcat(Nullpar.B,zeros(m,2));

               for j=1:M-1
                lod=@distributed (vcat) for l=j+1:M
                         XX=@views vcat(Xnul_t,X1[[j,l],:])
                         B0,Vc,Σ,loglik0 = ecmLMM(Y1,XX,B0,Nullpar.Vc,Nullpar.Σ,λg,ν₀,Ψ;tol=tol0)
                             lod0=(loglik0-Nullpar.loglik)/log(10)
                         est1=ecmNestrvAG(lod0,kmin,Y1,XX,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol1,ρ=ρ)
                        (est1.loglik-Nullpar.loglik)/log(10)
                #  println([j l])
                                       end
                 @views LODs[mindex[j+1:end],mindex[j]] .=lod
                end

    end #if cross

end

####


function gene2Scan(cross::Int64,Tg,Tc::Array{Float64,2},Λg,λc::Array{Float64,1},
        Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;
        ρ=0.001,Xnul::Array{Float64,2}=ones(1,size(Y,2)), m=length(λc),
        df_prior=m+1,Prior::Matrix{Float64}=diagm(ones(m)),
        kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4)

    p=Int(size(XX.X,1)/cross);q=size(Z,2);
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);
           ## initialization
     init=initial(Xnul,Y,Z)
     if (λc!= ones(m))
         if (Prior!= diagm(ones(m)))
              Z1, Σ1, Ψ =transForm(Tc,Z,init.Σ,Prior)
           else # prior =I 
              Z1,Σ1 =  transForm(Tc,Z,init.Σ,true)
              Ψ =Prior
          end
          Y1= transForm(Tc,Y,init.Σ,false) # transform Y only by row (Tc)
       else
          Z1=Z; Σ1 = init.Σ
          Y1=Y;Ψ = Prior
      end
            
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        est0=[];
         for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
       @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,Tg[:,:,i])
                if (cross!=1)
       @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
       @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end

           est=nulScan(init,kmin,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
          marker2Scan!(LODs,maridx,q,kmin,cross,est,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
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

                  est0=nulScan(init,kmin,Λg,λc,Y1,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
             for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ)
             end
    end
    return LODs,est0
end

##MVLMM
function gene2Scan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,LOCO::Bool=false;m=size(Y,1),
                   Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,
                  Prior::Matrix{Float64}=diagm(ones(m)),kmin::Int64=1,itol=1e-4,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);
    Chr=unique(XX.chr); nChr=length(Chr); LODs=zeros(p,p);est0=[];

    #check the prior
    if (!isposdef(Prior))
      println("Error! Plug in a postivie definite Prior!")
   end
      #initialization
       init=initial(Xnul,Y,false)
       if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
            for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
             @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   Y,X=transForm(Tg[:,:,i],Y,X0[maridx,:,:],cross)
                   else
                   Y,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end

                   
                 est=nulScan(init,kmin,Λg[:,i],Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,m,kmin,cross,est,Λg[:,i],Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

        else #no LOCO
#            Xnul_t=Xnul*Tg';
           Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y,X=transForm(Tg,Y,X0,cross)
                   else
                   Y,X=transForm(Tg,Y,XX.X,cross)
                 end


             est0=nulScan(init,kmin,Λg,Y,Xnul_t,df_prior,Prior;itol=itol,tol=tol,ρ=ρ)
        for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,m,kmin,cross,est0,Λg,Y,Xnul_t,X,df_prior,Prior;tol0=tol0,tol1=tol,ρ=ρ)
        end

        end #LOCO
    return LODs,est0
end


#new version adding estimating Kc inside
function gene2Scan(cross::Int64,Tg,Λg,Y::Array{Float64,2},XX::Markers,Z::Array{Float64,2},LOCO::Bool=false;m=size(Y,1),
    Xnul::Array{Float64,2}=ones(1,size(Y,2)),df_prior=m+1,Prior::Matrix{Float64}=diagm(ones(m)),
   itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);q=size(Z,2);
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);

     ## picking up initial values for parameter estimation under the null hypothesis
     init= getKc(Y;Z=Z, df_prior=df_prior, Prior=Prior,Xnul=Xnul,itol=itol,tol=tol0,ρ=ρ)
     Tc, λc = K2eig(init.Kc) 

       
     if (Prior!= diagm(ones(m)))
        Y1,Σ1,Ψ= transForm(Tc,Y,init.Σ,Prior) # transform Y only by row (Tc)
         
      else # prior =I 
         Y1,Σ1 =  transForm(Tc,Y,init.Σ,true)
         Ψ =Prior
     end
             
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        est0=[];
         for i=1:nChr
                maridx=findall(XX.chr.==Chr[i]);
                @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul, Tg[:,:,i])
                 if (cross!=1) #individual-wise tranformation 
             @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,X0[maridx,:,:],cross)
                   else
             @fastmath @inbounds Y2,X1=transForm(Tg[:,:,i],Y1,XX.X[maridx,:],cross)
                 end

                 est=nulScan(init,kmin,Λg[:,i],λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ)
                 marker2Scan!(LODs,maridx,q,kmin,cross,est,Λg[:,i],λc,Y2,Xnul_t,X1,Z1,df_prior,Ψ;tol0=tol0,tol1=tol,ρ=ρ) 
                est0=[est0;est];
            end

     else #no LOCO

        Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
        if (cross!=1)
          Y1,X1=transForm(Tg,Y1,X0,cross)
          else
          Y1,X1=transForm(Tg,Y1,XX.X,cross)
        end
                est0=nulScan(init,kmin,Λg,λc,Y2,Xnul_t,Z1,Σ1,df_prior,Ψ;itol=itol,tol=tol,ρ=ρ) 
            
             for i=1:nChr
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             end
    end
    
    return LODs,est0
end

 