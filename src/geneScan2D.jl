
function marker2Scan!(LODs,mindex::Array{Int64,1},q,kmin,cross,Nullpar::Approx,λg,λc,Y1,Xnul_t,X1,Z1;ρ=0.001,tol0=1e-3,tol1=1e-4)
    M=length(mindex)
    if (cross!=1)

        B0=hcat(Nullpar.B,zeros(q,2*(cross-1)))

              for j=1:M-1
               lod=@distributed (vcat) for l=j+1:M
                      XX=@views vcat(Xnul_t,X1[2:end,:,j],X1[2:end,:,l])
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
                           XX=@views vcat(Xnul_t,X1[2:end,:,j],X1[2:end,:,l])
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




function gene2Scan(Tg::Union{Array{Float64,3},Matrix{Float64}},Λg::Union{Matrix{Float64},Vector{Float64}},
          Y::Array{Float64,2},XX::Markers,LOCO::Bool,cross::Int64;m=size(Y,1),Z=diagm(ones(m)),
          Xnul::Array{Float64,2}=ones(1,size(Y,2)),itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)   

    p=Int(size(XX.X,1)/cross);q=size(Z,2);kmin=1;
    LODs=zeros(p,p);  Chr=unique(XX.chr);

           ## initialization
           if (Z!=diagm(ones(m)))
         init0=initial(Xnul,Y,Z,false)
      else #Z=I
         init0=initial(Xnul,Y,false)
      end

         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end

    if (LOCO)
        est0=[];
         for i = eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);
       λc, T0,init = getKc(init0,Y,Tg[:,:,i],Λg[:,i];Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)
              @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   X1=transForm(Tg[:,:,i],X0[:,:,maridx],cross)
                   else
                   X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end

                est00=nulScan(init,kmin,Λg[:,i],λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,true;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,q,kmin,cross,est00,Λg[:,i],λc,T0.Y,T0.Xnul,X1,T0.Z;tol0=tol0,tol1=tol,ρ=ρ)
                est0 =[est0;Result(est00.B,est00.τ2*init.Kc,est00.Σ,est00.loglik)] 
            end

     else #no LOCO

       λc,T0,init = getKc(init0,Y,Tg,Λg;Xnul=Xnul,m=m,Z=Z,itol=itol,tol=tol,ρ=ρ)
            Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                 if (cross!=1)
                   X1=transForm(Tg,X0,cross)
                   else
                   X1=transForm(Tg,XX.X,cross)
                 end

           est0=nulScan(init,kmin,Λg,λc,T0.Y,T0.Xnul,T0.Z,T0.Σ,true;itol=itol,tol=tol,ρ=ρ)

             for i=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,q,kmin,cross,est0,Λg,λc,T0.Y,T0.Xnul,X1,T0.Z;tol0=tol0,tol1=tol,ρ=ρ)
             end
           est0 = Result(est0.B,est0.τ2*init.Kc,est0.Σ,est0.loglik)
    end
    return LODs,est0
end

##MVLMM
function geneScan2(Tg,Λg,Y::Array{Float64,2},XX::Markers,LOCO::Bool,cross::Int64;Xnul::Array{Float64,2}=ones(1,size(Y,2)),
    itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

    p=Int(size(XX.X,1)/cross);kmin=1;m=size(Y,1)
    Chr=unique(XX.chr);  LODs=zeros(p,p);est0=[];
      #initialization
       init=initial(Xnul,Y,false)
       if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
            for i=eachindex(Chr)
                maridx=findall(XX.chr.==Chr[i]);
#                 Xnul_t=Xnul*Tg[:,:,i]';
             @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                if (cross!=1)
                   Y1,X=transForm(Tg[:,:,i],Y,X0[:,:,maridx],cross)
                   else
                   Y1,X=transForm(Tg[:,:,i],Y,XX.X[maridx,:],cross)
                 end

                   est=nulScan(init,kmin,Λg[:,i],Y1,Xnul_t;itol=itol,tol=tol,ρ=ρ)
                marker2Scan!(LODs,maridx,m,kmin,cross,est,Λg[:,i],Y1,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
                est0=[est0;est];
            end

        else #no LOCO
#            Xnul_t=Xnul*Tg';
           Xnul_t= BLAS.gemm('N','T',Xnul,Tg)
                if (cross!=1)
                   Y1,X=transForm(Tg,Y,X0,cross)
                   else
                   Y1,X=transForm(Tg,Y,XX.X,cross)
                 end


             est0=nulScan(init,kmin,Λg,Y1,Xnul_t;itol=itol,tol=tol,ρ=ρ)
        for i=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[i])
            marker2Scan!(LODs,maridx,m,kmin,cross,est0,Λg,Y1,Xnul_t,X;tol0=tol0,tol1=tol,ρ=ρ)
        end

        end #LOCO
    return LODs,est0
end



 