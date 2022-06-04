


# estimate Kc and update initial values of parameters for nulscan
function updateKc(m::Int64,initial::Union{Init0,Init1},Tg::Array{Float64,2},λg::Array{Float64,1},
        Y0::Array{Float64,2},Z0::Array{Float64,2},
        Xnul::Array{Float64,2};itol=1e-2,tol=1e-3,ρ::Float64)
        
         Y1,Xnul_t = transForm(Tg,Y0,Xnul,1) #null model transformation
      
     if (typeof(initial)==Init0)
          est0=nulScan1(initial,1,λg,Y1,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
          Tc, λc = K2eig(est0.Vc)
        # trait-wise transformation
          Z1,Σ1 =  transForm(Tc,Z0,est0.Σ,true)
          Y1 = transForm(Tc,Y1,est0.Σ)
        # update intial values
            Vc=copy(est0.Vc)
           lmul!(sqrt(1/m),Vc)
           τ2 =mean(Diagonal(Vc))
           
      return λc, Y1, Xnul_t,Z1, Init1(est0.B,τ2,Σ1,est0.Vc)
        
    else #typeof(initial==Init1)
          est0=nulScan1(initial,1,λg,Y1,Xnul_t,Z0;itol=itol,tol=tol,ρ=ρ)
        
          Tc, λc = K2eig(est0.Vc)
     # trait-wise transformation
          Z1,Σ1 =  transForm(Tc,Z0,est0.Σ,true)
          Y1 = transForm(Tc,Y1,est0.Σ)
    # update intial values
            Vc=copy(est0.Vc)
           lmul!(sqrt(1/m),Vc)
           τ2 =mean(Diagonal(Vc))
           
      return λc, Y1, Xnul_t,Z1, Init1(est0.B,τ2,Σ1,est0.Vc)
    
    end
        
        
end


function gene1Scan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
        tdata::Bool=false,LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y0,2)),
        itol=1e-3,tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        m=size(Y0,1);
        q=size(Z0,2);  p=Int(size(XX.X,1)/cross);

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0,Z0,false) #type: Init0

         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]

           for i=1:nChr
            #compute Kc & update intital values
       λc, Y1, Xnul_t,Z1,init1=updateKc(m,init,Tg[:,:,i],Λg[:,i],Y0,Z0,Xnul;itol=itol,tol=tol,ρ=ρ)
                maridx=findall(XX.chr.==Chr[i])
           
                 if (cross!=1) #individual-wise tranformation 
                   X1=transForm(Tg[:,:,i],X0[maridx,:,:],cross)
                   else
                   X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                est00=nulScan(init1,1,Λg[:,i],λc,Y1,Xnul_t,Z1;ρ=ρ,itol=itol,tol=tol); 
                lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y1,Xnul_t,X1,Z1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
                LODs[maridx]=lods;
                H1par=[H1par;H1par1]
                est0=[est0;est00];init=init1
            end
           # rearrange B into 3-d array
           B = arrngB(H1par,size(Xnul,1),q,p,cross)

        else #no LOCO

            λc, Y1, Xnul_t,Z1,init1=updateKc(m,init,Tg,Λg,Y0,Z0,Xnul;itol=itol,tol=tol,ρ=ρ)
                 if (cross!=1) #individual-wise tranformation 
                   X1=transForm(Tg,X0,cross)
                   else
                   X1=transForm(Tg,XX.X,cross)
                 end

                  est0=nulScan(init1,1,Λg,λc,Y1,Xnul_t,Z1;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)
    end

    # Output choice
    if (tdata) # should use with no LOCO to do permutation
        return LODs,B,est0,Y1,X1,Z1
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


function geneScan(cross::Int64,Tg,Λg,Y0::Array{Float64,2},XX::Markers,Z0::Array{Float64,2},LOCO::Bool=false;
        tdata::Bool=false,LogP::Bool=false,Xnul::Array{Float64,2}=ones(1,size(Y0,2)),itol=1e-3,
        tol0=1e-3,tol::Float64=1e-4,ρ=0.001)

        m=size(Y0,1);
        q=size(Z0,2);  p=Int(size(XX.X,1)/cross);

        ## picking up initial values for parameter estimation under the null hypothesis
            init=initial(Xnul,Y0,Z0)
             
         if (cross!=1)
            X0=mat2array(cross,XX.X)
         end
    if (LOCO)
        LODs=zeros(p);
        Chr=unique(XX.chr);nChr=length(Chr);est0=[];H1par=[]
           
           for i=1:nChr
                maridx=findall(XX.chr.==Chr[i])
             Kc, Y1, Xnul_t= obtainKc(Tg[:,:,i],Λg[:,i],Y0,Z0;Xnul=Xnul,itol=itol,tol=tol,ρ=ρ)
             Tc, λc = K2eig(Kc)
              
              if(λc!= ones(m))
                Z1,Σ1 =  transForm(Tc,Z0,init.Σ,true)
                Y1= transForm(Tc,Y1,init.Σ) # transform Y only by row (Tc)
               else
               Z1=Z0; Σ1 = init.Σ
#                Y1=Y0
              end
#                 Xnul_t=Xnul*Tg[:,:,i]';
#    @fastmath @inbounds Xnul_t=BLAS.gemm('N','T',Xnul,@view Tg[:,:,i])
                 if (cross!=1)
                   X1=transForm(Tg[:,:,i],X0[maridx,:,:],cross)
                   else
                   X1=transForm(Tg[:,:,i],XX.X[maridx,:],cross)
                 end
                #parameter estimation under the null
                  est00=nulScan(init,1,Λg[:,i],λc,Y1,Xnul_t,Z1,Σ1;ρ=ρ,itol=itol,tol=tol)
                lods,H1par1=marker1Scan(q,1,cross,est00,Λg[:,i],λc,Y1,Xnul_t,X1,Z1;ρ=ρ,tol0=tol0,tol1=tol,nchr=i)
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

                  est0=nulScan(init,1,Λg,λc,Y1,Xnul_t,Z1,Σ1;itol=itol,tol=tol,ρ=ρ)
                LODs,H1par=marker1Scan(q,1,cross,est0,Λg,λc,Y1,Xnul_t,X1,Z1;tol0=tol0,tol1=tol,ρ=ρ)
             # rearrange B into 3-d array
          B = arrngB(H1par,size(Xnul,1),q,p,cross)
    end

    # Output choice
    if (tdata) # should use with no LOCO to do permutation
        return LODs,B,est0,Y1,X1,Z1
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
