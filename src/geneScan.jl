
function marker1Scan(nmar,cross,nulpar::Estimat,Y,Xnul,X,Z,reml)
    
    if(cross!=1)
        lod= @distributed (vcat) for j=1:nmar
            est1 = mGLM(Y,hcat(Xnul,X[:,2:end,j]),Z,reml)
            [(est1.loglik-nulpar.loglik)/log(10) est1]
        end

     else #cross =1
        lod = @distributed (vcat) for j=1:nmar
        est1 = mGLM(Y,hcat(Xnul,X[:,j]),Z,reml)
        [(est1.loglik-nulpar.loglik)/log(10) est1]
         end

    end
     return lod[:,1], lod[:,2]
end

#Z=I
function marker1Scan(nmar,cross,nulpar::Estimat,Y,Xnul,X,reml)
    
    if(cross!=1)
        lod= @distributed (vcat) for j=1:nmar
            est1 = mGLM(Y,hcat(Xnul,X[:,2:end,j]),reml)
            [(est1.loglik-nulpar.loglik)/log(10) est1]
        end

     else #cross =1
        lod = @distributed (vcat) for j=1:nmar
        est1 = mGLM(Y,hcat(Xnul,X[:,j]),reml)
        [(est1.loglik-nulpar.loglik)/log(10) est1]
         end

    end
     return lod[:,1], lod[:,2]
end


function marker2Scan!(LODs,mindex::Array{Int64,1},cross,nulpar::Estimat,Y,Xnul,X,Z,reml)
      M=length(mindex)
    if(cross!=1)
        for j=1:M-1
            lod=@distributed (vcat) for l=j+1:M
            est1=mGLM(Y,hcat(Xnul,X[:,2:end,j],X[:,2:end,l]),Z,reml)
            (est1.loglk-nulpar.loglik)/log(10)
                  end
            LODs[mindex[j+1:end],mindex[j]].=lod
        end
    else #cross=1
        for j=1:M-1
            lod=@distributed (vcat) for l=j+1:M
            est1=mGLM(Y,hcat(Xnul,X[:,[j,l]]),Z,reml)
            (est1.loglk-nulpar.loglik)/log(10) 
               end
            LODs[mindex[j+1:end],mindex[j]].=lod
        end
    end

end

#Z=I
function marker2Scan!(LODs,mindex::Array{Int64,1},cross,nulpar::Estimat,Y,Xnul,X,reml)
    M=length(mindex)
  if(cross!=1)
      for j=1:M-1
          lod=@distributed (vcat) for l=j+1:M
          est1=mGLM(Y,hcat(Xnul,X[:,2:end,j],X[:,2:end,l]),reml)
          (est1.loglk-nulpar.loglik)/log(10) 
           end
          LODs[mindex[j+1:end],mindex[j]].=lod
      end
  else #cross=1
      for j=1:M-1
          lod=@distributed (vcat) for l=j+1:M
          est1=mGLM(Y,hcat(Xnul,X[:,[j,l]]),reml)
          (est1.loglk-nulpar.loglik)/log(10) 
            end
          LODs[mindex[j+1:end],mindex[j]].=lod
      end
    end

end

function mat2Array(cross::Int64,p,X)
   #size(X)=(n,p)
 
   X0=zeros(n,cross,p)
   @inbounds @views for j = 1:p
    X0[:,:,j] = X[:,cross*j-(cross-1):cross*j]
   end

   return X0

end

function getB(H1par,p0::Int64,q,p,cross)
  #p0= size(Xnul,2)
    if (cross!=1)
       B=zeros(cross-1+p0,q,p)
    else
       B=zeros(cross+p0,q,p)
    end

      @inbounds @views for j=1:length(H1par)
           B[:,:,j]=H1par[j].B
       end
   return B
end

##########

function mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;LogP::Bool=false,
    Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); q=size(Z,1)
    LODs = zeros(p); H1par=[]
    

     #nul scan
     est0=mGLM(Y,Xnul,Z,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
        LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,X,Z,reml)
    else
        LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,XX.X,Z,reml)     
    end
     
    B= getB(H1par,size(Xnul,2),q,p,cross)
    

    if(LogP)
        df = prod(size(B[:,:,1]))-prod(size(est0.B))
        logP=lod2logP(LODs,df)
        return logP, B, est0
    else #print lods

        return LODs,B, est0
    end
       
end

#Z=I
function mlm1Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;LogP::Bool=false,
    Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), 
    p=Int(size(XX.X,2)/cross); m=size(Y,2)
    LODs = zeros(p); H1par=[]
    

     #nul scan
     est0=mGLM(Y,Xnul,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
        LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,X,reml)
    else
        LODs[:],H1par = marker1Scan(p,cross,est0,Y,Xnul,XX.X,reml)     
    end
     
    B= getB(H1par,size(Xnul,2),m,p,cross)
    

    if(LogP)
        df = prod(size(B[:,:,1]))-prod(size(est0.B))
        logP=lod2logP(LODs,df)
        return logP, B, est0
    else #print lods

        return LODs,B, est0
    end
       
end

#######


function mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,Z::Matrix{Float64},reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))

    # size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); q=size(Z,1)
    LODs=zeros(p,p);  Chr=unique(XX.chr); nChr=length(Chr);
 
    #nul scan
    est0=mGLM(Y,Xnul,Z,reml)
    if(cross!=1)
        X = mat2Array(cross,p,XX.X)
        for j=eachindex(Chr)
            maridx=findall(XX.chr.==Chr[j])
            marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,X,Z,reml)
        end
       
     else #cross=1
        for j= eachindex(Chr)
            maridx=findall(XX.chr.==Chr[j])
         marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,XX.X,Z,reml) 
        end
    end
     
    
        return LODs, est0
    
    
end

#Z=I
function mlm2Scan(cross::Int64,Y::Matrix{Float64},XX::Markers,reml::Bool=false;Xnul::Matrix{Float64}=ones(size(Y,1),1))

# size(X)=(n,p), size(Z)=(q,m)
    p=Int(size(XX.X,2)/cross); 
    LODs=zeros(p,p);  Chr=unique(XX.chr);

#nul scan
    est0=mGLM(Y,Xnul,reml)
  if(cross!=1)
    X = mat2Array(cross,p,XX.X)
    for j=eachindex(Chr)
        maridx=findall(XX.chr.==Chr[j])
        marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,X,reml)
    end
   
   else #cross=1
    for j= eachindex(Chr)
        maridx=findall(XX.chr.==Chr[j])
     marker2Scan!(LODs,maridx,cross,est0,Y,Xnul,XX.X,reml) 
    end
  end
 
  return LODs, est0

end

#########

function mlmTperm()


end