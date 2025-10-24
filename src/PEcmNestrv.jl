

############

### ECM algorithm : Estimate parameters by updating them using eStep and cmStep functions.
# Synopsis: B_cur,τ2_cur,Σ_cur,loglik0 = ecmLMM(Y,X,Z,B0,τ2_0,Σ,λg,λc) %'tol' can be chosen to set.
#Input: See eStep, cmStep descriptions
#Output:
#B_cur,τ2_cur,Σ_cur:  parameter estimation of B,τ2,Σ, respectively
#loglik0 : loglikelihood value by estimated parameters

function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);q=size(Z,2);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(q,p);Vg=zeros(n);Ve=zeros(m,m,n)
    #B0,τ2_0,Σ are arbitrary initial values
    B_cur=B0;τ2_cur=τ2_0;Σ_cur=Σ;loglik0=0.0;tol=tol
    crit=1.0;
    while (crit >=tol)
         B_new,τ2_new,Σ_new,loglik1=fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B_cur,τ2_cur,Σ_cur,λg,λc,m,n,ν₀,Ψ)
#          crit=norm(Σ_new-Σ_cur)+abs(τ2_new-τ2_cur)+norm(B_new-B_cur)
         crit=abs(loglik1-loglik0)
         B_cur=B_new;τ2_cur=τ2_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,τ2_cur,Σ_cur,loglik0

end

# update Σ with prior Ψ, ν₀(>m-1)
function updateΣ(Ve,Ψ::Array{Float64,2},ν₀,m,n)
    return (sum(Ve,dims=3)[:,:,1]+ Ψ)/(n+ν₀+m+1.0)
end
                 

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B_cur,τ2_cur::Float64,Σ_cur,λg,λc,m,n,ν₀,Ψ::Array{Float64,2})
             eStep!(Ghat,Θ,Y,X,Z,B_cur,τ2_cur,Σ_cur,λg,λc,m)
             cmStep!(B_new,dev,Vg,Ve,Y,X,Z,symXs,Ghat,Θ,Σ_cur,λg,λc,m)
             τ2_new = mean(Vg);Σ_new=updateΣ(Ve,Ψ,ν₀,m,n)
            loglik1=Loglik(dev,Σ_new,τ2_new,λg,λc,m);

    return B_new, τ2_new, Σ_new,loglik1
end

#Z=I
function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(m,p);Vg=zeros(n);Ve=zeros(m,m,n)
    #B0,τ2_0,Σ are arbitrary initial values
    B_cur=B0;τ2_cur=τ2_0;Σ_cur=Σ;loglik0=0.0;tol=tol
    crit=1.0;
    while (crit >=tol)
         B_new,τ2_new,Σ_new,loglik1=fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,τ2_cur,Σ_cur,λg,λc,m,n,ν₀,Ψ)
#          crit=norm(Σ_new-Σ_cur)+abs(τ2_new-τ2_cur)+norm(B_new-B_cur)
         crit=abs(loglik1-loglik0)
         B_cur=B_new;τ2_cur=τ2_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,τ2_cur,Σ_cur,loglik0

end

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,τ2_cur::Float64,Σ_cur,λg,λc,m,n,ν₀,Ψ::Array{Float64,2})
             eStep!(Ghat,Θ,Y,X,B_cur,τ2_cur,Σ_cur,λg,λc,m)
             cmStep!(B_new,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,λc,m)
             τ2_new = mean(Vg);Σ_new=updateΣ(Ve,Ψ,ν₀,m,n)
            loglik1=Loglik(dev,Σ_new,τ2_new,λg,λc,m);

    return B_new, τ2_new, Σ_new,loglik1
end







#whole ECM procedures grouped in one function: embedded in Nesterov's scheme
# fullECM : update all parameters

function fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,Z,symXs,b1,τ1::Array{Float64,1},Σ1,λg,λc,m,n,ν₀,Ψ::Array{Float64,2};numChr=0,nuMarker=0,niter=0)
             eStep!(Ghat,Θ,Y,X,Z,b1,τ1[1],Σ1,λg,λc,m)
             cmStep!(Bnew,dev,Vg,Ve,Y,X,Z,symXs,Ghat,Θ,Σ1,λg,λc,m)
             τ1 = mean(Vg);Σ1=updateΣ(Ve,Ψ,ν₀,m,n)
            loglik1=Loglik(dev,Σ1,τ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker,niter=niter);
            τ1=[τ1];
    return Bnew, τ1, Σ1,loglik1
end

#Z=I
function fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,symXs,b1,τ1::Array{Float64,1},Σ1,λg,λc,m,n,ν₀,Ψ::Array{Float64,2};numChr=0,nuMarker=0,niter=0)
             eStep!(Ghat,Θ,Y,X,b1,τ1[1],Σ1,λg,λc,m)
             cmStep!(Bnew,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,λc,m)
             τ1 = mean(Vg);Σ1=updateΣ(Ve,Ψ,ν₀,m,n)
            loglik1=Loglik(dev,Σ1,τ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker,niter=niter);
            τ1=[τ1];
    return Bnew, τ1, Σ1,loglik1
end



#full Nesterov's+debugging code included
function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};ρ=0.001,tol::Float64,numChr=0,nuMarker=0)

         symXs = fixX(X)
         m,n=size(Y);q=size(Z,2);p=size(X,1)
        #initialize parameters
        τ0=[τ2_0];τ00=copy(τ0);  τ1=copy(τ0);τ2=copy(τ0);
        b0=B0;b00=copy(b0); b2=copy(b0); b1=copy(b0);
        Σ0=Σ;Σ00=copy(Σ0);Σ2=copy(Σ0);Σ1=copy(Σ0);

         Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
         dev=zeros(m,n); Bnew=zeros(q,p);Vg=zeros(n);Ve=zeros(m,m,n)


        # l=1;
        crit=1.0; j=1;loglik0=0.0;
        #    println("Chr $(numChr) and marker $(nuMarker).")
         itrnum=1
        while (crit >=tol)

            b1, τ1, Σ1, loglik1 = fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,Z,symXs,b1,τ1,Σ1,λg,λc,m,n,ν₀,Ψ;numChr=numChr,nuMarker=nuMarker,niter=itrnum)
            #some tweak for τ2
                 τ1[1] = max(τ1[1],ρ)
            #Speed restarting Nesterov's Scheme
               updatNestrvAG!(j,b0,b1,b2,τ0,τ1,τ2,Σ0,Σ1,Σ2)

            if (norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(τ0-τ00)+norm(b0-b00)) & (j>=kmin)
                j=1
                itrnum+=1
            else
                j=j+1
                itrnum+=1
            end

#             crit=norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)
            crit=abs(loglik1-loglik0)

            b00=b0;b0=b1;b1=b2; τ00=τ0;τ0=τ1;τ1=τ2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;

        end

        #  println("Chr $(numChr) and marker $(nuMarker) with total $(itrnum) to be done.")
         
        return Approx(b1,τ1[1],Σ1,loglik0)

end


#Z=I
function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};ρ=0.001,tol::Float64,numChr=0,nuMarker=0)

         symXs = fixX(X)
         m,n=size(Y);p=size(X,1)
        #initialize parameters
        τ0=[τ2_0];τ00=copy(τ0);  τ1=copy(τ0);τ2=copy(τ0);
        b0=B0;b00=copy(b0); b2=copy(b0); b1=copy(b0);
        Σ0=Σ;Σ00=copy(Σ0);Σ2=copy(Σ0);Σ1=copy(Σ0);

         Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
         dev=zeros(m,n); Bnew=zeros(m,p);Vg=zeros(n);Ve=zeros(m,m,n)


        # l=1;
        crit=1.0; j=1;loglik0=0.0;
        #    println("Chr $(numChr) and marker $(nuMarker).")
         itrnum=1
        while (crit >=tol)

            b1, τ1, Σ1, loglik1 = fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,symXs,b1,τ1,Σ1,λg,λc,m,n,ν₀,Ψ;numChr=numChr,nuMarker=nuMarker,niter=itrnum)
            #some tweak for τ2
                 τ1[1] = max(τ1[1],ρ)
            #Speed restarting Nesterov's Scheme
               updatNestrvAG!(j,b0,b1,b2,τ0,τ1,τ2,Σ0,Σ1,Σ2)

            if (norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(τ0-τ00)+norm(b0-b00)) & (j>=kmin)
                j=1
                itrnum+=1
            else
                j=j+1
                itrnum+=1
            end

#             crit=norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)
            crit=abs(loglik1-loglik0)

            b00=b0;b0=b1;b1=b2; τ00=τ0;τ0=τ1;τ1=τ2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;

        end
       
        return Approx(b1,τ1[1],Σ1,loglik0)

end



### ECM embeded in Nesterov's accelerated gradient method with speed restarting scheme
#Synopsis : result=ecmNestrvAG(kmin,Y,X,Y,Z,B0,τ2_0,Σ,λg,λc) %'tol' can be chosen to set.
#
# Input:
# kmin : an integer. A minimum iteration (default=1) to restart the algorithm. Depending on the data, this value can be freely
#       set to prevent any loglikelihood value from yielding a domain error until reaching kmin.
# other inputs : see eStep, cmStep, or K2eig
# Output: result: a struct of Approx. Approx.B, Approx.τ2, Approx.Σ   : parameter estimation of B,τ2,Σ, respectively
# Approx.loglik : loglikelihood value by estimated parameters


function ecmNestrvAG(lod0::Float64,kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
         τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64,ρ=0.001,numChr=0,nuMarker=0)

              if (lod0>0.0)
                result = NestrvAG(kmin,Y,X,Z,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol,numChr=numChr,nuMarker=nuMarker)
                else #keep running ecmLMM
                B0,τ2_0,Σ,loglik=ecmLMM(Y,X,Z,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;tol=tol)
                 result = NestrvAG(kmin,Y,X,Z,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol,numChr=numChr,nuMarker=nuMarker)
              end
                return result
 end

#Z=I
function ecmNestrvAG(lod0::Float64,kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
         τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64,ρ=0.001,numChr=0,nuMarker=0)

              if (lod0>0.0)
                result = NestrvAG(kmin,Y,X,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol,numChr=numChr,nuMarker=nuMarker)
               else #keep running ecmLMM
                 B0,τ2_0,Σ,loglik = ecmLMM(Y,X,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;tol=tol)
                 result = NestrvAG(kmin,Y,X,B0,τ2_0,Σ,λg,λc,ν₀,Ψ;ρ=ρ,tol=tol,numChr=numChr,nuMarker=nuMarker)
              end
              return result
    
 end

#MVLMM
#fullECM : update all parameters
#Z=I
function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B,Vc::Array{Float64,2},Σ,λg,m,n,ν₀,Ψ::Array{Float64,2})

          eStep!(Ghat,Θ,Y,X,B,Vc,Σ,λg,m)
          cmStep!(B_new,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,m)
          Vc_new=mean(Vg,dims=3)[:,:,1]; Σ_new=updateΣ(Ve,Ψ,ν₀,m,n)
          loglik1=Loglik(dev,Σ_new,Vc_new,λg)

    return B_new,Vc_new,Σ_new,loglik1

end

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B::Array{Float64,2},Vc::Array{Float64,2},Σ,λg,m,n,ν₀,Ψ::Array{Float64})

          eStep!(Ghat,Θ,Y,X,Z,B,Vc,Σ,λg,m)
          cmStep!(B_new,dev,Vg,Ve,Y,X,Z,symXs,Ghat,Θ,Σ,λg,m)
          Vc_new=mean(Vg,dims=3)[:,:,1]; Σ_new=updateΣ(Ve,Ψ,ν₀,m,n)
          loglik1=Loglik(dev,Σ_new,Vc_new,λg)

    return B_new,Vc_new,Σ_new,loglik1

end

#Z=I
function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(m,p);Vg=zeros(m,m,n);Ve=zeros(m,m,n)
    #B0,Vc,Σ are arbitrary initial values
    B_cur=B0;Vc_cur=Vc;Σ_cur=Σ;loglik0=0.0;tol=tol

    # i=1;
    crit=1.0;
    while (crit >=tol)
         B_new, Vc_new, Σ_new, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,Vc_cur,Σ_cur,λg,m,n,ν₀,Ψ)
#          crit=norm(Σ_new-Σ_cur)+norm(Vc_new-Vc_cur)+norm(B_new-B_cur)
         crit=abs(loglik1-loglik0)
         B_cur=B_new;Vc_cur=Vc_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,Vc_cur,Σ_cur,loglik0

end

function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);p=size(X,1);q=size(Z,2);
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(q,p);Vg=zeros(m,m,n);Ve=zeros(m,m,n)
    #B0,Vc,Σ are arbitrary initial values
    B_cur=B0;Vc_cur=Vc;Σ_cur=Σ;loglik0=0.0;tol=tol

    # i=1;
    crit=1.0;
    while (crit >=tol)
         B_new, Vc_new, Σ_new, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B_cur,Vc_cur,Σ_cur,λg,m,n,ν₀,Ψ)
#          crit=norm(Σ_new-Σ_cur)+norm(Vc_new-Vc_cur)+norm(B_new-B_cur)
         crit=abs(loglik1-loglik0)
         B_cur=B_new;Vc_cur=Vc_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,Vc_cur,Σ_cur,loglik0

end

#Z=I
function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};ρ=0.001,tol::Float64)

      m,n =size(Y);p=size(X,1);
      symXs = fixX(X)
      Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
      dev=zeros(m,n);B_new=zeros(m,p);Vg=zeros(m,m,n);Ve=zeros(m,m,n)

        #initialize parameters
            V0=Vc;V00=copy(V0);  V1=copy(V0);V2=copy(V0);
            b0=B0;b00=copy(b0); b2=copy(b0); b1=copy(b0);
            Σ0=Σ;Σ00=copy(Σ0);Σ2=copy(Σ0);Σ1=copy(Σ0);

            # l=1;
             crit=1.0; j=1;loglik0=0.0;  #i=1;
           while (crit >=tol)
             b1, V1, Σ1, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,b1,V1,Σ1,λg,m,n,ν₀,Ψ)
            #Speed restarting Nesterov's Scheme
            updatNestrvAG!(j,b0,b1,b2,V0,V1,V2,Σ0,Σ1,Σ2)

               if (!isposdef(V2))
                     V2 = V2+(abs(eigmin(V2))+ρ)*I
               end

               if (!isposdef(Σ2))
                     Σ2 = Σ2 +(abs(eigmin(Σ2))+ρ)*I
               end

            if (norm(Σ1-Σ0)+norm(V1-V0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(V0-V00)+norm(b0-b00)) & (j>=kmin)
                j=1
            else
                j=j+1
            end

#             crit=norm(Σ1-Σ0)+norm(V1-V0)+norm(b1-b0)
              crit=abs(loglik1-loglik0)
            b00=b0;b0=b1;b1=b2; V00=V0;V0=V1;V1=V2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;
#             i+=1
           end
           return Result(b1,V1,Σ1,loglik0)

end

function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};ρ=0.001,tol::Float64)

      m,n =size(Y);p=size(X,1);q=size(Z,2)
      symXs = fixX(X)
      Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
      dev=zeros(m,n);B_new=zeros(q,p);Vg=zeros(m,m,n);Ve=zeros(m,m,n)

        #initialize parameters
            V0=Vc;V00=copy(V0);  V1=copy(V0);V2=copy(V0);
            b0=B0;b00=copy(b0); b2=copy(b0); b1=copy(b0);
            Σ0=Σ;Σ00=copy(Σ0);Σ2=copy(Σ0);Σ1=copy(Σ0);

            # l=1;
             crit=1.0; j=1;loglik0=0.0;  #i=1;
           while (crit >=tol)
             b1, V1, Σ1, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,b1,V1,Σ1,λg,m,n,ν₀,Ψ)
            #Speed restarting Nesterov's Scheme
            updatNestrvAG!(j,b0,b1,b2,V0,V1,V2,Σ0,Σ1,Σ2)

               if (!isposdef(V2))
                     V2 = V2+(abs(eigmin(V2))+ρ)*I
               end

               if (!isposdef(Σ2))
                     Σ2 = Σ2 +(abs(eigmin(Σ2))+ρ)*I
               end

            if (norm(Σ1-Σ0)+norm(V1-V0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(V0-V00)+norm(b0-b00)) & (j>=kmin)
                j=1
            else
                j=j+1
            end

#             crit=norm(Σ1-Σ0)+norm(V1-V0)+norm(b1-b0)
            crit=abs(loglik1-loglik0)
            b00=b0;b0=b1;b1=b2; V00=V0;V0=V1;V1=V2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;
#             i+=1
           end
           return Result(b1,V1,Σ1,loglik0)

end


#Z=I
function ecmNestrvAG(lod0::Float64,kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64,ρ=1e-4)

             if(lod0>0.0)
             result=  NestrvAG(kmin,Y,X,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol,ρ=ρ)
                else #keep running ecmLMM
              B0,Vc,Σ,loglik0 = ecmLMM(Y,X,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol)
              result=  NestrvAG(kmin,Y,X,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol,ρ=ρ)
             end
             
      return result

end





function ecmNestrvAG(lod0::Float64,kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},ν₀,Ψ::Array{Float64,2};tol::Float64,ρ=1e-4)

             if(lod0>0.0)
             result=  NestrvAG(kmin,Y,X,Z,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol,ρ=ρ)
                else #keep running ecmLMM
              B0,Vc,Σ,loglik0 = ecmLMM(Y,X,Z,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol)
              result=  NestrvAG(kmin,Y,X,Z,B0,Vc,Σ,λg,ν₀,Ψ;tol=tol,ρ=ρ)
             end
             
      return result

end









