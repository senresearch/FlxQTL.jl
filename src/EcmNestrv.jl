"""

    EcmNestrv

A module for baseline algorithm using ECM (Expectation-Conditional Maxization) with Speed restarting Nesterov's accelerated gradient method 
to fit a flexible multivariate linear mixed model (flxMLMM).

"""
module EcmNestrv

using LinearAlgebra
import Statistics: mean
import Distributions: MvNormal, loglikelihood

export Approx, Result

#export functions
# export fixX,fixZ,fixVar,fixVar!,eStep!,cmStep,cmStep!,ecmLMM,Loglik

# export ecmNestrvAG, updatNestrvAG!, fullECM, NestrvAG



# symSq(X::Array{Float64,2},trans::Char)

# Effectively computes symmetric X'X or XX' using a upper trangular part of an input matrix

# Input:
# X : a matrix
# trans: a character determining 'T'(transpose),'C'(conjugated and transposed),'N' (neither of them)

# Output : either X'X or XX' or its conjugate symetric matrix

# ex:
# for X'X :: symSq(X,'T')
# for XX' :: symSq(X,'N')

function symSq(X::Array{Float64,2},trans::Char)
    XtX=Symmetric(BLAS.syrk('U',trans,1.0,X))
    return XtX
end

export  symSq, fixX, fixZ



# fixZ(Z::Array{Float64,2},Σ::Array{Float64,2})
# Precomputes (Z'*(Σ\Z))\Z'

function fixZ(Z::Array{Float64,2},Σ::Array{Float64,2})

                return Symmetric(BLAS.gemm('T','N',Z,Σ\Z))\Z'
end


# fixX(X::Array{Float64,2})
# Precomputes X'(XX')^{-1}

function fixX(X::Array{Float64,2})

       return convert(Array{Float64,2},(symSq(X,'N')\X)')
end




# fixVar!(Vc::Array{Float64,2},V::Array{Float64,2},Σ::Array{Float64,2},
#         λg::Float64,Λc::Union{Array{Float64,2},Diagonal{Float64,Array{Float64,1}}},dev0::Array{Float64,1})

# Precomputes variance component parts and returns Vc= λg*Diagonal(τ2_0*λc), V[:,1]=inv(Vc+Σ)*dev0, V[:,2:end]=inv(Vc+Σ)*Vc

function fixVar!(Vc::Array{Float64,2},V::Array{Float64,2},Σ::Array{Float64,2},
        λg::Float64,Λc::Union{Array{Float64,2},Diagonal{Float64,Array{Float64,1}}},dev0::Array{Float64,1})

#        Λc=Diagonal(τ2_0*λc)
         Vc[:,:] = λg*Λc
         V[:,:] = (Vc+Σ)\[dev0 Vc]
   
end



##MVLMM
function fixVar!(V::Array{Float64,2},Vinv::Array{Float64,2},Vc::Array{Float64,2},
        Σ::Array{Float64,2},λg::Float64,dev0::Array{Float64,1})

                V[:,:]=λg*Vc
                Vinv[:,:]=(V+Σ)\[dev0 V]

end





##eStep : conditional expectation step in ECM algorithm.

#Output:
# Ghat : a matrix of conditional expected value of a random effect G
# Θ : an array of variance-covariance matrices of G


#estep: for full updates and caching Variances for cmStep

function eStep!(Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},
        B0::Array{Float64,2},τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},m::Int64)

    #finding G_i|y_i,B(t),Vg(t)(=τ2(t)*Λc),Ve(t)(=Σ(t))~ MVN(Ghat_i(t),Θ(t)), where V(t)=λg[i]*Vg(t)+Ve(t),
    #at the current iteration t
    Vc=zeros(m,m);
    V=zeros(m,m+1); dev0=similar(Y);

          # compute deviation
#           dev0=Y-(Z*B0)*X
           mul!(dev0,(Z*B0),X)
           axpby!(1.0,Y,-1.0,dev0)

           if(λc!=ones(m))
               Λc=Diagonal(τ2_0*λc)
             else
               Λc=τ2_0*Matrix(1.0I,m,m)
           end

  @fastmath @inbounds  for i= eachindex(λg)
                fixVar!(Vc,V,Σ,λg[i],Λc,dev0[:,i])
                 @views Ghat[:,i] = BLAS.symv('U',Vc,V[:,1])
                 @views Θ[i] = Symmetric(Vc-BLAS.symm('L','U',Vc,V[:,2:end]))
                       end


end

## Z=I
function eStep!(Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},Y::Array{Float64,2},X::Array{Float64,2},
        B0::Array{Float64,2},τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},m::Int64)

    #finding G_i|y_i,B(t),Vg(t)(=τ2(t)*Λc),Ve(t)(=Σ(t))~ MVN(Ghat_i(t),Θ(t)), where V(t)=λg[i]*Vg(t)+Ve(t),
    #at the current iteration t
    Vc=zeros(m,m);
    V=zeros(m,m+1); dev0=similar(Y);

          # compute deviation
#           dev0=Y-(Z*B0)*X
           mul!(dev0,B0,X)
           axpby!(1.0,Y,-1.0,dev0)
       if(λc!=ones(m))
           Λc=Diagonal(τ2_0*λc)
         else
           Λc=τ2_0*Matrix(1.0I,m,m)
       end

  @fastmath @inbounds for i= eachindex(λg)         
                       fixVar!(Vc,V,Σ,λg[i],Λc,dev0[:,i])
                 @views Ghat[:,i] = BLAS.symv('U',Vc,V[:,1])
                 @views Θ[i] = Symmetric(Vc-BLAS.symm('L','U',Vc,V[:,2:end]))
                       end


end


##MVLMM
function eStep!(Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},Y::Array{Float64,2},X::Array{Float64,2},
        B0::Array{Float64,2},Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1},m::Int64)

    V=zeros(m,m);
    Vinv=zeros(m,m+1); dev0=similar(Y);
    #finding G_i|y_i,B(t),Vg(t),Ve(t)(=Σ(t))~ MVN(Ghat_i(t),Θ(t)), where V(t)=λg[i]*Vg(t)+Ve(t),
    #at the current iteration t

    #      dev0=Y-(Z*B0)*X
           mul!(dev0,B0,X)
           axpby!(1.0,Y,-1.0,dev0)

    @fastmath @inbounds for i in eachindex(λg)
               fixVar!(V,Vinv,Vc,Σ,λg[i],dev0[:,i])
           @views Ghat[:,i]=BLAS.symv('U',V,Vinv[:,1])
           @views Θ[i]=Symmetric(V-BLAS.symm('L','U',V,Vinv[:,2:end]))
             end

end


## cmStep! : conditional maximization step in ECM. Update parameters(B,τ2,Σ) using return values from eStep
#
#Output:
# Bnew,Vg, Ve : updated B,τ2,Σ, respectively.
# dev : a deviation matrix (dev= Y-Z*B1*X)

function cmStep!(Bnew::Array{Float64,2},dev::Array{Float64,2},Vg::Array{Float64,1},Ve::Array{Float64,3},Y::Array{Float64,2},
        X::Array{Float64,2},Z::Array{Float64,2},symXs::Array{Float64,2},Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},
        Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1},m::Int64)


    # update B, Vg(=τ2*Λc), Ve(=Σ)
      Bnew[:,:],dev[:,:] =cmStep(Y,X,Z,symXs,Ghat,Σ)
      Ehat=dev-Ghat
    if(λc!= ones(m))
           Λc= Diagonal(1.0./λc)
         @fastmath @inbounds @views for i in eachindex(λg)
           Ve[:,:,i]=Θ[i]
           Vg[i]=tr(LinearAlgebra.lmul!(Λc,Float64.(Symmetric(BLAS.syr!('U',1.0,Ghat[:,i],Θ[i])))))/(λg[i]*m)
           Ve[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ehat[:,i],Ve[:,:,i]))
                  end
       else
         @fastmath @inbounds @views for i in eachindex(λg)
         Ve[:,:,i]=Θ[i]
         Vg[i]=tr(Symmetric(BLAS.syr!('U',1.0,Ghat[:,i],Θ[i])))/(λg[i]*m)
         Ve[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ehat[:,i],Ve[:,:,i]))
                 end
    end

end

#Z=I,
function cmStep!(Bnew::Array{Float64,2},dev::Array{Float64,2},Vg::Array{Float64,1},Ve::Array{Float64,3},
        Y::Array{Float64,2},X::Array{Float64,2},symXs::Array{Float64,2},Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},
        λg::Array{Float64,1},λc::Array{Float64,1},m::Int64)

    # update B, Vg(=τ2*Λc), Ve(=Σ)
       Bnew[:,:],dev[:,:] =cmStep(Y,X,symXs,Ghat)
      Ehat=dev-Ghat

    if(λc!= ones(m))
         Λc= Diagonal(1.0./λc)
         @fastmath @inbounds @views for i in eachindex(λg)
               Ve[:,:,i]=Θ[i]
               Vg[i]=tr(LinearAlgebra.lmul!(Λc,Float64.(Symmetric(BLAS.syr!('U',1.0,Ghat[:,i],Θ[i])))))/(λg[i]*m)
               Ve[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ehat[:,i],Ve[:,:,i]))
                        end
         else
           @fastmath @inbounds @views for i in eachindex(λg)
              Ve[:,:,i]=Θ[i]
              Vg[i]=tr(Symmetric(BLAS.syr!('U',1.0,Ghat[:,i],Θ[i])))/(λg[i]*m)
              Ve[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ehat[:,i],Ve[:,:,i]))
                      end
      end

end



#compute B, dev
function cmStep(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},symXs::Array{Float64,2},
         Ghat::Array{Float64,2},Σ::Array{Float64,2})

#          B1= (fixZ(Z,Σ)*(Σ\(Y-Ghat)))*symXs
         B1 = BLAS.gemm('N','N',fixZ(Z,Σ),(Σ\(Y-Ghat))*symXs)
        dev=Y- BLAS.gemm('N','N',(Z*B1),X)

     return B1, dev
end

#Z=I
function cmStep(Y::Array{Float64,2},X::Array{Float64,2},symXs::Array{Float64,2},Ghat::Array{Float64,2})

         B1 = BLAS.gemm('N','N',(Y-Ghat),symXs)
#          B1=(Y-Ghat)*symXs
         #dev=Y-(Z*B1)*X
        dev=Y- BLAS.gemm('N','N',B1,X)

     return B1, dev
end


#MVLMM
function cmStep!(Bnew::Array{Float64,2},dev::Array{Float64,2},Vg::Array{Float64,3},Ve::Array{Float64,3},Y::Array{Float64,2},
    X::Array{Float64,2},symXs::Array{Float64,2},Ghat::Array{Float64,2},Θ::Array{Array{Float64,2},1},
        λg::Array{Float64,1},m::Int64)

    # update B, Vg(=τ2*Λc), Ve(=Σ)
#     B1= fixZ(Z,Σ)*(Σ\(Y-Ghat))*symXs
#     dev=Y-(Z*B1)*X
     Bnew[:,:],dev[:,:] =cmStep(Y,X,symXs,Ghat)
     Ehat=dev-Ghat

     @fastmath @inbounds @views for i in eachindex(λg)
        Ve[:,:,i]=Θ[i]
        Vg[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ghat[:,i],Θ[i]))/λg[i]
        Ve[:,:,i]=Symmetric(BLAS.syr!('U',1.0,Ehat[:,i],Ve[:,:,i]))

    end


end





# Loglik: evaluating a loglikelihood
# Synopsis:  loglik=Loglik(dev,Σ,τ2,λg,λc,m)
# Input: See the cmStep description.

function Loglik(dev::Array{Float64,2},Σ::Array{Float64,2},τ2::Float64,λg::Array{Float64,1},λc::Array{Float64,1},m::Int64
        ;numChr=0,nuMarker=0,niter=0)

          loglik=zero(eltype(dev))
          if(λc!=ones(m))
              Λc= τ2*Diagonal(1.0./λc)
           else
              Λc=τ2*Matrix(1.0I,m,m)
         end

   @fastmath @inbounds @views  for j in eachindex(λg)
#          try
        V=MvNormal(λg[j]*Λc+Σ)
        loglik += loglikelihood(V, dev[:,j])
#         catch
#              println("Chr $(numChr) and marker $(nuMarker), while $(niter).")
#              error("logdet error")
#              interrupt()
#          end
       end

    return loglik

end



##MVLMM
function Loglik(dev::Array{Float64,2},Σ::Array{Float64,2},Vc::Array{Float64,2},λg::Array{Float64,1},m::Int64)

    loglik=zero(eltype(dev))


     @fastmath @inbounds @views for j in eachindex(λg)
           V=MvNormal(λg[j]*Vc+Σ)
           loglik += loglikelihood(V,dev[:,j])
    end


    return loglik

end



### ECM algorithm : Estimate parameters by updating them using eStep and cmStep functions.
# Synopsis: B_cur,τ2_cur,Σ_cur,loglik0 = ecmLMM(Y,X,Z,B0,τ2_0,Σ,λg,λc) %'tol' can be chosen to set.
#Input: See eStep, cmStep descriptions
#Output:
#B_cur,τ2_cur,Σ_cur:  parameter estimation of B,τ2,Σ, respectively
#loglik0 : loglikelihood value by estimated parameters

function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);q=size(Z,2);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(q,p);Vg=zeros(n);Ve=zeros(m,m,n)
    #B0,τ2_0,Σ are arbitrary initial values
    B_cur=B0;τ2_cur=τ2_0;Σ_cur=Σ;loglik0=0.0;tol=tol
    crit=1.0;
    while (crit >=tol)
         B_new,τ2_new,Σ_new,loglik1=fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B_cur,τ2_cur,Σ_cur,λg,λc,m)
         crit=norm(Σ_new-Σ_cur)+abs(τ2_new-τ2_cur)+norm(B_new-B_cur)
         B_cur=B_new;τ2_cur=τ2_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,τ2_cur,Σ_cur,loglik0

end

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,Z,symXs,B_cur,τ2_cur::Float64,Σ_cur,λg,λc,m)
             eStep!(Ghat,Θ,Y,X,Z,B_cur,τ2_cur,Σ_cur,λg,λc,m)
             cmStep!(B_new,dev,Vg,Ve,Y,X,Z,symXs,Ghat,Θ,Σ_cur,λg,λc,m)
             τ2_new = mean(Vg);Σ_new=mean(Ve,dims=3)[:,:,1]
            loglik1=Loglik(dev,Σ_new,τ2_new,λg,λc,m);

    return B_new, τ2_new, Σ_new,loglik1
end

#Z=I
function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(m,p);Vg=zeros(n);Ve=zeros(m,m,n)
    #B0,τ2_0,Σ are arbitrary initial values
    B_cur=B0;τ2_cur=τ2_0;Σ_cur=Σ;loglik0=0.0;tol=tol
    crit=1.0;
    while (crit >=tol)
         B_new,τ2_new,Σ_new,loglik1=fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,τ2_cur,Σ_cur,λg,λc,m)
         crit=norm(Σ_new-Σ_cur)+abs(τ2_new-τ2_cur)+norm(B_new-B_cur)
         B_cur=B_new;τ2_cur=τ2_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,τ2_cur,Σ_cur,loglik0

end

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,τ2_cur::Float64,Σ_cur,λg,λc,m)
             eStep!(Ghat,Θ,Y,X,B_cur,τ2_cur,Σ_cur,λg,λc,m)
             cmStep!(B_new,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,λc,m)
             τ2_new = mean(Vg);Σ_new=mean(Ve,dims=3)[:,:,1]
            loglik1=Loglik(dev,Σ_new,τ2_new,λg,λc,m);

    return B_new, τ2_new, Σ_new,loglik1
end







struct Approx
B::Array{Float64,2}
τ2::Float64
Σ::Array{Float64,2}
loglik::Float64
end

#MVLMM
struct Result
B::Array{Float64,2}
Vc::Array{Float64,2}
Σ::Array{Float64,2}
loglik::Float64
end


#whole ECM procedures grouped in one function: embedded in Nesterov's scheme
# fullECM : update all parameters

function fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,Z,symXs,b1,τ1::Array{Float64,1},Σ1,λg,λc,m;numChr=0,nuMarker=0,niter=0)
             eStep!(Ghat,Θ,Y,X,Z,b1,τ1[1],Σ1,λg,λc,m)
             cmStep!(Bnew,dev,Vg,Ve,Y,X,Z,symXs,Ghat,Θ,Σ1,λg,λc,m)
             τ1 = mean(Vg);Σ1=mean(Ve,dims=3)[:,:,1]
            loglik1=Loglik(dev,Σ1,τ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker,niter=niter);
            τ1=[τ1];
    return Bnew, τ1, Σ1,loglik1
end

#Z=I
function fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,symXs,b1,τ1::Array{Float64,1},Σ1,λg,λc,m;numChr=0,nuMarker=0,niter=0)
             eStep!(Ghat,Θ,Y,X,b1,τ1[1],Σ1,λg,λc,m)
             cmStep!(Bnew,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,λc,m)
             τ1 = mean(Vg);Σ1=mean(Ve,dims=3)[:,:,1]
            loglik1=Loglik(dev,Σ1,τ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker,niter=niter);
            τ1=[τ1];
    return Bnew, τ1, Σ1,loglik1
end



#full Nesterov's
function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},Z::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};ρ=0.001,tol::Float64,numChr=0,nuMarker=0)

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
           # println("Chr $(numChr) and marker $(nuMarker).")
#          itrnum=1
        while (crit >=tol)

            b1, τ1, Σ1, loglik1 = fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,Z,symXs,b1,τ1,Σ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker)
        #,niter=itrnum)
            #some tweak for τ2
                 τ1[1] = max(τ1[1],ρ)
            #Speed restarting Nesterov's Scheme
               updatNestrvAG!(j,b0,b1,b2,τ0,τ1,τ2,Σ0,Σ1,Σ2)

            if (norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(τ0-τ00)+norm(b0-b00)) & (j>=kmin)
                j=1
#                 itrnum+=1
            else
                j=j+1
#                 itrnum+=1
            end

            crit=norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)

            b00=b0;b0=b1;b1=b2; τ00=τ0;τ0=τ1;τ1=τ2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;

        end
        return Approx(b1,τ1[1],Σ1,loglik0)

end


#Z=I
function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};ρ=0.001,tol::Float64,numChr=0,nuMarker=0)

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
           # println("Chr $(numChr) and marker $(nuMarker).")
#          itrnum=1
        while (crit >=tol)

            b1, τ1, Σ1, loglik1 = fullECM(Vg,Ve,Bnew,dev,Ghat,Θ,Y,X,symXs,b1,τ1,Σ1,λg,λc,m;numChr=numChr,nuMarker=nuMarker)
        #,niter=itrnum)
            #some tweak for τ2
                 τ1[1] = max(τ1[1],ρ)
            #Speed restarting Nesterov's Scheme
               updatNestrvAG!(j,b0,b1,b2,τ0,τ1,τ2,Σ0,Σ1,Σ2)

            if (norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)<norm(Σ0-Σ00)+norm(τ0-τ00)+norm(b0-b00)) & (j>=kmin)
                j=1
#                 itrnum+=1
            else
                j=j+1
#                 itrnum+=1
            end

            crit=norm(Σ1-Σ0)+norm(τ1-τ0)+norm(b1-b0)

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
         τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};tol::Float64,ρ=0.001,numChr=0,nuMarker=0)

              if (lod0>0.0)
                result = NestrvAG(kmin,Y,X,Z,B0,τ2_0,Σ,λg,λc;ρ=ρ,tol=tol,numChr=0,nuMarker=0)
                else #keep running ecmLMM
                B0,τ2_0,Σ,loglik=ecmLMM(Y,X,Z,B0,τ2_0,Σ,λg,λc;tol=tol)
                 result = NestrvAG(kmin,Y,X,Z,B0,τ2_0,Σ,λg,λc;ρ=ρ,tol=tol,numChr=0,nuMarker=0)
              end
                return result
 end

#Z=I
function ecmNestrvAG(lod0::Float64,kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
         τ2_0::Float64,Σ::Array{Float64,2},λg::Array{Float64,1},λc::Array{Float64,1};tol::Float64,ρ=0.001,numChr=0,nuMarker=0)

              if (lod0>0.0)
                result = NestrvAG(kmin,Y,X,B0,τ2_0,Σ,λg,λc;ρ=ρ,tol=tol,numChr=0,nuMarker=0)
               else #keep running ecmLMM
                 B0,τ2_0,Σ,loglik = ecmLMM(Y,X,B0,τ2_0,Σ,λg,λc;tol=tol)
                 result = NestrvAG(kmin,Y,X,B0,τ2_0,Σ,λg,λc;ρ=ρ,tol=tol,numChr=0,nuMarker=0)
              end
              return result
    
 end

#MVLMM
#fullECM : update all parameters

function fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B,Vc::Array{Float64,2},Σ,λg,m)

          eStep!(Ghat,Θ,Y,X,B,Vc,Σ,λg,m)
          cmStep!(B_new,dev,Vg,Ve,Y,X,symXs,Ghat,Θ,λg,m)
          Vc_new=mean(Vg,dims=3)[:,:,1]; Σ_new=mean(Ve,dims=3)[:,:,1]
          loglik1=Loglik(dev,Σ_new,Vc_new,λg,m)

    return B_new,Vc_new,Σ_new,loglik1

end



function ecmLMM(Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1};tol::Float64=1e-4)

    symXs=fixX(X)
    m,n = size(Y);p=size(X,1)
    Ghat=zeros(m,n); Θ=Array{Array{Float64,2}}(undef,n);fill!(Θ,zeros(m,m))
    dev=zeros(m,n);B_new=zeros(m,p);Vg=zeros(m,m,n);Ve=zeros(m,m,n)
    #B0,Vc,Σ are arbitrary initial values
    B_cur=B0;Vc_cur=Vc;Σ_cur=Σ;loglik0=0.0;tol=tol

    # i=1;
    crit=1.0;
    while (crit >=tol)
         B_new, Vc_new, Σ_new, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,B_cur,Vc_cur,Σ_cur,λg,m)
         crit=norm(Σ_new-Σ_cur)+norm(Vc_new-Vc_cur)+norm(B_new-B_cur)
         B_cur=B_new;Vc_cur=Vc_new;Σ_cur=Σ_new;loglik0=loglik1;
    end

    return B_cur,Vc_cur,Σ_cur,loglik0

end



function NestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1};ρ=0.001,tol::Float64)

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
             b1, V1, Σ1, loglik1 = fullECM(Vg,Ve,B_new,dev,Ghat,Θ,Y,X,symXs,b1,V1,Σ1,λg,m)
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

            crit=norm(Σ1-Σ0)+norm(V1-V0)+norm(b1-b0)+abs(loglik1-loglik0)
            b00=b0;b0=b1;b1=b2; V00=V0;V0=V1;V1=V2;Σ00=Σ0;Σ0=Σ1; Σ1=Σ2;loglik0=loglik1;
#             i+=1
           end
           return Result(b1,V1,Σ1,loglik0)

end




function ecmNestrvAG(kmin::Int64,Y::Array{Float64,2},X::Array{Float64,2},B0::Array{Float64,2},
        Vc::Array{Float64,2},Σ::Array{Float64,2},λg::Array{Float64,1};tol::Float64,ρ=1e-5)


             result=  NestrvAG(kmin,Y,X,B0,Vc,Σ,λg;tol=tol,ρ=ρ)

              return result

end


### An updating scheme of iterations in Nesterov's accelarated gradient method
#Synopsis: updateNestrvAG!(j,b0,b1,b2,τ0,τ1,τ2,Σ0,Σ1,Σ2) or updateNestrvAG!(j,b0,b1,b2)
#
# Input & Output :
# b2,τ2 (or Vc),Σ2 : updated iterations (or parameters) by previous two iterations (b0,b1,τ0,τ1,Σ0,Σ1)
#
function updatNestrvAG!(j::Int64,b0::Array{Float64,2},b1::Array{Float64,2},b2::Array{Float64,2},
        τ0::Array{Float64,1},τ1::Array{Float64,1},τ2::Array{Float64,1},Σ0::Array{Float64,2},Σ1::Array{Float64,2},Σ2::Array{Float64,2})

    b2[:,:]=b1+(j-1)/(j+2)*(b1-b0);
    τ2[:]=τ1+(j-1)/(j+2)*(τ1-τ0);
    Σ2[:,:]=Σ1+(j-1)/(j+2)*(Σ1-Σ0);

end


#MVLMM
function updatNestrvAG!(j::Int64,b0::Array{Float64,2},b1::Array{Float64,2},b2::Array{Float64,2},
        V0::Array{Float64,2},V1::Array{Float64,2},V2::Array{Float64,2},Σ0::Array{Float64,2},Σ1::Array{Float64,2},Σ2::Array{Float64,2})

    b2[:,:]=b1+(j-1)/(j+2)*(b1-b0);
    V2[:]=V1+(j-1)/(j+2)*(V1-V0);
    Σ2[:,:]=Σ1+(j-1)/(j+2)*(Σ1-Σ0);

end





end