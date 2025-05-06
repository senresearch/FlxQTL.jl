#genoprob
gpr = readdlm("genopr_4way.csv",',');

X1=FlxQTL.Markers(marname,chr,pos,gpr')
@test size(X1.X)== size(gpr')

# for MLM
X2= FlxQTL.Markers(marname,chr,pos,gpr)
#no loco
K3=FlxQTL.kinshipLin(X1.X,4);
@test isposdef(K3)
T2,λ2 =FlxQTL.K2eig(K3)

#loco
K4= FlxQTL.kinshipLoco(kinshipLin,X1,4)
T3,Λ3 = FlxQTL.K2eig(K4,true)


#no loco
lod2,b2,es2=FlxQTL.geneScan(4,T2,Tc,λ2,λc,y,X1,Matrix(1.0I,m,m));
@test sum((lod2.< 0.0))==0
lod,b,es=FlxQTL.geneScan(4,T2,Tc,λ2,λc,y,X1);
@test sum((lod.< 0.0))==0
lod1,b1,es1=FlxQTL.gene1Scan(4,T2,λ2,y,X1);
@test sum(lod1.<0.0)==0.0
@test typeof(b1)==Array{Float64,3}
@test isposdef(es1.Σ)
@test es1.τ2>0.0


@test typeof(b)==Array{Float64,3}
@test typeof(b2) ==Array{Float64,3}
@test es.τ2>0.0 
@test es2.τ2 >0.0
@test isposdef(es.Σ)
@test isposdef(es2.Σ)


#MVLMM
lod3,b3,es3=FlxQTL.geneScan(4,T2,λ2,y,X1)
@test sum((lod3.< 0.0))==0
@test typeof(b3)== Array{Float64,3}
@test isposdef(es3.Vc) 
@test isposdef(es3.Σ)
@test es3.loglik <=0.0

#MLM:mle
mlod, bm,mest0= FlxQTL.mlm1Scan(4,y1,X2,Z)
mlogp,bm,mest0=FlxQTL.mlm1Scan(4,y1,X2,Z;LogP=true)
#Z=I
mlodi, bmi,mest0i= FlxQTL.mlm1Scan(4,y1,X2)
mlogpi,bmi,mest0i=FlxQTL.mlm1Scan(4,y1,X2;LogP=true)

@test sum((mlod.<0.0))==0
@test sum((mlodi.<0.0))==0
for j=eachindex(mlod)
    print(@test mlod[j]≈ mlodi[j])
    print(@test bm[:,:,j]≈ bmi[:,:,j])
    print(@test mlogp[j]≈ mlogpi[j])
end
@test isposdef(mest0.Σ)
@test isposdef(mest0i.Σ)
@test mest0.Σ≈ mest0i.Σ

#MLM;reml
mlodr, bmr,mest0r= FlxQTL.mlm1Scan(4,y1,X2,Z,true)
mlogpr,bmr,mest0r=FlxQTL.mlm1Scan(4,y1,X2,Z,true;LogP=true)
#Z=I
rlodi, bri,rest0i= FlxQTL.mlm1Scan(4,y1,X2,true)
rlogpi,bri,rest0i=FlxQTL.mlm1Scan(4,y1,X2,true;LogP=true)
@test sum((mlodr.<0.0))==0
@test sum((rlodi.<0.0))==0
for j=eachindex(rlodi)
    print(@test mlodr[j]≈ rlodi[j])
    print(@test bmr[:,:,j]≈ bri[:,:,j])
    print(@test mlogpr[j]≈rlogpi[j])
end
@test isposdef(rest0i.Σ)
@test isposdef(mest0r.Σ)
@test rest0i.Σ≈ mest0r.Σ

#mle
mlod2,mes02 = mlm2Scan(4,y1,X2,Z)
mlod2i,mes02i = mlm2Scan(4,y1,X2)
#reml
rlod2,res02 = mlm2Scan(4,y1,X2,Z,true)
rlod2i,res02i = mlm2Scan(4,y1,X2,true)
@test sum(mlod2.<0.0)==0
@test sum(mlod2i.<0.0)==0
@test sum(rlod2.<0.0)==0
@test sum(rlod2i.<0.0)==0
@test mlod2≈ mlod2i
@test rlod2≈ rlod2i
@test isposdef(mes02.Σ)
@test isposdef(mes02i.Σ)
@test mes02.Σ≈ mes02i.Σ
@test isposdef(res02.Σ)
@test isposdef(res02i.Σ)
@test res02.Σ≈ res02i.Σ

mcut1,mxlod,h1p= mlmTest(4,4,y1,X2,Z)
micut1,mxlodi,h1pi= mlmTest(4,4,y1,X2)
rcut1,mxr,h1r= mlmTest(4,4,y1,X2,Z,true)
ricut1,mxri,h1ri= mlmTest(4,4,y1,X2,true)
for j=eachindex(mcut1)
    print(@test mcut1[j]≈ micut1[j])
    print(@test rcut1[j]≈ ricut1[j])
end
for j=eachindex(mxlod)
    print(@test mxlod[j]≈ mxlodi[j])
    print(@test mxr[j]≈ mxri[j])
end
@test sum(mxlod.<0.0)==0
@test sum(mxlodi.<0.0)==0
@test sum(mxr.<0.0)==0
@test sum(mxri.<0.0)==0



#permutation
K1=FlxQTL.getKc(y)
maxlod, H1perm, cutoff1= FlxQTL.permTest(4,4,K3,K1.Kc,y,X1;Z=Z,pval=[0.05,0.01]);
@test sum(maxlod.<0.0)==0
for l=1:2
println(@test isless(0.0,cutoff1[l]))
end

maxlod, H1per0, cutoff2= FlxQTL.permTest(4,4,K3,y,X1;pval=[0.05 0.01])
@test sum(maxlod.<0.0)==0
for l=1:2
println(@test isless(0.0,cutoff2[l]))
end


#loco
lod1,b1,es01=FlxQTL.geneScan(4,T3,Tc,Λ3,λc,y,X1,true);
@test sum((lod1.< 0.0))==0.0
lod0,b0,es00=FlxQTL.geneScan(4,T3,Tc,Λ3,λc,y,X1,Z,true);
@test sum((lod0.< 0.0))==0.0
lod4,b4,es04=FlxQTL.gene1Scan(4,T3,Λ3,y,X1,true);
@test sum(lod4.<0.0)==0.0
 for i=axes(Λ3,2)
    print( @test isposdef(es04[i].Σ))
    print(@test es04[i].τ2>0.0)
    print(@test es00[i].τ2 >0.0)
     print(@test es01[i].τ2>0.0)
     print(@test isposdef(es00[i].Σ))
    print(@test isposdef(es01[i].Σ))
 end
@test typeof(b4)==Array{Float64,3}
@test typeof(b0)==Array{Float64,3}
@test typeof(b1)==Array{Float64,3}




#MVLMM
lod4,b4,es4=FlxQTL.geneScan(4,T3,Λ3,y,X1,true)
@test sum(lod4.<0.0)==0
@test typeof(b4)== Array{Float64,3}
for i=axes(Λ3,2)
    print(@test isposdef(es4[i].Vc))
    print(@test isposdef(es4[i].Σ))
    print(@test es4[i].loglik<=0.0)
end

#
# #2d-scan
# gen3=readdlm("genopr_3way.txt"); #12 x 1212
# lnY= readdlm("log_16wk_weights.txt"); #16 x 1212
# Kin3= readdlm("kinship_3way.csv",','); 
# Kin = [Kin3;;;Kin3];
# X7 = Markers(marname,chr,pos,gen3);
# m,n =size(lnY)
# K3=getKc(lnY);
# @test isposdef(K3.Kc)
# T2,λ2,Tc,λc = FlxQTL.K2Eig(Kin,K3.Kc,true);

# #no loco
#  lod2d,es2d=FlxQTL.gene2Scan(3,T2[:,:,1],Tc,λ2[:,1],λc,lnY,X7,Matrix(1.0I,m,m));
# @test sum(lod2d.<0.0)==0
# @test es2d.τ2 >0.0
# @test isposdef(es2d.Σ)


# #MVLMM
# lod2d0,es2d0=FlxQTL.gene2Scan(3,T2[:,:,1],λ2[:,1],lnY,X7)
# @test sum(lod2d0.<0.0)==0
# @test isposdef(es2d0.Vc)
# @test isposdef(es2d0.Σ)



# lod2d,es2d=FlxQTL.gene2Scan(3,T2[:,:,1],λ2[:,1],lnY,X7,Matrix(1.0I,m,m));
# @test sum(lod2d.<0.0)==0
# @test es2d.τ2 >0.0
# @test isposdef(es2d.Σ)


# #loco
# Lod2d,Es2d=FlxQTL.gene2Scan(3,T2,Tc,λ2,λc,lnY,X7,Matrix(1.0I,m,m),true);
# @test sum(lod2d.<0.0)==0


# #MVLMM
# lod2d0,es2d0=FlxQTL.gene2Scan(3,T2,λ2,lnY,X7,true)
# @test sum(lod2d0.<0.0)==0
#  for j=axes(Λ3,2)
#     print(@test Es2d[j].τ2 >0.0)
#     print(@test isposdef(Es2d[j].Σ))
#     print(@test isposdef(es2d0[j].Vc))
#     print(@test isposdef(es2d0[j].Σ))
   
#  end


