
#test for cross=1 (genotype data)
geno=[0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0
 1.0  1.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 1.0  0.0  1.0  0.0
 0.0  1.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  1.0  1.0  0.0
 0.0  1.0  0.0  0.0
 1.0  1.0  0.0  1.0
 1.0  0.0  1.0  1.0
 0.0  0.0  1.0  1.0
 1.0  0.0  0.0  0.0
 1.0  1.0  0.0  1.0
 0.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0
 1.0  0.0  0.0  0.0]

y=[3.0  19.7616   24.1761   15.6778;
4.0   6.28846   1.77401   3.23889;
5.0  12.034    12.4605   10.601;
6.0  20.1253   14.1169   12.9035;
7.0  15.1278   13.2925   15.0646;
8.0  19.2091   13.9549   12.9868;
9.0  12.9434   11.9672   11.4831;
10.0  14.2244    5.45326   5.124;
11.0  20.5921   14.8691   11.1797;
12.0  15.6141   12.0889   10.2923;
13.0  10.9073    5.67946   6.61925;
14.0  11.8156    8.59027   8.35106;
15.0  13.0657    7.62999  10.055;
16.0  13.9874    9.36124   7.45863;
17.0  16.9105   17.3268   13.2894;
18.0  15.3467   10.772    10.4977;
19.0  13.3412    3.87934   7.89231;
20.0  12.1524    9.46482  10.8092;
21.0  16.2286    8.80221   8.11992;
22.0  14.7742    6.47541  15.525]'
y=Float64.(y)
marname=["X1" ;"X2";"X3";"X4"]
chr=Any[1;1;2;2]
pos=[1.4;2.0;3.44;4.25]
XX=FlxQTL.Markers(marname,chr,pos,geno')
@test XX.name == marname
@test XX.chr == chr
@test size(XX.X)== size(geno')

#test shrinkgLoco with kinshipMan, shrinkg
 Kg=FlxQTL.shrinkgLoco(FlxQTL.GRM.kinshipMan,6,XX)
for j=1:2
      println(@test isposdef(Kg[:,:,j])== true)
end

K= FlxQTL.shrinkg(FlxQTL.GRM.kinshipMan,6,XX.X)
@test isposdef(K)==true
#precompute Kc
K1= getKc(y)
Z1=hcat(ones(4),vcat(-ones(2),ones(2)));
K0=getKc(y;Z=Z1)
@test isposdef(K1.Kc)
@test isposdef(K0.Kc)
@test isposdef(K1.Σ)
@test isposdef(K0.Σ)
@test K0.τ2 > 0.0
@test K1.τ2 >0.0


#eigen decomposition
Tg,Λg,Tc,λc = FlxQTL.K2Eig(Kg,K1.Kc,true)
T,λ =FlxQTL.K2eig(K)
T1,λ1=K2eig(K0.Kc)
@test typeof(Tg)==Array{Float64,3}
@test typeof(T)==Array{Float64,2}
@test typeof(Λg)==Array{Float64,2}
@test typeof(λ)==Array{Float64,1}
@test typeof(λc)== Array{Float64,1}
@test typeof(λ1)==Array{Float64,1}
@test typeof(T1)==Array{Float64,2}

#test Z=I vs no Z & loco vs no loco
Z=Matrix(1.0I,4,4)


#no loco
LOD2,B2,est2=FlxQTL.geneScan(1,T,Tc,λ,λc,y,XX,Z);
@test sum((LOD2.< 0.0))==0
LOD,B,est=FlxQTL.geneScan(1,T,Tc,λ,λc,y,XX);
@test sum((LOD.< 0.0))==0
LOD_1,B_2,est_2=FlxQTL.geneScan(1,T,Matrix(1.0I,3,3),λ,ones(3),y,XX,Z); 
@test sum((LOD_1.< 0.0))==0
lod2,b2,est02=gene1Scan(1,T,λ,XX,Z);
@test sum(lod2.<0.0)==0
@test est02.τ2 >0.0
@test typeof(b2)==Array{Float64,3}
@test typeof(B_2)==Array{Float64,3}
@test LOD ≈ LOD2
@test B ≈ B2
@test est.τ2 ≈ est2.τ2
@test est.Σ ≈ est2.Σ
@test est.loglik ≈ est2.loglik

#MVLMM
LOD3,B3,est3=FlxQTL.geneScan(1,T,λ,y,XX)
@test sum((LOD3.< 0.0))==0
@test typeof(B3)== Array{Float64,3}
@test isposdef(est3.Vc)
@test isposdef(est3.Σ)
@test est3.loglik <=0.0

#environment scan
Q=findall(LOD2.==maximum(LOD2))
Ze=[ -1.80723   -1.33892   -0.625303  -0.164235   0.490013
 -1.48507   -1.18942   -1.1961    -0.417583  -0.125115
 -0.749826  -0.327169   0.20022    0.158106   0.993343]
eLOD,eB,este =FlxQTL.envScan(Q,1,T,Tc,λ,λc,y,XX,Ze)
@test sum(eLOD.<0.0)==0.0
@test typeof(eB)==Array{Float64,3}
@test este[1].τ2>0.0
@test isposdef(este[1].Σ)
@test este[1].loglik<=0.0

#loco
LOD1,B1,est01=FlxQTL.geneScan(1,Tg,Tc,Λg,λc,y,XX,true);
@test sum((LOD1.< 0.0))==0.0
LOD0,B0,est00=FlxQTL.flxMLMM.geneScan(1,Tg,Tc,Λg,λc,y,XX,Z,true);
@test sum((LOD0.< 0.0))==0.0
lod1,b1,est1 =FlxQTL.gene1Scan(1,Tg,Λg,y,XX,Z1,true);
lod11,b11,est11 =FlxQTL.gene1Scan(1,Tg,Λg,y,XX,true);
@test sum(lod1.<0.0)==0.0
@test sum(lod11.<0.0)==0.0
@test typeof(b1)==Array{Float64,3}
@test typeof(b11)==Array{Float64,3}
for j=1:2
       println(@test isposdef(est1[j].Σ)==true)
       println(@test isposdef(est1[j].τ2>0.0))
end

for j=1:2
       println(@test isposdef(est11[j].Σ)==true)
       println(@test  est11[j].τ2>0.0)
end
@test LOD0 ≈ LOD1
@test B0 ≈ B1
for j=1:2
       println(@test est00[j].τ2 ≈ est01[j].τ2)
       println(@test est00[j].Σ ≈ est01[j].Σ)
       println(@test est00[j].loglik ≈ est01[j].loglik)
end
#MVLMM
LOD4,B4,est4=FlxQTL.geneScan(1,Tg,Λg,y,XX,true)
@test sum(LOD4.<0.0)==0
@test size(B4)== (3,2,4)
for j=1:2
       println(@test isposdef(est4[j].Vc)==true)
       println(@test isposdef(est4[j].Σ)==true )
       println(@test est4[j].loglik<=0.0)
end

#environment scan
Q=findall(LOD0.==maximum(LOD0))
eLOD0,eB0,este0 =FlxQTL.envScan(Q,1,Tg,Tc,Λg,λc,y,XX,Ze,true)
@test sum(eLOD0.<0.0)==0.0
@test typeof(eB0)== Array{Float64,3}
@test este0[1].τ2>0.0
@test isposdef(este0[1].Σ)
@test este0[1].loglik<=0.0


#2d-scan
#no loco
 LOD2d,est2d=FlxQTL.gene2Scan(1,T,Tc,λ,λc,y,XX,Z);
@test sum(LOD2d.<0.0)==0
@test est2d.τ2 >0.0
@test isposdef(est2d.Σ)
@test est2d.loglik <=0.0

#MVLMM
LOD2d0,est2d0=FlxQTL.gene2Scan(1,T,λ,y,XX)
@test sum(LOD2d0.<0.0)==0
@test isposdef(est2d0.Vc)
@test isposdef(est2d0.Σ)
@test est2d0.loglik <=0.0

#loco
 LOD2d1,est2d1=FlxQTL.gene2Scan(1,Tg,Tc,Λg,λc,y,XX,Z,true);
@test sum(LOD2d1.<0.0)==0
for j=1:2
       println(@test est2d1[j].τ2 >0.0)
       println(@test isposdef(est2d1[j].Σ)==true )
       println(@test est2d1[j].loglik<=0.0)
end

#MVLMM
 LOD2d2,est2d2=FlxQTL.gene2Scan(1,Tg,Λg,y,XX,true);
@test sum(LOD2d2.<0.0)==0
for j=1:2
       println(@test isposdef(est2d2[j].Vc)==true)
       println(@test isposdef(est2d2[j].Σ)==true )
       println(@test est2d2[j].loglik<=0.0)
end


#permutation
maxLODs, H1par_perm, cutoff= FlxQTL.permTest(4,1,K,Kc,y,XX,Z;pval=[0.05,0.01]);
@test sum(maxLODs.<0.0)==0
for j=1:2
println(@test isless(0.0,cutoff[j]))
end


maxLODs0, H1par_perm0, cutoff0= FlxQTL.permTest(4,1,K,y,XX;pval=[0.05 0.01])
@test sum(maxLODs0.<0.0)==0
for j=1:2
println(@test isless(0.0,cutoff[j]))
end

#######testing kinships

@test isposdef(FlxQTL.kinshipGs(geno,std(geno)))


K0=FlxQTL.kinshipLoco(FlxQTL.kinshipCtr,XX)
for j=1:2
      println(@test isposdef(K0[:,:,j])== true)
end

K1=FlxQTL.kinshipLoco(FlxQTL.kinshipStd,XX)
for j=1:2
      println(@test isposdef(K1[:,:,j])== true)
end

### testing maf
A=rand(15,15)
Aidx= getGenoidx(A,0.25)
@test length(Aidx)<=size(A,1)
