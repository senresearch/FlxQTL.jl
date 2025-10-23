
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

y1=[3.0  19.7616   24.1761   15.6778;
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
22.0  14.7742    6.47541  15.525]
y=Float64.(y1'); m=size(y,1)
marname=["X1" ;"X2";"X3";"X4"]
chr=Any[1;1;2;2]
pos=[1.4;2.0;3.44;4.25]
XX=FlxQTL.Markers(marname,chr,pos,geno')
@test XX.name == marname
@test XX.chr == chr
@test size(XX.X)== size(geno')
# for MLM setup
XX1=FlxQTL.Markers(marname,chr,pos,geno)

#test shrinkgLoco with kinshipMan, shrinkg
 Kg=FlxQTL.shrinkgLoco(FlxQTL.GRM.kinshipMan,15,XX)
for j=1:2
      println(@test isposdef(Kg[:,:,j])== true)
end

K= FlxQTL.shrinkg(FlxQTL.GRM.kinshipMan,15,XX.X)
@test isposdef(K)
#######testing kinships

@test isposdef(FlxQTL.kinshipGK(geno,std(geno)))

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


Z=Matrix(1.0I,m,m)
Z1=hcat(ones(m),vcat(-ones(2),ones(2)));

#MLM:mle
mlod, bm,mest0= FlxQTL.mlm1Scan(1,y1,XX1,Z)
mlogp,bm,mest0=FlxQTL.mlm1Scan(1,y1,XX1,Z;LogP=true)
#Z=I
mlodi, bmi,mest0i= FlxQTL.mlm1Scan(1,y1,XX1)
mlogpi,bmi,mest0i=FlxQTL.mlm1Scan(1,y1,XX1;LogP=true)

@test sum((mlod.<0.0))==0
@test sum((mlodi.<0.0))==0
for j=eachindex(mlod)
       print(@test isapprox(mlod[j],mlodi[j];atol=0.01))
       print(@test isapprox(bm[:,:,j],bmi[:,:,j];atol=0.01))
       print(@test isapprox(mlogp[j],mlogpi[j];atol=0.01))
end
@test isposdef(mest0.Σ)
@test isposdef(mest0i.Σ)
@test mest0.Σ≈ mest0i.Σ

#MLM;reml
mlodr, bmr,mest0r= FlxQTL.mlm1Scan(1,y1,XX1,Z,true)
mlogpr,bmr,mest0r=FlxQTL.mlm1Scan(1,y1,XX1,Z,true;LogP=true)
#Z=I
rlodi, bri,rest0i= FlxQTL.mlm1Scan(1,y1,XX1,true)
rlogpi,bri,rest0i=FlxQTL.mlm1Scan(1,y1,XX1,true;LogP=true)
@test sum((mlodr.<0.0))==0
@test sum((rlodi.<0.0))==0
for j=eachindex(mlodr)
       println(@test mlodr[j]≈rlodi[j])
       println(@test mlogpr[j]≈rlogpi[j] )
       println(@test bmr[:,:,j]≈ bri[:,:,j])
end


@test isposdef(mest0r.Σ)
@test isposdef(rest0i.Σ)
@test mest0r.Σ≈rest0i.Σ

#mle
mlod2,mes02 = mlm2Scan(1,y1,XX1,Z)
mlod2i,mes02i = mlm2Scan(1,y1,XX1)
#reml
rlod2,res02 = mlm2Scan(1,y1,XX1,Z,true)
rlod2i,res02i = mlm2Scan(1,y1,XX1,true)
@test sum(mlod2.<0.0)==0
@test sum(mlod2i.<0.0)==0
@test mlod2≈ mlod2i
@test sum(rlod2.<0.0)==0
@test sum(rlod2i.<0.0)==0
@test rlod2≈ rlod2i
@test isposdef(mes02.Σ)
@test isposdef(mes02i.Σ)
@test mes02.Σ≈ mes02i.Σ
@test isposdef(res02.Σ)
@test isposdef(res02i.Σ)
@test res02.Σ≈ res02i.Σ

mxlod,h1p,mcut1= mlmTest(1,1,y1,XX1,Z)
mxlodi,h1pi,micut1= mlmTest(1,1,y1,XX1)
mxr,h1r,rcut1= mlmTest(1,1,y1,XX1,Z,true)
mxri,h1ri,ricut1= mlmTest(1,1,y1,XX1,true)

       print(@test isless(0.0,mcut1[1]))
       print(@test isless(0.0, micut1[1]))
       print(@test isless(0.0,mcut1[2])) 
       print(@test isless(0.0,micut1[2]))
       print(@test isless(0.0,rcut1[1])) 
       print(@test isless(0.0,ricut1[1]))
       print(@test isless(0.0,rcut1[2])) 
       print(@test isless(0.0,ricut1[2]))


@test sum(mxlod.<0.0)==0
@test sum(mxlodi.<0.0)==0
@test sum(mxr.<0.0)==0
@test sum(mxri.<0.0)==0



