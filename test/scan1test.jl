using Distributed, LinearAlgebra
addprocs(4)
@everywhere using FlxQTL, Random,Test

FlxQTL.Util.setSeed(2,100);
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

pheno=[ 3.0719   -2.42177  -0.0790196  3.22021  1.18245  2.84663  0.291765   0.814116  3.41052   1.38076   -0.538445  -0.168097   0.341638   0.717475  1.90936  1.27174    0.453986  -0.0307716   1.63132    1.03828;
 5.14641  -2.01593   1.40071    1.93031  1.66674  1.87851  1.24301   -0.839608  2.17081   1.28193   -0.767287   0.163348  -0.143671   0.40984   2.95656  0.860896  -1.34282    0.442958    0.23111   -0.512809;
 2.66991  -2.24506   0.663908   1.57371  2.42762  1.60664  1.01247   -1.5002    0.892569  0.541949  -0.909383  -0.225096   0.448193  -0.577719  1.72619  0.623101  -0.406358   0.746183   -0.316426   2.60954]
marname=["X1" ;"X2";"X3";"X4"]
chr=Any[1;1;2;2]
pos=[1.4;2.0;3.44;4.25]
XX=FlxQTL.Util.Markers(marname,chr,pos,geno')
@test XX.name == marname
@test XX.chr == chr
@test size(XX.X)== size(geno')

#test shrinkgLoco with kinshipMan, shrinkg
Kg=FlxQTL.GRM.shrinkgLoco(FlxQTL.GRM.kinshipMan,6,XX)
for j=1:2 
      println(@test isposdef(Kg[:,:,j])== true)
end

K= FlxQTL.GRM.shrinkg(FlxQTL.GRM.kinshipMan,6,XX.X)
@test isposdef(K)==true
Kc=Matrix(1.0I,3,3)

#eigen decomposition
Tg,Λg,Tc,λc = FlxQTL.flxMLMM.K2Eig(Kg,Kc,true)
T,λ =FlxQTL.flxMLMM.K2eig(K)

#test Z=I vs no Z & loco vs no loco
Z=Matrix(1.0I,3,3)


#no loco 
LOD2,B2,est2=FlxQTL.flxMLMM.geneScan(1,T,Tc,λ,λc,pheno,XX,Z); 
@test sum((LOD2.< 0.0))==0
LOD,B,est=FlxQTL.flxMLMM.geneScan(1,T,Tc,λ,λc,pheno,XX); 
@test sum((LOD.< 0.0))==0

@test LOD ≈ LOD2
@test B ≈ B2
@test est.τ2 ≈ est2.τ2
@test est.Σ ≈ est2.Σ 
@test est.loglik ≈ est2.loglik

#MVLMM
LOD3,B3,est3=FlxQTL.geneScan(1,T,λ,pheno,XX)
@test sum((LOD3.< 0.0))==0
@test size(B3)== (2,3,4)
@test est3.Vc >0.0
@test size(est3.Σ)==(3,3)
@test est3.loglik <=0.0

#loco
LOD1,B1,est01=FlxQTL.flxMLMM.geneScan(1,Tg,Tc,Λg,λc,pheno,XX,true); 
@test sum((LOD1.< 0.0))==0.0
LOD0,B0,est00=FlxQTL.flxMLMM.geneScan(1,Tg,Tc,Λg,λc,pheno,XX,Z,true); 
@test sum((LOD0.< 0.0))==0.0

@test LOD0 ≈ LOD1
@test B0 ≈ B1
for j=1:2 
       println(@test est00[j].τ2 ≈ est01[j].τ2)
end
for j=1:2 
       println(@test est00[j].Σ ≈ est01[j].Σ)
end
 for j=1:2 
       println(@test est00[j].loglik ≈ est01[j].loglik)
end
#MVLMM
LOD4,B4,est4=FlxQTL.geneScan(1,Tg,Λg,pheno,XX,true)
@test sum(LOD4.<0.0)==0
@test size(B4)== (2,3,4)
for j=1:2 
       println(@test est4[j].Vc >0.0)
end
for j=1:2 
       println(@test size(est4[j].Σ)== (3,3) )
end
 for j=1:2 
       println(@test est4[j].loglik<=0.0)
end


#2d-scan
#no loco
 LOD2d,est2d=FlxQTL.gene2Scan(1,T,Tc,λ,λc,pheno,XX,Z);
@test sum(LOD2d.<0.0)==0
@test est2d.τ2 >0.0
@test size(est2d.Σ)== (3,3)
@test est2d.loglik <=0.0

#MVLMM
LOD2d0,est2d0=FlxQTL.gene2Scan(1,T,λ,pheno,XX)
@test sum(LOD2d0.<0.0)==0
@test est2d0.Vc >0.0
@test size(est2d0.Σ)== (3,3)
@test est2d0.loglik <=0.0                      

#loco
 LOD2d1,est2d1=FlxQTL.gene2Scan(1,Tg,Tc,Λg,λc,pheno,XX,Z,true);
@test sum(LOD2d1.<0.0)==0
for j=1:2 
       println(@test est2d1[j].τ2 >0.0)
end
for j=1:2 
       println(@test size(est2d1[j].Σ)== (3,3) )
end
 for j=1:2 
       println(@test est2d1[j].loglik<=0.0)
end

#MVLMM
 LOD2d2,est2d2=FlxQTL.gene2Scan(1,Tg,Λg,pheno,XX,Z,true);
@test sum(LOD2d2.<0.0)==0
for j=1:2 
       println(@test est2d2[j].Vc >0.0)
end
for j=1:2 
       println(@test size(est2d2[j].Σ)== (3,3) )
end
 for j=1:2 
       println(@test est2d2[j].loglik<=0.0)
end


#permutation
maxLODs, H1par_perm, cutoff= FlxQTL.permTest(4,1,K,Kc,pheno,XX,Z;pval=[0.01]);
@test sum(maxLODs.<0.0)==0
@test cutoff > 0.0


maxLODs0, H1par_perm0, cutoff0= FlxQTL.permTest(4,1,K,pheno,XX;pval=[0.01])
@test sum(maxLODs0.<0.0)==0
@test cutoff0 > 0.0
