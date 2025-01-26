#genoprob
gpr=[4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 1.21848e-6  4.44965e-9  0.999998    1.21848e-6  2.23014e-12  8.34238e-16  1.0          2.23014e-12  2.25839e-12  8.34192e-16  1.0          2.25839e-12
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 1.21848e-6  0.999998    4.44965e-9  1.21848e-6  2.23014e-12  1.0          8.34238e-16  2.23014e-12  2.25839e-12  1.0          8.34192e-16  2.25839e-12
 0.999998    1.21848e-6  1.21848e-6  4.44965e-9  1.0          2.23014e-12  2.23014e-12  8.34238e-16  1.0          2.25839e-12  2.25839e-12  8.34192e-16
 0.999998    1.21848e-6  1.21848e-6  4.44965e-9  1.0          2.23014e-12  2.23014e-12  8.34238e-16  1.0          2.25839e-12  2.25839e-12  8.34192e-16
 0.999998    1.21848e-6  1.21848e-6  4.44965e-9  1.0          2.23014e-12  2.23014e-12  8.34238e-16  1.0          2.25839e-12  2.25839e-12  8.34192e-16
 2.52546e-5  0.999973    2.03526e-7  1.21937e-6  0.000287185  0.999713     1.16563e-7   6.15819e-9   0.000315571  0.999684     1.1656e-7    6.76649e-9
 1.21848e-6  4.44965e-9  0.999998    1.21848e-6  2.23014e-12  8.34238e-16  1.0          2.23014e-12  2.25839e-12  8.34192e-16  1.0          2.25839e-12
 1.21848e-6  0.999998    4.44965e-9  1.21848e-6  2.23014e-12  1.0          8.34238e-16  2.23014e-12  2.25839e-12  1.0          8.34192e-16  2.25839e-12
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 0.999998    1.21848e-6  1.21848e-6  4.44965e-9  1.0          2.23014e-12  2.23014e-12  8.34238e-16  1.0          2.25839e-12  2.25839e-12  8.34192e-16
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 1.21848e-6  4.44965e-9  0.999998    1.21848e-6  2.23014e-12  8.34238e-16  1.0          2.23014e-12  2.25839e-12  8.34192e-16  1.0          2.25839e-12
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 1.21848e-6  0.999998    4.44965e-9  1.21848e-6  2.23014e-12  1.0          8.34238e-16  2.23014e-12  2.25839e-12  1.0          8.34192e-16  2.25839e-12
 1.21848e-6  0.999998    4.44965e-9  1.21848e-6  2.23014e-12  1.0          8.34238e-16  2.23014e-12  2.25839e-12  1.0          8.34192e-16  2.25839e-12
 0.999998    1.21848e-6  1.21848e-6  4.44965e-9  1.0          2.23014e-12  2.23014e-12  8.34238e-16  1.0          2.25839e-12  2.25839e-12  8.34192e-16
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
 4.44965e-9  1.21848e-6  1.21848e-6  0.999998    8.34238e-16  2.23014e-12  2.23014e-12  1.0          8.34192e-16  2.25839e-12  2.25839e-12  1.0
]

mname=["X1" ;"X2";"X3"]
chr1=Any[1;1;2]
pos1=[1.4;2.22;5.44]
X1=FlxQTL.Markers(mname,chr1,pos1,gpr')
@test X1.name == mname
@test X1.chr == chr1
@test size(X1.X)== size(gpr')


K3=FlxQTL.kinshipLin(X1.X,4);
@test isposdef(K3)

T2,λ2 =FlxQTL.K2eig(K3)

#for loco
chr0=Any[1;1;1]
X0=Markers(mname,chr0,pos1,gpr')

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
lod1,b1,es01=FlxQTL.geneScan(4,T2,Tc,λ2,λc,y,X0,true);
@test sum((lod1.< 0.0))==0.0
lod0,b0,es00=FlxQTL.geneScan(4,T2,Tc,λ2,λc,y,X0,Z,true);
@test sum((lod0.< 0.0))==0.0
lod4,b4,es04=FlxQTL.gene1Scan(4,T2,λ2,y,X0,true);
@test sum(lod4.<0.0)==0.0
 i=1
 @test isposdef(es04[i].Σ)
 @test es04[i].τ2>0.0
@test typeof(b4)==Array{Float64,3}
@test typeof(b0)==Array{Float64,3}
@test typeof(b1)==Array{Float64,3}

println(@test es00[i].τ2 >0.0)
println(@test es01[i].τ2>0.0)
println(@test isposdef(es00[i].Σ))
println(@test isposdef(es01[i].Σ))

#MVLMM
lod4,b4,es4=FlxQTL.geneScan(4,T2,λ2,y,X0,true)
@test sum(lod4.<0.0)==0
@test typeof(b4)== Array{Float64,3}

@test isposdef(es4[i].Vc)
@test isposdef(es4[i].Σ)
@test es4[i].loglik<=0.0

#
#2d-scan
#no loco
 lod2d,es2d=FlxQTL.gene2Scan(4,T2,Tc,λ2,λc,y,X1,Z);
@test sum(lod2d.<0.0)==0
@test es2d.τ2 >0.0
@test isposdef(es2d.Σ)
@test es2d.loglik <=0.0

#MVLMM
lod2d0,es2d0=FlxQTL.gene2Scan(4,T2,λ2,y,X1)
@test sum(lod2d0.<0.0)==0
@test isposdef(es2d0.Vc)
@test isposdef(es2d0.Σ)
@test es2d0.loglik <=0.0


lod2d,es2d=FlxQTL.gene2Scan(4,T2,λ2,y,X1,Z);
@test sum(lod2d.<0.0)==0
@test es2d.τ2 >0.0
@test isposdef(es2d.Σ)
@test es2d.loglik <=0.0