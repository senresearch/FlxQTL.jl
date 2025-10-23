
## cross = 1 with larger data.
geno=readdlm("../data/Arabidopsis_genotypes_2d.csv",',';skipstart=1)[1:4,:];
y=readdlm("../data/Arabidopsis_fitness.csv",',';skipstart=1); # 400 x 6
K = readdlm("./kinship_genotypes.txt");
Kg=cat(K,K,dims=3);
 m=size(y,2)
marname=["X1" ;"X2";"X3";"X4"]
chr=Any[1;1;2;2]
pos=[1.4;2.0;3.44;4.25]
XX=FlxQTL.Markers(marname,chr,pos,geno)
y=Float64.(y');
#eigen decomposition
Tg,Λg = FlxQTL.K2eig(Kg,true)
T,λ =FlxQTL.K2eig(K)
# T1,λ1=K2eig(K0.Kc)
@test typeof(Tg)==Array{Float64,3}
@test typeof(T)==Array{Float64,2}
@test typeof(Λg)==Array{Float64,2}
@test typeof(λ)==Array{Float64,1}
# @test typeof(λc)== Array{Float64,1}
# @test typeof(λ1)==Array{Float64,1}
# @test typeof(T1)==Array{Float64,2}

#test Z=I vs no Z & loco vs no loco
Z=Matrix(1.0I,m,m)
Z1=hcat(ones(m),vcat(-ones(3),ones(3)));



#no loco
LOD2,B2,est2=FlxQTL.geneScan(1,T,λ,y,XX,Z1;H0_up=true);
@test sum((LOD2.< 0.0))==0
LOD,B,est=FlxQTL.geneScan(1,T,λ,y,XX;H0_up=true);
@test sum((LOD.< 0.0))==0

@test typeof(B2)==Array{Float64,3}
@test typeof(B)==Array{Float64,3}
@test est.τ2 >0.0 
@test est2.τ2 >0.0 
@test isposdef(est.Σ)
@test isposdef(est2.Σ)


#MVLMM
LOD3,B3,est3=FlxQTL.geneScan(T,λ,y,XX,1)
@test sum((LOD3.< 0.0))==0
@test typeof(B3)== Array{Float64,3}
@test isposdef(est3.Vc)
@test isposdef(est3.Σ)
@test est3.loglik <=0.0

# #environment scan
# Q=findall(LOD2.==maximum(LOD2))
# Ze=[ -1.80723   -1.33892   -0.625303  -0.164235   0.490013
#  -1.48507   -1.18942   -1.1961    -0.417583  -0.125115
#  -0.749826  -0.327169   0.20022    0.158106   0.993343
#  1.01404  0.47499  1.76577  -1.65295  1.41504]
# eLOD,eB,este =FlxQTL.envScan(Q,1,T,Tc,λ,λc,y,XX,Ze)
# @test sum(eLOD.<0.0)==0.0
# @test typeof(eB)==Array{Float64,3}
# @test este[1].τ2>0.0
# @test isposdef(este[1].Σ)
# @test este[1].loglik<=0.0

#loco
LOD1,B1,est01=FlxQTL.geneScan(1,Tg,Λg,y,XX,true;H0_up=true);
@test sum((LOD1.< 0.0))==0.0
LOD0,B0,est00=FlxQTL.flxMLMM.geneScan(1,Tg,Λg,y,XX,Z1,true;H0_up=true);
@test sum((LOD0.< 0.0))==0.0

@test typeof(B0)==Array{Float64,3}
@test typeof(B1)==Array{Float64,3}
for j=1:2
       println(@test est00[j].τ2 >0.0)
       println(@test est01[j].τ2 >0.0 )
       println(@test isposdef(est00[j].Σ))
       println(@test isposdef(est01[j].Σ))
end
#MVLMM
LOD4,B4,est4=FlxQTL.geneScan(Tg,Λg,y,XX,1,true)
@test sum(LOD4.<0.0)==0
@test typeof(B4)==Array{Float64,3}
for j=1:2
       println(@test isposdef(est4[j].Vc)==true)
       println(@test isposdef(est4[j].Σ)==true )
       println(@test est4[j].loglik<=0.0)
end

# #environment scan
# Q=findall(LOD0.==maximum(LOD0))
# eLOD0,eB0,este0 =FlxQTL.envScan(Q,1,Tg,Tc,Λg,λc,y,XX,Ze,true)
# @test sum(eLOD0.<0.0)==0.0
# @test typeof(eB0)== Array{Float64,3}
# @test este0[1].τ2>0.0
# @test isposdef(este0[1].Σ)
# @test este0[1].loglik<=0.0


#2d-scan
#no loco
 LOD2d,est2d=FlxQTL.gene2Scan(1,T,λ,y,XX);
@test sum(LOD2d.<0.0)==0
@test est2d.τ2 >0.0
@test isposdef(est2d.Σ)
@test est2d.loglik <=0.0

#MVLMM
LOD2d0,est2d0=FlxQTL.gene2Scan(T,λ,y,XX,1)
@test sum(LOD2d0.<0.0)==0
@test isposdef(est2d0.Vc)
@test isposdef(est2d0.Σ)
@test est2d0.loglik <=0.0

#loco
 LOD2d1,est2d1=FlxQTL.gene2Scan(1,Tg,Λg,y,XX,true);
@test sum(LOD2d1.<0.0)==0
for j=1:2
       println(@test est2d1[j].τ2 >0.0)
       println(@test isposdef(est2d1[j].Σ)==true )
       println(@test est2d1[j].loglik<=0.0)
end

#MVLMM
 LOD2d2,est2d2=FlxQTL.gene2Scan(Tg,Λg,y,XX,1,true);
@test sum(LOD2d2.<0.0)==0
for j=1:2
       println(@test isposdef(est2d2[j].Vc)==true)
       println(@test isposdef(est2d2[j].Σ)==true )
       println(@test est2d2[j].loglik<=0.0)
end

#permutation: no loco
#MVLMM
maxLODs0, H1par_perm0, cutoff0= FlxQTL.mlmmTest(4,1,K,y,XX;pval=[0.10 0.05],df_prior=Int64(ceil(1.9m)))
@test sum(maxLODs0.<0.0)==0
for j=1:2
println(@test isless(0.0,cutoff0[j]))
end

# maxLODs, H1par_perm, cutoff= FlxQTL.permTest(4,1,K,y,XX;pval=[0.10 0.05],df_prior=Int64(ceil(1.9m)));
# @test sum(maxLODs.<0.0)==0
# for j=1:2
# println(@test isless(0.0,cutoff[j]))
# end

#permutation: loco

# mlods,h1par,cuts = permutationTest(4,1,Kg,y,XX;df_prior=Int64(ceil(1.9m)))
# @test sum(mlods.<0.0)==0
# for j=1:2
#        println(@test isless(0.0,cuts[j]))
# end

# mlods0,h1par0,cuts0 = permutationTest(4,1,Kg,y,XX;LOCO_all=true,df_prior=Int64(ceil(1.9m)))
# @test sum(mlods0.<0.0)==0
# for j=1:2
#        println(@test isless(0.0,cuts0[j]))
# end


