"""

    FlxQTL

flexible QTL analysis tools for structured multiple traits fitting a Multivariate Linear Mixed Model or a Multivariate 
    Linear Model.

"""
module FlxQTL

# # Write your package code here.
 __precompile__(true)



include("EcmNestrv.jl")
include("Miscellanea.jl")
include("MLM.jl")
include("flxMLM.jl")
include("GRM.jl")
include("flxMLMM.jl")


using .MLM:mGLM, Estimat
export mGLM, Estimat

using .flxMLM: mlm1Scan, mlm2Scan, mlmTest
export mlm1Scan, mlm2Scan, mlmTest

using .GRM:kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr,kinshipStd,shrinkg,shrinkgLoco,kinshipLoco
export kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr,kinshipStd
export shrinkg,shrinkgLoco,kinshipLoco

using .flxMLMM: geneScan,gene1Scan,gene2Scan,envScan,permTest,K2eig, K2Eig,getKc 
#selectQTL
export geneScan,gene2Scan,envScan,permTest,K2eig, K2Eig, gene1Scan,getKc 

using .Util:setSeed, Markers, newMarkers, mat2vec,mat2array,array2mat, getGenoidx,getFinoidx,lod2logP,ordrMarkers,sortBycM,Y_huber
export setSeed, Markers, newMarkers, mat2vec,mat2array,array2mat, getGenoidx,getFinoidx,lod2logP,ordrMarkers,sortBycM,Y_huber


end






