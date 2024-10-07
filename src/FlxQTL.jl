"""

    FlxQTL

flexible Multivariate Linear Mixed Model based QTL analysis tools for structured multiple traits.

"""
module FlxQTL

# # Write your package code here.
 __precompile__(true)


include("MLM.jl")
include("Miscellanea.jl")
include("GRM.jl")
include("EcmNestrv.jl")
include("flxMLMM.jl")

using .MLM:mGLM, Estimat
export mGLM, Estimat

using .GRM:kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr,kinshipStd,shrinkg,shrinkgLoco,kinshipLoco
export kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr,kinshipStd
export shrinkg,shrinkgLoco,kinshipLoco

using .flxMLMM: geneScan,gene1Scan,gene2Scan,envScan,permTest,K2eig, K2Eig,getKc 
#selectQTL
export geneScan,gene2Scan,envScan,permTest,K2eig, K2Eig, gene1Scan,getKc 

using .Util:setSeed, Markers, newMarkers, mat2vec,mat2array,array2mat, getGenoidx,getFinoidx,lod2logP,ordrMarkers,sortBycM,Y_huber
export setSeed, Markers, newMarkers, mat2vec,mat2array,array2mat, getGenoidx,getFinoidx,lod2logP,ordrMarkers,sortBycM,Y_huber


end






# module flxQTL

# using Random

# greet()= print("Hello ", Random.randstring(8))

# end
