"""

    flxQTL  

flexible Multivariate Linear Mixed Model based QTL analysis tools for structured multiple traits. 

"""
module flxQTL

# # Write your package code here.
 __precompile__(true)


include("MLM.jl")
include("QTLplot.jl")
include("Miscellanea.jl")
include("GRM.jl")
include("EcmNestrv.jl")
include("flxMLMM.jl")

using .MLM:mGLM, Estimat
export mGLM, Estimat

using .GRM
export kinshipMan,kinship4way,kinshipGs,kinshipLin,kinshipCtr
export shrinkg,shrinkgLoco,kinshipLoco

using .QTLplot
export layers, plot1d, plot2d, subplot2d

using .flxMLMM: geneScan,gene2Scan,permTest,K2eig, K2Eig
#selectQTL
export geneScan,gene2Scan,permTest,K2eig, K2Eig

using .Util
export setSeed, Markers, newMarkers, mat2vec,mat2array,array2mat, getGenoidx,getFinoidx,lod2logP,ordrMarkers,sortBycM,Y_huber



end






# module flxQTL

# using Random

# greet()= print("Hello ", Random.randstring(8))

# end