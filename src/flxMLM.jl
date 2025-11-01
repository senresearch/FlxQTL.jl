"""

    flxMLM


A module designed for fitting a Multivariate Linear Model by the (Residual) Maximum Likelihood (REML or MLE) method.  
The model:

``Y=XBZ'+E``, 

where ``E(vec(Y))= (Z \\otimes X)vec(B)``,  ``Var(vec(Y))=  \\Sigma \\otimes I_n``,

 dim(Y)= (n individuals, m traits), dim(X) = (n,p markers), and dim(Z) = (m, q trait covariates).

"""
module flxMLM

using LinearAlgebra, Distributed, Random
import Statistics: quantile

using ..MLM:mGLM, Estimat
using ..Util:lod2logP, Markers

include("geneScan.jl")
    


end