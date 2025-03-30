# FlxQTL.jl

A comprehensive package of *Fl*e*x*ible multivariate linear (mixed) model (MLMM/MLM) for *QTL (Quantitative Trait Loci)* analysis of structured multivariate traits.  Note that MLM-based scan functions can be used for a low heritability case for efficient computation.      

## Package Features

- Genome scans (1D, 2D) and permutation tests for univariate, multivariate trait(s), and genotype (probability) data 
- LOCO (Leave One Chromosome Out) support for MLMM-based genome scans 
- Computation for Genetic Relatedness matrix (GRM or kinship) 
- CPU parallelization

## Guide

```@contents
Pages = ["guide/tutorial.md",
        "guide/analysis.md",   
        ]
Depth = 1

```

## Manual

The descriptions of functions and types arranged by module.

```@contents
Pages = ["functions.md"]

```

## Index


```@index
Pages = ["functions.md"]
```