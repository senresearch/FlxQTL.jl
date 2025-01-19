# About Data

 
## Fitness for Arabidopsis thaliana data: RIL (Recombinant Inbred Line)

The data is sourced from the paper, Agren, J *et al*. [Genetic mapping of adaptation reveals fitness tradeoffs in Arabidopsis thaliana](https://www.pnas.org/content/110/52/21077). *PNAS* (2013).  It is processed in [R/qtl](https://rqtl.org) and generated three files for Julia 
implementation.

**Phenotype data:** 
Phenotypes of a trait is the mean number of fruits per seedling planted in Sweden and Italy from July, 2009 to June, 2012.  The data is 
imputed for missing values and named as `Arabidopsis_fitness.csv`  in this folder (6 quantitative traits x 400 individuals). 

**Genotype data:**
The genotypes have 699 markers across 5 Chromosomes after adding pseudo markers, labelled `a(=1)`, `b(=2)` as Italian parent, Swedish parent, 
respectively, and named as `Arabidopsis_genotypes.csv` for 1D genome scan, and an additional genotype data of 398 markers for 2D genome scan is `Arabidopsis_genotypes_2d.csv`.

**Genetic marker information data:**
This auxilliary data sets are genetic marker information consisting of marker names, chromosome, and marker position (cM) in this order and are named as 
`Arabidopsis_markerinfo_1d.csv` and `Arabidopsis_markerinfo_2d.csv` for 1D and 2D scans, respectively.



