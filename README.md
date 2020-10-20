# Assumptions about frequency-dependent architectures of complex traits bias measures of functional enrichment

**Authors**: Shadi Zabad, Aaron P. Ragsdale, Rosie Sun, Yue Li, Simon Gravel

This repository contains code to compute LD Scores using the D^2 
statistic and carry out the analyses discussed in the manuscript.

**Abstract**:
Linkage-Disequilibrium Score Regression (LDSC) is a popular framework for analyzing GWAS summary statistics that allows
for estimating SNP heritability, confounding, and functional 
enrichment of genetic variants with different annotations. 
Recent work has highlighted the influence of implicit and 
explicit assumptions of the model on the biological interpretation 
of the results. In this work, we explored a formulation of 
LDSC that replaces the $r^2$ measure of LD with a 
recently-proposed unbiased estimator of the $D^2$ statistic. 
In addition to modest statistical difference across estimators, 
this derivation highlighted implicit and unrealistic assumptions 
about the relationship between allele frequency, 
effect size, and annotation status. We carry out a systematic 
comparison of alternative LDSC formulations by applying them 
to summary statistics from 47 GWAS traits. Our results show that 
commonly used models likely underestimate functional enrichment. 
These results highlight the importance of calibrating the 
LDSC model to achieve a more robust understanding of polygenic traits.


