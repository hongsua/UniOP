# UniOP: self-supervised universal operon prediction for prokaryotic (meta-)genomics data
The manuscript is in preparation.

## Overview
We propose a fast and accurate computational method for systematic operon prediction, which does not depend on experimental information such as microarray and RNA-seq data, or functional information from databases like COG, KEGG, and STRING. UniOP takes as input one or more prokaryotic genomes or metagenomic assembled genomes (MAGs) and annotates the genes using Prodigal v2.6.3. Our approach leverages the features extracted solely from genomic sequences, including (i) transcription co-directionality, where genes transcribed in opposite directions are part of different operons; (ii) intergenic distances, as genes within the same operon tend to have smaller intergenic distances compared to genes not belonging to the same operon; (iii) conserved gene neighborhood, where genes within an operon exhibit conserved gene order and strand orientation across genomes due to evolutionary relationships or horizontal gene transfer. 


## Installation
