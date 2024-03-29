# UniOP: unsupervised single-sequence operon prediction for all prokaryotic (meta-)genomics data
The manuscript is currently being prepared.
```
UniOP is implemented in Python3 and runs on MacOS and Linux systems.
```

## Overview
**UniOP** is a fast and accurate computational method for operon prediction, independent of experimental or functional information. It takes as input prokaryotic genomes or metagenomic assembled genomes (MAGs).
![](figures/flowchart_UniOP.png)

## Installation
### Dependencies
**UniOP** requires:
```
* Prodigal(https://github.com/hyattpd/Prodigal/wiki/installation)
```
### Install UniOP
```
wget https://github.com/hongsua/UniOP/archive/refs/heads/main.zip
unzip main.zip
```
Install **Prodigal** into the working directory, ~/UniOP-main/src.

## Usage
## Input
The starting point should be either a FASTA file of the nucleotide genome sequence (`.fna`) or protein-coding sequences (**CDS**) (`.faa`). This is typically achieved by running a gene prediction program such as [Prodigal](https://github.com/hyattpd/Prodigal).

```
## demo: GCF_000005845.2.fna
cd UniOP-main/src
```
### Operon prediction based on intergenic distance
```
python3 UniOP.py -i ../demo/GCF_000005845.2.fna
```
This will output gene predictions into the same path as the input file, resulting in the following output files: **GCF_000005845.2.faa**, **GCF_000005845.2.gff**, and the operon prediction file named **uniop.pred**.


## Support
If you have questions or found any bug in the program, please write to us at
hong.su[at]mpinat.mpg.de
