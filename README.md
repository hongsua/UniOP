# UniOP: Universal operon prediction for high-throughput prokaryotic (meta-)genomics data
```
UniOP was implemented in Python 3.11.5 and runs on MacOS or Linux systems.
```

## Overview
Operon prediction remains challenging for poorly characterized prokaryotic genomes, particularly for metagenome-assembled genomes (MAGs) that lack functional annotations and evolutionary relatives. Existing methods typically rely on supervised training, comparative genomics, or external databases, limiting their applicability to such data.

**UniOP** address this gap by inferring operon structure directly from intergenic distance distributions within the target genome. It requires no external information, no training data, and no species-specific assumptions. 

UniOP takes as input either a nucleotide genome sequence (FASTA), a protein sequence file (FAA), or a gene annotation file (GFF), and outputs pairwise operon probabilities and assembled operon units. The tool is implemented in Python and runs on macOS or Linux systems.

![](figures/Graphical_abstract.png)

## How to use UniOP
### Dependencies
**UniOP** requires the following:
```
* Python (version 3.11.5 recommended; the results in the paper were obtained with this version)
* Python libraries: argparse, pandas, numpy, scikit-learn, datetime
* Prodigal (required only when using nucleotide FASTA as input)

You can install Prodigal via:

``` bash
conda install -c bioconda prodigal
```
The starting point should be either a FASTA file of the nucleotide genome sequence (`.fna`) or protein-coding sequences (**CDS**) (`.faa`). This is typically achieved by running a gene prediction program such as [Prodigal](https://github.com/hyattpd/Prodigal).

### Quickstart
```
git clone https://github.com/hongsua/UniOP.git
cd UniOP/src
python UniOP -a ../demo/GCF_000005845.2.faa
```
Note: Install **Prodigal** into the working directory, ~/UniOP/src, is necessary.

This will output files **uniop.pred** and **uniop.operon** into the same path (../demo/) as the input file by default. You can also specify a folder with the command:
```
python UniOP -i ../demo/GCF_000005845.2.fna -t your_folder
```
If the input file is the nucleotide genomic sequence, you will get the following files: **GCF_000005845.2.faa**, **GCF_000005845.2.gff** as well.

```
python UniOP -i ../demo/GCF_000005845.2.fna
```
You can type:
```
python UniOP --help
```
to find all parameters in UniOP.


## Support
If you have questions or found any bug in the program, please write to us at
hong.su[at]nankai.edu.cn
