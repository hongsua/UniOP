#!/usr/bin/env python

"""
UniOP_dst - Predicting operons using intergenic distance
Created on Wednesday May 24 2023
@author: Hong Su
"""

import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import KernelDensity

def create_parser():
	'''
	get command line arguments
	'''
	parser = argparse.ArgumentParser(description='Operon prediction using intergenic distance and conservation of adjacent gene pairs.',
									 epilog="""An example:\npython3 UniOP_dst.py -a ../demo/GCF_000005845.2.faa\n""")
	parser.add_argument('-i','--fna_file',required=False,help='fasta genome sequence.')
	parser.add_argument('-a','--faa_file',required=False,help='fasta amino acids sequence predicted by Prodigal.')
	parser.add_argument('-t','--path',required=False,help='optional folder path where output files go (if not specified, the input file path is used)')
	parser.add_argument('-n','--n_sample',required=False,help='optional, the number of samples to generate a random combination of convergent and divergent distances, default 10**4)')
	return parser

def read_faa(faa):
	'''
	put the faa file (predicted by Prodigal) into the data frame
	'''
	reader = open(faa,'r')
	lines = reader.readlines()
	lines = [l.split('#') for l in lines if l.startswith('>')]
	df = pd.DataFrame(lines,columns=['nc','start','stop','strand','attribute'])
	df['nc'] = df['nc'].replace(r'>|_\d+\s+','',regex=True)
	df = df[['nc','start','stop','strand']].astype({'nc':'str','start':'int','stop':'int','strand':'int'})
	df['strand'] = df['strand'].replace(1,'+',regex=True)
	df['strand'] = df['strand'].replace(-1,'-',regex=True)
	return df

def extract_pairs_singlecontig(df_GFF):
	'''
	extract three types of adjacent gene pairs including the same strand, the convergent, and the divergent adjacent gene pairs
	'''
	# extract different types of neighboring genes
	df_gff = df_GFF.copy()	
	df_gff['strandnext'] = df_gff.strand.shift(-1)
	df_gff['startnext'] = df_gff.start.shift(-1)
	df_gff['IGD'] = df_gff['startnext'] - df_gff['stop'] + 1
	df_gff = df_gff.dropna() ## exclude the last gene
	pairs = []
	for i in df_gff.index:
		if df_gff.at[i,"strand"] == df_gff.at[i,"strandnext"]: 
			pairs.append("SGs") ## the same strand adjacent gene pairs
		elif df_gff.at[i,"strand"] == "+": 
			pairs.append("CGs") ## the convergent adjacent gene pairs
		else:
			pairs.append("DGs") ## the divergent adjacent gene pairs
	df_gff['pairs'] = pairs
	return df_gff

def extract_pairs_multicontigs(df_GFF):
	# extract different types of neighboring genes
	df_gff = df_GFF.copy()	
	n_ctgs = df_gff['nc'].unique() # all contigs
	all_df_pairs = []
	if len(n_ctgs) == 1:
		df_i_ctg = df_gff[df_gff['nc']==n_ctgs[0]][['nc','start','stop','strand']]
		df_pairs = extract_pairs_singlecontig(df_i_ctg)
		return df_pairs
	else:
		for i in n_ctgs:
			df_i_ctg = df_gff[df_gff['nc']==i][['nc','start','stop','strand']]
			if len(df_i_ctg) <= 1: # remove the contigs with less than 2 genes
				continue
			df_pairs = extract_pairs_singlecontig(df_i_ctg)
			all_df_pairs.append(df_pairs) # contain all contigs with >= 2 genes

		df_allpairs = pd.concat(all_df_pairs, axis=0)
		return df_allpairs

def estimate_q(df_pairs):
	'''
	q is a probability that a pair of genes, located adjacent to each other on the same DNA strand, are part of an operon. 
	'''
	M = len(df_pairs) # the total number of gene-gene transitions.
	S = len(df_pairs[df_pairs['pairs']=='SGs']) # the number of same-strand gene-gene transitions.
	O = M - S # the number of opposite-strand gene-gene transitions.
	q = (M-2*O)/(M-O)
	return q

def distPred(q, df_pairs, n_sample, smooth=1):
	'''
	generate the probability of each same-strand adjacent gene pair belonging to the same operon. 
	'''
    ## sort the distances
	df_igd = df_pairs[df_pairs['pairs']=='SGs'].sort_values("IGD")
	igd_lst = df_igd.IGD.values
    ## transform all intergenic distancesN_data = len(igd_lst)
	# get the unique distance and its frequency
	N_data = len(igd_lst)
	val, counts = np.unique(igd_lst, return_counts=True)
	# store the unique distance into a dict
	dist_dict = {}
	for i in range(len(val)):
		dist_dict[val[i]] = counts[i]
	Qd_igd = []
	for v in igd_lst:
		lt_v = [e for e in val if e <= v]
		num = sum([dist_dict[e] for e in lt_v]) + 0.5
		Qd_igd.append(num/(N_data + 1))
	Qd_igd = np.array(Qd_igd).reshape(len(Qd_igd), 1)
	## sampling the distance of non-operonic pairs by convergent and divergent gene pairs
	np.random.seed(0)
	CGDs = df_pairs[df_pairs['pairs']=='CGs'].IGD.values # the distances of convergent gene pairs
	DGDs = df_pairs[df_pairs['pairs']=='DGs'].IGD.values # the distances of divergent gene pairs
	# sample with repeats
	new_CGDs = np.random.choice(CGDs, n_sample)
	new_DGDs = np.random.choice(DGDs, n_sample)
	acd = np.add(new_CGDs, new_DGDs)/2.0 # the average of convergent and divergent distances
	Qd_acd = []
	for v in acd:
		lt_v = [e for e in val if e <= v]
		num = sum([dist_dict[e] for e in lt_v]) + 0.5
		Qd_acd.append(num/(N_data + 1))
	Qd_acd = np.array(Qd_acd).reshape(len(Qd_acd), 1)
	## estimate the probability of abjacent genes belonging to the same operon using KDE
	bw0 = float("{:.2f}".format(100/N_data)) ## bandwidth
	kde1 = KernelDensity(kernel="gaussian", bandwidth=bw0*2).fit(Qd_acd)
	kde2 = KernelDensity(kernel="gaussian", bandwidth=bw0*2).fit(Qd_igd)
	p1 = (1-q)*np.exp(kde1.score_samples(Qd_igd)) ## p(d|zij=0)
	p2 = np.exp(kde2.score_samples(Qd_igd)) ## p(d)
	p = []
	for i in range(len(p1)):
		pij = 1 - p1[i]/p2[i]
		if pij <= 0:
			pij = 10**-2
		p.append(pij)
	## smooth the prediction
	p_final = []
	if smooth:
		p_min = min(p)
		min_idx = p.index(p_min)
		p_modify = [p[i] if i<=min_idx else p_min for i in range(len(p))]
		p_final = p_modify
	else:
		p_final = p
	## put the output into data frame
	prob = pd.DataFrame()
	prob['IGD'] = igd_lst
	prob['p'] = p_final
	prob['idx'] = df_igd.index.values
	prob['nc'] = df_igd.nc.values
	prob = prob.astype({'idx':'int','IGD':'int','p':'float'})
	out = prob.sort_values('idx')
	out.index = out.idx.values
	return out

def gene_prediction(inputfile, path=None):
	'''
	predict .gff and .faa files by Prodigal
	'''
	head, tail = os.path.split(inputfile)
	qname = tail.replace('.fna','')
	gff_file = f"{path}/{qname}.gff"
	faa_file = f"{path}/{qname}.faa"

	os.system(f"prodigal -i {inputfile} -f gff -o {gff_file} -a {faa_file}")
	return faa_file

def output_distPred(inputfile, n_sample, path=None):
	'''
	with a prediction file in the format Gene A\tGene B\tPrediction
	'''
	df_cds = read_faa(inputfile)
	df_pairs = extract_pairs_multicontigs(df_cds)
	q = estimate_q(df_pairs)
	preds = distPred(q, df_pairs, n_sample)
	
	head, tail = os.path.split(inputfile)
	#outfile = f"{path}/{tail.replace('.faa','.dst.pred')}"
	outfile = f"{path}/dist.pred"
	with open(outfile, 'w') as fout:
		fout.write(f"operon prediction scores\n")
		fout.write(f"Gene A\tGene B\tPrediction\n")
		for i in df_cds.index:
			if i == df_cds.index[-1]:
				continue
			row = preds[preds.index == i]
			if len(row) > 0:
				prob = "{:.6f}".format(preds.at[i,'p'])
			else:
				prob = ''
			fout.write(f"{i+1}\t{i+2}\t{prob.ljust(9)}\n")

def main():
	parser = create_parser()
	args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
	
	start = datetime.now()
	if args.faa_file:
		faa_file = args.faa_file
		if args.path:
			data_path = args.path
		else:
			head, tail = os.path.split(faa_file) 
			data_path = head	
	else:
		if args.fna_file:
			fna_file = args.fna_file
			if args.path:
				data_path = args.path
			else:
				# the input path is used as the output path is not specified
				head, tail = os.path.split(fna_file) 
				data_path = head
			faa_file = gene_prediction(fna_file, path=data_path)
		else:
			raise ValueError('Please specify the input file, either faa file or fna file')
	# predict operons
	if args.n_sample:
		n_sample = args.n_sample
	else:
		n_sample = 10**4
	output_distPred(faa_file, n_sample, path=data_path)
	print(f"Time for UniOP_dst prediction: {datetime.now() - start}")
	

if __name__ == '__main__':
	main()
