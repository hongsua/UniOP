#!/usr/bin/env python

"""
refs_selection - selecting valid reference genomes from a relative big database.
Created on Tuesday August 01 2023
@author: Hong Su
"""

from UniOP_dst import *

def get_parser():
	'''
	get command line arguments
	'''
	parser = argparse.ArgumentParser(description='Selecting valid reference genomes from a relative big database.')
	parser.add_argument('-a','--faa_file',required=True,help='fasta amino acids sequence predicted by Prodigal.')
	parser.add_argument('-db_dir','--ref_db',required=True,help='required folder path where all reference genomes are located.')
	parser.add_argument('-db_lst','--ref_db_lst',required=True,help='required list where all reference genomes contain absolute paths.')
	parser.add_argument('-db_msh','--ref_db_msh',required=False,help='optional, a compressed .msh file contains the entire reference genome. If no such file is provided, this code will generate it')
	parser.add_argument('-db_pred','--distPred_db',required=True,help='optional folder path where output files go (if not specified, the input file path is used)')
	return parser

def select_refs(ref_db_msh, faa_file, sele_n=1):
	'''
	Select sub reference genomes for each query based on evolutionary distance. 
	The entire reference genome is compressed into a single reference.msh file.
	'''
	## compute the evolutionary distance between query and all reference genomes using mash.
	path, tail = os.path.split(faa_file)

	dist_file = f"{path}/mash_dist.txt"
	os.system(f"mash dist {ref_db_msh} {faa_file} -p 8 | cut -f 1,3,4,5 > {dist_file}")
	## select <= n genomes at each distance.
	df_mash = pd.read_csv(dist_file, sep='\t', header=None, 
						  names=['refid','dist','pval','match'], 
						  dtype={'refid':'str','dist':'float'})
	sele_dict = {'index':[],'dist':[]}
	for i in df_mash.index:
		dst = df_mash.at[i,'dist']
		if dst == 1.0 or dst == 0.0: ## skip the genomes with evolutionary distance == 1.0/0.0
			continue
		n_dst = sele_dict['dist'].count(dst)
		if n_dst <= sele_n: ## the num of selected refs at a given distance equals or less than 2.
			sele_dict['dist'].append(dst)
			sele_dict['index'].append(i)
	sele_df = df_mash[df_mash.index.isin(sele_dict['index'])]
	subrefs = f"{path}/sele_refs.lst"
	with open(subrefs, 'w') as fout:
		fout.write('\n'.join(sele_df.refid.values))
	return subrefs

def save_distPred(inputfile, outfile):
	df_cds = read_faa(inputfile)
	df_pairs = extract_pairs_multicontigs(df_cds)
	q = estimate_q(df_pairs)
	preds = distPred(q, df_pairs, 10**4)
	preds.to_csv(outfile)
		
def main():
	parser = get_parser()
	args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
	
	start = datetime.now()
	
	if args.faa_file:
		faa_file = args.faa_file	
	else:
		raise ValueError('Please specify the input file, it must be a faa file')

	if args.ref_db_lst:
		db_lst = args.ref_db_lst
	else:
		raise ValueError('Please specify the reference genomes list used by the UniOP_cons.')
		
	with open(db_lst) as f:
		lines = f.readlines()
		lines = [l.strip() for l in lines]
	
	lst_nofaa = []
	for g in lines:
		if os.path.exists(g):
			continue
		else:
			lst_nofaa.append(g)
			
	## run prodigal to predict .faa files for the genomes without .faa files
	### 
	if len(lst_nofaa) > 1:
		print(f"# of genomes without faa files: {len(lst_nofaa)}")
		if args.ref_db:
			db_dir = args.ref_db
			for g in lst_nofaa:
				_, tail = os.path.split(g)
				fna_file = f"{db_dir}/{tail.replace('.faa','.fna')}"
				if not os.path.exists(fna_file):
					raise ValueError(f"{tail} has no .fna file")
				else:
					gff_file = f"{db_dir}/{tail.replace('.faa','.gff')}"
					os.system(f"prodigal -i {fna_file} -f gff -o {gff_file} -a {g}")
		else:
			raise ValueError('Please specify the reference genomes path where the nucleotide genome sequence is located')
	else:
		print(f"all reference genomes have faa files")
	##### selection process #####
	## generate ref.msh file using mash
	if args.ref_db_msh:
		ref_db_msh = args.ref_db_msh
	else:
		os.system(f"mash sketch -o ../data/ncbi_reference $(cat {db_lst}) -p 8 -a")
		ref_db_msh = f"../data/ncbi_reference.msh"
	## selecting
	sele_refs_file = select_refs(ref_db_msh, faa_file, sele_n=1)
	
	## run UniOP_dst to generate operon prediction for the selected genomes
	if args.distPred_db:
		pred_db = args.distPred_db
	else:
		pred_db = os.getcwd()
	
	with open(sele_refs_file) as fin:
		sele_refs = fin.readlines()
		sele_refs = [l.strip() for l in sele_refs]
	
	n_noPred = 0
	for ref_g in sele_refs:
		_, gname = os.path.split(ref_g)
		gname = gname.replace(".faa","")
		outfile = f"{pred_db}/{gname}_operon_scores"
		if not os.path.exists(outfile):
			n_noPred += 1
			save_distPred(ref_g, outfile)

	
	print(f"# of no dist prediction: {n_noPred}")
	print(f"Time for preparing the prediction of selected reference genomes: {datetime.now() - start}")
	


if __name__ == '__main__':
	main()