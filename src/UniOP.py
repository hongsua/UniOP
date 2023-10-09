#!/usr/bin/env python

"""
SelfTrainingModel - Predicting operons based on intergenic distance and conservation of neighboring gene pairs aross multiple genomes
Created on Wednesday May 24 2023
@author: Hong Su
"""

from UniOP_dst import *
from LR import *


def create_parser2():
	'''
	get command line arguments
	'''
	parser = argparse.ArgumentParser(description='Operon prediction using intergenic distance and conservation of adjacent gene pairs.',
									 epilog="""An example:\npython3 UniOP.py -a ../demo/GCF_000005845.2.faa -db_msh ../data/ncbi_reference.msh\n""")
	parser.add_argument('-i','--fna_file',required=False,help='fasta genome sequence.')
	parser.add_argument('-a','--faa_file',required=False,help='fasta amino acids sequence predicted by Prodigal.')
	parser.add_argument('-t','--path',required=False,help='optional folder path where output files go (if not specified, the input file path is used)')
	parser.add_argument('-n','--n_sample',required=False,help='optional, the number of samples to generate a random combination of convergent and divergent distances, default 10**4)')
	parser.add_argument('-db_msh','--ref_db_msh',required=True,help='required by conservation model (UniOP_cons): a compressed .msh file contains the entire reference genome.')
	parser.add_argument('-db_pred','--distPred_db',required=True,help='required by conservation model: operon prediction database of the selected reference set. This is prepared by UniOP_dst.')
	return parser

def run_spacedust(faa_file, ref_faa_lst, maxseq=500, e_val=10, cluw=0, maxggap=5, covm=2, path=None):
	'''
	Perform cluster searching using Spacedust, the parameters in spacedust are fixed.
	'''
	
	clusterresult_dir = f"{path}/spacedust" #put the results of spacedust in the same folder as the input files
	if not os.path.exists(clusterresult_dir):
		os.makedirs(clusterresult_dir)
	querydb = f"{clusterresult_dir}/querydb"
	if not os.path.exists(querydb):
		print(f"prepare querydb...")
		os.system(f"spacedust createsetdb {faa_file} {querydb} {clusterresult_dir}/q_tmp")
	targetdb = f"{clusterresult_dir}/targetdb"
	if not os.path.exists(targetdb):
		print(f"prepare targetdb...")
		os.system(f"spacedust createsetdb $(cat {ref_faa_lst}) {targetdb} {clusterresult_dir}/t_tmp")
	os.chdir(clusterresult_dir)
	resultdb = f"{clusterresult_dir}/resultdb"
	if not os.path.exists(resultdb):
		print(f"run clustersearch...")
		command = f"spacedust clustersearch {querydb} {targetdb} resultdb res_tmp --max-seqs {maxseq} -e {e_val} --threads 16 --cluster-use-weight {cluw} --max-gene-gap {maxggap} --suboptimal-hits 0 --cov-mode {covm}"
		os.system(command)
	os.system(f"spacedust prefixid resultdb {clusterresult_dir}/resultdb_pref --tsv 1")
	os.system(f"spacedust prefixid resultdb_h {clusterresult_dir}/resultdb_h_pref --tsv 1")

def read_lookup(lookup_file):
	mapping = pd.read_csv(lookup_file, sep="\t", header=None,
						  names=['seqid','header','setid'],
						  dtype = {'seqid': 'int', 'header': 'str', 'setid': 'int'})
	#mapping[['nc','acc','idx0','idx','start','end']] = mapping['header'].str.split("_", n = 6, expand = True)
	mapping[['acc','idx0','idx','start','end']] = mapping['header'].str.rsplit("_", n = 4, expand = True)
	mapping = mapping.astype({'idx0':'int','idx':'int','start':'int','end':'int'})
	return mapping
	
def separate_clustersearch_lookup(lookup_file):
	mapping = read_lookup(lookup_file)
	mapping.sort_values('idx') #sort per set by idx
	mapping['strand'] = np.where(mapping["start"] < mapping["end"], True, False) 
	mapping['accnext'] = mapping.acc.shift(-1) ## for multi-contig genomes, acc is the contig id in the genome.
	#seperate clustersearch results into per query genome
	n_query_genomes = mapping['setid'].nunique()
	qs_mapping = {}
	qs_cssp = {}
	for i in range(n_query_genomes):
		qs_mapping[i] = mapping[mapping['setid'] == i]
		qs_mapping[i]['strandnext'] = qs_mapping[i].strand.shift(-1)
		qs_mapping[i]['iscssp'] = (qs_mapping[i]['strand'] == qs_mapping[i]['strandnext'])
		qs_mapping[i]['isacc'] = (qs_mapping[i]['acc']==qs_mapping[i]['accnext']) # exclude the adjacent gene pairs on the same strand but the different contigs
		df_cssp = qs_mapping[i][(qs_mapping[i]['iscssp']==True)&(qs_mapping[i]['isacc']==True)] 
		qs_cssp[i] = df_cssp['idx'].reset_index(drop=True)        
	return (n_query_genomes, qs_mapping, qs_cssp)

def process_clusterresult(clusterresult_dir):
	clusterresult_file = f"{clusterresult_dir}/resultdb_pref"
	clusterheaderresult_file = f"{clusterresult_dir}/resultdb_h_pref"
	## clusterresult_file 
	clusterresult = pd.read_csv(clusterresult_file, sep='\t', header=None)[[0,1,2]]
	clusterresult = clusterresult.rename(columns={0:'clusterid',1:'qid',2:'tid'})
	clusterresult = clusterresult.astype({'clusterid':'int', 'qid':'int', 'tid':'int'})
	clusterresult = clusterresult.sort_values(['clusterid']).reset_index(drop=True)
	g = clusterresult.groupby('clusterid').qid
	clusterresult_agg = pd.concat([g.apply(list)], axis=1, keys=['genes'])
	clusterresult_agg['clusterid'] = clusterresult_agg.index
	## clusterheaderresult_file
	clusterheaderresult = pd.read_csv(clusterheaderresult_file, sep="\t", header=None, names=['clusterid','qsetid','tsetid','clustp','orderp','num'])
	clusterheaderresult = clusterheaderresult.sort_values('clusterid').reset_index(drop=True)
	## collect tid of each qid in each reference genome
	col_dict = {0:'clusterid',1:'qid',2:'tid'}
	clusterresult_dict = {}
	for i in clusterresult.index:
		cluster = clusterresult.iat[i,0]
		qid = clusterresult.iat[i,1] 
		tid = clusterresult.iat[i,2] 
		if cluster not in clusterresult_dict.keys():
			clusterresult_dict[cluster] = {}
		clusterresult_dict[cluster][qid] = tid
	return (clusterresult_agg, clusterresult_dict, clusterheaderresult)

def get_hits_and_tid(clusterresult_dir, query_lookup, target_lookup):
	'''
	Generate 0-1 matrix and corresponding target id in reference genomes
	'''
	## query lookup
	n_query_genomes, qs_mapping, qs_cssp = separate_clustersearch_lookup(query_lookup)
	## target lookup
	df_target_lookup = read_lookup(target_lookup)
	## cluster results
	clusterresult_agg, clusterresult_dict, clusterheaderresult = process_clusterresult(clusterresult_dir)
	##
	qs_clusterheaderresult = {}
	## matrix of all abjacent gene pairs
	qs_matrix = {} # conservation matrix for all query genomes
	tid_1_matrix = {} # tid of qid i
	tid_2_matrix = {} # tid of qid i+1
	## df of all adjacent gene pairs in cssps
	qs_mat_df = {}
	tid_1_mat_df = {}
	tid_2_mat_df = {}
	for k in range(n_query_genomes):
		result_df = clusterheaderresult[clusterheaderresult['qsetid'] == k]
		mapping = qs_mapping[k]
		pairs_query = len(mapping['seqid']) - 1
		n_reference = result_df['tsetid'].max() + 1
		dimensions = (pairs_query, n_reference)
		qs_matrix[k] = np.zeros(dimensions)
		tid_1_matrix[k] = np.zeros(dimensions)
		tid_2_matrix[k] = np.zeros(dimensions)
		for cluster in clusterresult_agg['clusterid']:
			j = result_df['tsetid'][cluster] #extract tsetid in cluster
			for i in clusterresult_agg['genes'][cluster]:
				if (i+1) in clusterresult_agg['genes'][cluster]:
					qs_matrix[k][i][j] = 1
					## tid_1 mapped to i
					tseqid_1 = clusterresult_dict[cluster][i]
					tid_1 = df_target_lookup.at[tseqid_1,'idx'] # sorted in all contigs
					tid_1_matrix[k][i][j] = tid_1
					#tsetid = df_target_lookup.at[tseqid_1,'setid'] # double check
					## tid_2 mapped to i+1
					tseqid_2 = clusterresult_dict[cluster][i+1]
					tid_2 = df_target_lookup.at[tseqid_2,'idx']
					tid_2_matrix[k][i][j] = tid_2
		## old conservation matrix
		idx_query = mapping['idx'][:-1].values.flatten().reshape(1, pairs_query)  
		qs_mat = np.append(idx_query, qs_matrix[k].T, axis=0).astype(int)
		qs_mat_df[k] = pd.DataFrame(data=qs_mat[1:,:],columns=qs_mat[0,:])
		qs_mat_df[k] = qs_mat_df[k][qs_mat_df[k].columns.intersection(qs_cssp[k])]
		## tid_1 in each tset 
		tid_1_mat = np.append(idx_query, tid_1_matrix[k].T, axis=0).astype(int)    
		tid_1_mat_df[k] = pd.DataFrame(data=tid_1_mat[1:,:],columns=tid_1_mat[0,:])
		tid_1_mat_df[k] = tid_1_mat_df[k][tid_1_mat_df[k].columns.intersection(qs_cssp[k])]
		## tid_2 in each tset 
		tid_2_mat = np.append(idx_query, tid_2_matrix[k].T, axis=0).astype(int)     
		tid_2_mat_df[k] = pd.DataFrame(data=tid_2_mat[1:,:],columns=tid_2_mat[0,:])
		tid_2_mat_df[k] = tid_2_mat_df[k][tid_2_mat_df[k].columns.intersection(qs_cssp[k])]  
	return (qs_mat_df, tid_1_mat_df, tid_2_mat_df)

def assign_pseudolabel(prob, q, alpha):
	#number of pairs to assign a pseudolabel of z_gij = 1 and 0
	a1 = np.floor(q * alpha * len(prob)).astype(int)
	a2 = np.floor((1 - q) * alpha * len(prob)).astype(int)
	#get list of gene pairs of z_gij = 1 and 0
	idx_1 = prob.sort_values(by='p',ascending=False)[:a1]['idx'].values
	idx_0 = prob.sort_values(by='p')[:a2]['idx'].values
	return (idx_1, idx_0)

def add_pseudolables(qs_mat_df_centralized, idx_1, idx_0):
	#add pseudolabels
	qs_mat_zgij = {}
	n_query_genomes = len(qs_mat_df_centralized)
	for k in range(n_query_genomes):
		mat_df_centralized = qs_mat_df_centralized[k]
		qs_mat_zgij_0 = mat_df_centralized[mat_df_centralized.columns.intersection(idx_0)].T
		qs_mat_zgij_0['label'] = 0
		qs_mat_zgij_1 = mat_df_centralized[mat_df_centralized.columns.intersection(idx_1)].T
		qs_mat_zgij_1['label'] = 1
		qs_mat_zgij[k] = pd.concat([qs_mat_zgij_0, qs_mat_zgij_1]).to_numpy()
	return qs_mat_zgij
	
def single_LRmodel(q, dist_prob, qs_mat_df_centralized, alpha=0.5):
	idx_1, idx_0 = assign_pseudolabel(dist_prob, q, alpha)
	qs_mat_zgij = add_pseudolables(qs_mat_df_centralized, idx_1, idx_0)
	###input data
	mat_zgij = qs_mat_zgij[0]
	X = mat_zgij[:,:-1] #feature vector
	y = mat_zgij[:,-1] #pseudolabels
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)
	### best parameters
	parameters = {'learningRate'  : 0.01,
				  'numIterations' : 100,
				  'penalty'       : 'L2',
				  'lamda'         : 10}
	### set up model
	model = LogisticRegression().set_params(**parameters)
	### train model
	model.fit(X, y)   
	return model

def best_hyperparams(q, dist_prob, qs_mat_df_centralized, parameters, alpha=0.5):
	idx_1, idx_0 = assign_pseudolabel(dist_prob, q, alpha)
	qs_mat_zgij = add_pseudolables(qs_mat_df_centralized, idx_1, idx_0)
	###input data
	mat_zgij = qs_mat_zgij[0]
	X = mat_zgij[:,:-1] #feature vector
	y = mat_zgij[:,-1] #pseudolabels
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)
	grid_result = tune_hyperparameters(X_train, y_train, parameters, CV=5)
	best_params = grid_result.best_params_
	model = LogisticRegression().set_params(**best_params)
	model.fit(X, y)
	model.plotCost()
	return model
	
def collect_cons_result(mat, q, dst_pred):
	LR = single_LRmodel(q, dst_pred, mat)
	## conservation results
	X = mat[0].T.to_numpy()
	cons = pd.DataFrame()
	_, prob = LR.predict(X)
	cons['p'] = prob
	cons.index = mat[0].T.index
	return (cons, LR)

def sigmoid_func(x):
	return 1/(1+np.exp(-x))

def combine_dist_conserv_model(dist_prob, LR_model, cons_mat):
	#weights from the logistic regression without distance information
	w = LR_model.weights_.numpy()
	#predictions of conservation model
	X_test = cons_mat[0].T.to_numpy()
	z = w[0] + np.dot(X_test, w[1:])
	#distance prediction
	d_pred = list(dist_prob['p'])
	#combine distance prob into LR model
	z_new = np.array([z[i]+np.log((d_pred[i])/(1-d_pred[i])) for i in range(len(z))])
	#logistic regression prediction
	LR_probs = np.array([sigmoid_func(i) for i in z_new])
	prob = pd.DataFrame()
	prob['p'] = LR_probs
	prob.index = cons_mat[0].T.index
	return prob
def scale_conservation_matrix(qs_mat_df, lookup_file):
	qs_mat_df_centralized = {}
	keys = qs_mat_df.keys()
	n_query_genomes, qs_mapping, qs_cssp = separate_clustersearch_lookup(lookup_file)
	for k in keys:
		qs_mat_df_centralized[k] = qs_mat_df[k].subtract(qs_mat_df[k].sum(axis=1)/len(qs_cssp[k]), axis=0)
	return qs_mat_df_centralized
	
def conservation_mat(distPred_db, path=None):
	'''
	Generate conservation matrix of adjacent gene pairs based on the results of spacedust and the operon prediction of distance model
	'''
	## input files 
	clusterresult_dir = f"{path}/spacedust"
	query_lookup = f"{clusterresult_dir}/querydb.lookup"
	target_lookup = f"{clusterresult_dir}/targetdb.lookup"
	targetdb_source_file = f"{clusterresult_dir}/targetdb.source"
	##
	qs_mat_df, tid_1_mat_df, tid_2_mat_df = get_hits_and_tid(clusterresult_dir, query_lookup, target_lookup)
	## 
	df_source = pd.read_csv(targetdb_source_file, header=None, sep='\t', names=['ref_idx', 'gcf_name'])
	df_source2 = df_source.replace(".faa", "", regex=True)
	## matrix
	mat_df = {}
	tid_1 = {}
	tid_2 = {}
	keys = qs_mat_df.keys()
	for k in keys:
		mat_df[k] = qs_mat_df[k].copy()
		for i in qs_mat_df[k].index:
			i_gcfname = df_source2.at[i, 'gcf_name']
			ref_dstPre= pd.read_csv(f"{distPred_db}/{i_gcfname}_operon_scores", index_col=0)
			## loop all gene pairs in all CSSPs of query
			for j in qs_mat_df[k].columns:
				if mat_df[k].at[i,j] == 0:
					continue
				else:
					tid_1 = tid_1_mat_df[k].at[i, j]
					tid_2 = tid_2_mat_df[k].at[i, j]
					prob_ij = 0
					if abs(tid_1-tid_2) == 1: ## target genes are adjacent
						if len(ref_dstPre[ref_dstPre['idx']==min(tid_1, tid_2)]) > 0:
							prob_ij = ref_dstPre[ref_dstPre['idx']==min(tid_1, tid_2)].p.values[0]
					mat_df[k].at[i, j] = prob_ij
	nodst_mat_df = scale_conservation_matrix(qs_mat_df, query_lookup)
	return (mat_df, nodst_mat_df)

def cons_dist_Pred(infile, distPred_db, n_sample, d_path=None):
	'''
	Generate the final prediction using both integenic distance and conservation sources
	'''
	## self-training
	# prediction of distance model
	start = datetime.now()
	df_gff = read_faa(infile)
	df_pairs = extract_pairs_multicontigs(df_gff)
	q = estimate_q(df_pairs)
	dist_pred = distPred(q, df_pairs, n_sample)
	print(f"Time for UniOP_dst prediction: {datetime.now() -  start}")
	# generting conservation matrix
	start = datetime.now()
	mat_df, nodst_mat_df = conservation_mat(distPred_db, path=d_path)
	print(f"Time for generating conservation matrix: {datetime.now() -  start}")
	# conservation model
	start = datetime.now()
	cons, LR_model = collect_cons_result(mat_df, q, dist_pred)
	#cons_nodst, LR_model_nodst = collect_cons_result(nodst_mat_df, q, dist_pred)
	
	# combinde model
	comb = combine_dist_conserv_model(dist_pred, LR_model, mat_df)
	print(f"Time for generating UniOP prediction: {datetime.now() -  start}")
	#comb_nodst = combine_dist_conserv_model(dist_pred, LR_model_nodst, nodst_mat_df)
	## out
	all_pred = pd.DataFrame()
	all_pred['pr_dist'] = dist_pred.p.values
	all_pred['pr_cons'] = cons.p.values
	#all_pred['pr_cons'] = cons_nodst.p.values
	all_pred['pr_comb'] = comb.p.values
	#all_pred['pr_comb_nodst'] = comb_nodst.p.values
	all_pred.index = dist_pred.idx.values
	return all_pred
	

def save_final_prediction(infile, distPred_db, n_sample, path=None):
	'''
	with a prediction file in the format Gene A\tGene B\tPrediction
	'''
	
	all_pred = cons_dist_Pred(infile, distPred_db, n_sample, d_path=path)
	head, _ = os.path.split(infile)
	outfile = f"{head}/uniop.pred"
	df_gff = read_faa(infile)
	with open(outfile, 'w') as fout:
		fout.write(f"operon prediction scores\n")
		fout.write(f"Gene A\tGene B\tPr(O|ID)\tPr(O|CL)\tPr(O|ID, CL)\n")
		for i in df_gff.index:
			if i == df_gff.index[-1]:
				continue
			row = all_pred[all_pred.index == i]
			if len(row) > 0:
				dist_prob = "{:.6f}".format(all_pred.at[i,'pr_dist'])
				cons_prob = "{:.6f}".format(all_pred.at[i,'pr_cons'])
				comb_prob = "{:.6f}".format(all_pred.at[i,'pr_comb'])
				#comb_nodst_prob = "{:.6f}".format(all_pred.at[i,'pr_comb_nodst'])
			else:
				dist_prob = ''
				cons_prob = ''
				comb_prob = ''
				#comb_nodst_prob = ''
			fout.write(f"{i+1}\t{i+2}\t{dist_prob.ljust(9)}\t{cons_prob.ljust(9)}\t{comb_prob.ljust(9)}\n")
			
def main():
	parser = create_parser2()
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

	# checking if the following required files exist
	if args.ref_db_msh:
		ref_db_msh = args.ref_db_msh
	else:
		raise ValueError('Please prepare the ref_db_msh file first by refs_selection.py')
	
	##### whole uniop processing and operon prediction #######
	result_file = f"{data_path}/spacedust/resultdb_pref"
	if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
		## selecting reference genomes
		ref_faa_lst = f"{data_path}/sele_refs.lst"
		if not os.path.exists(ref_faa_lst):
			raise ValueError('sele_refs.lst is missed')
		with open(ref_faa_lst) as fin:
			sele_refs = fin.readlines()
		N_maxseq = 2*len(sele_refs)
		run_spacedust(faa_file, ref_faa_lst, maxseq=N_maxseq, e_val=10, cluw=0, maxggap=5, covm=2, path=data_path)
	else:
		print(f"spacedust is already done!")
			
	## predict operons
	if args.distPred_db:
		pred_db = args.distPred_db
	else:
		raise ValueError(f"Please specify the pred_db path")
		
	if args.n_sample:
		n_sample = args.n_sample
	else:
		n_sample = 10**4
	save_final_prediction(faa_file, pred_db, n_sample, path=data_path)
	print(f"Time for processing: {datetime.now() - start}")
	


if __name__ == '__main__':
	main()
