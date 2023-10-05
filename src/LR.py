from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score, precision_recall_curve
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class LogisticRegression():
	def __init__(self, optimizer='adam', learningRate=0.001, numIterations=100, lamda=100, lamda2=10, penalty = 'L2'):
		self.optimizer = optimizer
		self.learningRate = learningRate
		self.numIterations = numIterations
		self.penalty = penalty
		self.lamda = lamda
		self.lamda2 = lamda2
	
	def fit(self, x, y):
		x = tf.concat([tf.ones_like(y.reshape(-1,1), dtype=tf.float32), x], axis=1)
		m, n = x.shape
		self.weights_ = tf.Variable(tf.random.normal(stddev=1.0/n, shape=(n,)))
		self.costs_ = []
		
		if self.optimizer == 'adam':
			opt = tf.keras.optimizers.legacy.Adam(self.learningRate)
		elif self.optimizer == 'sgd':
			opt = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
		elif self.optimizer == 'ftrl':
			opt = tf.keras.optimizers.legacy.Ftrl(learning_rate=learning_rate)
		else:
			opt = tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate)
		
		for i in range(self.numIterations):
			if self.penalty == 'L2':
				grads = L2_grad(x, y, self.weights_, self.lamda)
			elif self.penalty == 'L2d':
				grads = L2d_grad(x, y, self.weights_, ped, self.lamda, self.lamda2)
			
			self.iterationsPerformed_ = i
			
			##update weights
			opt.apply_gradients(zip([grads], [self.weights_]))
			##make the weights non-negative
			self.weights_ = tf.Variable(tf.maximum(self.weights_, tf.zeros((n,), dtype=tf.float32)))
			##costs
			if self.penalty == 'L2':
				self.costs_.append(reg_logLiklihood(x, y, self.weights_, self.lamda))
			elif self.penalty == 'L2d':
				self.costs_.append(reg_logLiklihood_ped(x, y, self.weights_, ped, self.lamda, self.lamda2))
			
		return self
	
	def predict(self, x_test, pi=0.5):
		w = self.weights_.numpy()
		z = w[0] + np.dot(x_test, w[1:])
		probs = np.array([logistic_func(i) for i in z])
		predictions = np.where(probs >= pi, 1, 0)
		return predictions, probs 
	
	def get_params(self, deep=False):
		params = {'optimizer': self.optimizer,
				  'learningRate': self.learningRate,
				  'numIterations': self.numIterations,
				  'lamda': self.lamda,
				  'lamda2': self.lamda2,
				  'penalty': self.penalty
				 }
		return params
	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self
	def score(self, x_test, y_test):
		predictions, probs = self.predict(x_test) 
		auc = roc_auc_score(y_test, probs)
		mcc = matthews_corrcoef(y_test, predictions)
		pre = precision_score(y_test, predictions)
		recall = recall_score(y_test, predictions)
		pre_lst,rec_lst,_ = precision_recall_curve(y_test, probs)
		pre_rec_auc = metrics.auc(rec_lst, pre_lst)
		performance = {'AUC': auc, 'MCC': mcc, 'precision': pre, 'recall': recall, 'Pre_Rec_AUC':pre_rec_auc}
		return pre_rec_auc 
	def plotCost(self):
		plt.figure()
		plt.plot(np.arange(1, self.iterationsPerformed_ + 2), self.costs_, marker = '.')
		plt.xlabel('Iterations')
		plt.ylabel('Log-Liklihood J(w)')
	
def logistic_func(x):
	return tf.math.sigmoid(x)
	
def pre_ped(ped):
	n = ped.shape[0]
	out_mat = ped.copy()
	for i in range(1, n-1):
		for j in range(i+1, n):
			out_mat[i][j] = max(0, (ped[0][i] + ped[0][j]) / ped[i][j])
	return out_mat
	
def reg_logLiklihood(x, y, weights, lamda):
	epsilon = 1e-5
	m = y.shape[0]
	z = tf.tensordot(x, weights, axes=1) 
	reg_term = (lamda / 2) * tf.tensordot(tf.transpose(weights), weights, axes=1)
	loss = -1 / m * np.sum((y * np.log(logistic_func(z + epsilon))) + ((1 - y) * np.log(1 - logistic_func(z) + epsilon))) + (1 / m) * reg_term
	return loss

def reg_logLiklihood_ped(x, y, weights, new_ped, lamda, lamda2):
	epsilon = 1e-5
	m = y.shape[0]
	z = tf.tensordot(x, weights, axes=1) 
	reg_term1 = lamda / 2 * tf.tensordot(tf.transpose(weights), weights, axes=1)
	reg_term2 = lamda2 / 2 * (np.multiply(new_ped, (np.subtract.outer(weights, weights))**2)[1:, 1:].sum())
	reg_term = reg_term1 + reg_term2
	loss =  -1 / m * np.sum((y * np.log(logistic_func(z + epsilon))) + ((1 - y) * np.log(1 - logistic_func(z) + epsilon))) +  (1/m) * reg_term 
	return loss
	
def L2_grad(x, y, weights, lamda):
	m = y.shape[0]
	z = tf.tensordot(x, weights, axes=1)
	y_pred = logistic_func(z)
	errors = y_pred - y
	grads = 1 / m * (tf.tensordot(tf.transpose(errors), x, axes=1) + lamda * weights)
	return grads

def L2d_grad(x, y, weights, new_ped, lamda, lamda2):
	m = y.shape[0]
	z = tf.tensordot(x, weights, axes=1)
	y_pred = logistic_func(z)
	errors = y_pred - y
	term1 = weights
	term2 = np.append(0, np.multiply(new_ped, abs(np.subtract.outer(weights, weights)))[1:, 1:].sum(axis=1))
	grads = 1/m * (tf.tensordot(tf.transpose(errors), x, axes=1) + lamda * term1 + lamda2 * term2)
	return grads
	
def tune_hyperparameters(x, y, parameters, CV=5):
	model = LogisticRegression()
	grid_result = GridSearchCV(model, parameters, cv=CV, return_train_score=True).fit(x, y)
	return grid_result

def print_train_result(result):
	for name in result.keys():
		print('Parameters:\t{}'.format(result.best_params_))
		print('Score:\t\t{:.2%}'.format(result.best_score_))	
