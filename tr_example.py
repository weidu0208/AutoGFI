import numpy as np
from lib.TensorRegression import TensorRegression, TensorRegression_CV
from lib.Fiducial_Sampler import Fiducial_Sample_Generator, Anal_Samples
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# simulated data
pattern = 'R3-ex'
file_path = './data/'+pattern+'.txt'
B = np.loadtxt(file_path)
p1 = B.shape[0]
p2 = B.shape[1]
n = 400
sd_true = np.sqrt(0.5)         

print('Simulated data generation is in progress...')
X = np.random.RandomState(256).normal(size=p1*p2*n).reshape((n, p1, p2))
Y = np.einsum('nij,ij->n', X, B) + np.random.RandomState(781).normal(size=n, scale=sd_true)

# basic parameters used in simulations for tensor regression         
r = 10             
sample_size = 1000  
rcond = 0.05
sd = False 

# choose penalty parameter
print('Finding penalty parameter ...')
ftr = TensorRegression_CV(X, Y, r, n_lamb = 20, k_fold = 10, tol = 1e-3, debias = True, rcond = 0.05)
lamb = ftr.lamb_sel

# generate fiducial samples
print('Generating fiducial sample ...')
model = TensorRegression(X, Y, r)
fsg = Fiducial_Sample_Generator(model = model, sd = sd, lamb = lamb, rcond = rcond, sample_size = sample_size)
fsg.get_samples()

# analyze fiducial samples and summarize the evaluation scores
print('Analyzing result ...')
re = Anal_Samples(fsg)
#re.anal_samples(B, ci_level = 0.9, debiased = False, thre = True, show_figures = False, sparse_case = True)
a = re.summary_report(B, folder = 'results/result_tr', sparse_case = True)