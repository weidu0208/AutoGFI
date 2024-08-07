import numpy as np
from scipy.stats import ortho_group  
from lib.MatrixCompletion import MatrixCompletion, MatrixCompletion_CV
from lib.Fiducial_Sampler import Fiducial_Sample_Generator, Anal_Samples
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# simulated data
m = 250
n = 250
r = 2
p_true  = 0.4
sd_true = 1e-3

print('Simulated data generation is in progress...')
U = ortho_group.rvs(dim = n, random_state = 265)[:,:r]
V = ortho_group.rvs(dim = n, random_state = 178)[:,:r]
M = U@V.T

noise =  np.random.RandomState(13317).normal(loc=0, scale=sd_true, size=n*n)
omega_vec =np.random.RandomState(8217).binomial(n=1, p=p_true, size=n*n)

Omega = np.reshape(omega_vec, (m,n))
Y = (M + np.reshape(noise, (m,n)))*Omega

# basic parameters used in simulations for matrix completion
p = False
sigma = False
rcond = 0.05
# sample_size smaller than 30 may raise error
sample_size = 1000

# choose penalty parameter
print('Finding penalty parameter. This may take some time ...')
mc_cv = MatrixCompletion_CV(Y, Omega, r, p, n_lamb = 10, k_fold = 10, rcond = rcond, tol = 1e-3)
lamb = mc_cv.lamb_sel

# generate fiducial samples
print('Generating fiducial sample ...')
model = MatrixCompletion(Y, Omega, r, p)
fsg = Fiducial_Sample_Generator(model=model, sd=sigma, lamb=lamb, rcond=rcond, sample_size=sample_size)
fsg.get_samples()

# analyze fiducial samples and summarize the evaluation scores
print('Analyzing result ...')
re = Anal_Samples(fsg)
#re.anal_samples(M, ci_level = 0.9, debiased = True, thre = True, show_figures = False)
summary = re.summary_report(M, folder = './results/result_mc', sparse_case = False)