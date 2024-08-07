source('./source.R')
options(warn = -1)

n = 300  # number of observations
s = 0.1  # noise for alpha
sigma = 0.5  # noise for Y
p = 10  # number of features of beta

b1 = 0.2 
b2 = 0.01

replicates = 100 # number of replicate experiments
sample_size = 1000 # number of fiducial samples to generate for each replicate experiment

# generate simulated data
cat('Simulated data generation is in progress... \n')
sim_data = data_generation(n, p, b1, b2, s, seed=1028)
X = sim_data$X; A = sim_data$A; mu = sim_data$mu; alpha = sim_data$alpha; beta = sim_data$beta; group = sim_data$group

# evaluate the fiducial method on estimating beta
cat('The replicated simulation experiments are beginning. This process may take some time. Please be patient...\n')

# w/o debiasing
beta_be = matrix(NA,replicates,p)
beta_rmse_be = rep(NA,replicates)
CI90beta_be = CI95beta_be = CI99beta_be = matrix(NA, p, replicates)
WD90beta_be = WD95beta_be = WD99beta_be = matrix(NA, p, replicates)

# w/ debiasing
beta_de = matrix(NA,replicates,p)
beta_de_rmse = rep(NA,replicates)
CI90beta_de = CI95beta_de = CI99beta_de = matrix(NA, p, replicates)
WD90beta_de = WD95beta_de = WD99beta_de = matrix(NA, p, replicates)

debias_mat1 = debias_mat(A,X)
# Create a progress bar
pb <- txtProgressBar(min = 0, max = replicates, style = 3)
for(iter in 1:replicates){
  Y0 = mu + rnorm(n, 0, sigma)
  
  # choose penalty parameter by cross validation and estimate noise scale
  rep_cv = cv(A, X, Y0, nLamb = 10,k_fold = 10,gamma = 0)
  lambda = rep_cv$lamb_sel
  sigma_est = sqrt(min(rep_cv$sigma2_de))
  
  # generate fiducial samples
  rep_one = fiducial_sampler(A, X, Y0, debias_mat1, sigma_est, lambda, sample_size)
  
  # analyze fiducial samples
  rep_pest = point_est(rep_one)
  rep_ci_90 = ci(rep_one, 0.9)
  rep_ci_95 = ci(rep_one, 0.95)
  rep_ci_99 = ci(rep_one, 0.99)
  
  beta_rmse_be[iter] = sqrt(mean((rep_pest$beta_be_est-beta)^2))
  beta_be[iter,] = rep_pest$beta_be_est
  beta_de_rmse[iter] = sqrt(mean((rep_pest$beta_de_est-beta)^2))
  beta_de[iter,] = rep_pest$beta_de_est
  
  ub = rep_ci_90$ub_beta_be
  lb = rep_ci_90$lb_beta_be
  CI90beta_be[,iter] = 1-(lb<beta & ub>beta)
  WD90beta_be[,iter] = ub-lb
  ub = rep_ci_90$ub_beta_de
  lb = rep_ci_90$lb_beta_de
  CI90beta_de[,iter] = 1-(lb<beta & ub>beta)
  WD90beta_de[,iter] = ub-lb
  
  ub = rep_ci_95$ub_beta_be
  lb = rep_ci_95$lb_beta_be
  CI95beta_be[,iter] = 1-(lb<beta & ub>beta)
  WD95beta_be[,iter] = ub-lb
  ub = rep_ci_95$ub_beta_de
  lb = rep_ci_95$lb_beta_de
  CI95beta_de[,iter] = 1-(lb<beta & ub>beta)
  WD95beta_de[,iter] = ub-lb
  
  ub = rep_ci_99$ub_beta_be
  lb = rep_ci_99$lb_beta_be
  CI99beta_be[,iter] = 1-(lb<beta & ub>beta)
  WD99beta_be[,iter] = ub-lb
  ub = rep_ci_99$ub_beta_de
  lb = rep_ci_99$lb_beta_de
  CI99beta_de[,iter] = 1-(lb<beta & ub>beta)
  WD99beta_de[,iter] = ub-lb
  
  # if(iter%%50==0){cat('Running Dataset:',iter,'\n')}
  setTxtProgressBar(pb, iter)
}
close(pb)
cat('The replicated simulation experiments have ended.\n')

df = data.frame( beta_rmse_de = mean(beta_de_rmse), beta_rmse_be = mean(beta_rmse_be),
                 beta_cov_de_90 = mean(CI90beta_de==0), beta_width_de_90 = mean(WD90beta_de),
                 beta_cov_be_90 = mean(CI90beta_be==0), beta_width_be_90 = mean(WD90beta_be),
                 beta_cov_de_95 = mean(CI95beta_de==0), beta_width_de_95 = mean(WD95beta_de),
                 beta_cov_be_95 = mean(CI95beta_be==0), beta_width_be_95 = mean(WD95beta_be),
                 beta_cov_de_99 = mean(CI99beta_de==0), beta_width_de_99 = mean(WD99beta_de),
                 beta_cov_be_99 = mean(CI99beta_be==0), beta_width_be_99 = mean(WD99beta_be))
print(df)

# save results
if (!dir.exists("./results/")){
  dir.create("./results")
}
if (!dir.exists("./results/result_nr")){
  dir.create("./resultsresult_nr")
}
write.csv(df, "./results/result_nr/GFI_df.csv")


