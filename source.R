# List of required packages
required_packages <- c("netcoh", "pracma")

# Install missing packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
  }
}
install_if_missing(required_packages)

# Import required packages
library(netcoh)
library(pracma)

# Function to generate simulated data
data_generation = function(n, p, b1, b2, s, seed = 206){
  B = matrix(c(b1, b2, b2, b2, b1, b2, b2, b2, b1), nrow = 3)
  P = kronecker(B, matrix(1, n/3, n/3))
  A = matrix(0, n, n)
  set.seed(seed)
  for(i in 1:n){
    for(j in i:n){
      A[i, j] = rbinom(1, 1, P[i, j])
      A[j, i] = A[i, j]
    }
  }
  diag(A) = 0
  D = diag(rowSums(A))
  L = D - A
  
  X = matrix(rnorm(n*p), ncol=p)
  beta <- 1*matrix(rnorm(p,1,1))
  alpha = matrix(c(rep(-1, n/3), rep(0, n/3), rep(1, n/3)) + rnorm(n) * s)
  group = as.factor(c(rep(1,n/3),rep(2,n/3),rep(3,n/3)))
  mu = X %*% beta + alpha
  
  Lalpha = norm(L%*%alpha, type='2')
  
  return(list(A = A, X = X, alpha = alpha, beta = beta, mu = mu, group = group,
              Lalpha = Lalpha, b1 = b1, b2 = b2, s = s))
}

# function to split data into k_fold
KFold = function(n,k_fold){
  num_test = n%/%k_fold
  num_test_group = num_test%/%3
  num_train = n-num_test
  new_order1 = sample(1:(n/3))
  new_order2 = sample((1+(n/3)):(2*n/3))
  new_order3 = sample((2*n/3+1):n)
  
  indice = lapply(1:k_fold, function(k){
    test_index = sort(c(new_order1[((k-1)*num_test_group+1):(k*num_test_group)],new_order2[((k-1)*num_test_group+1):(k*num_test_group)],new_order3[((k-1)*num_test_group+1):(k*num_test_group)]))
    train_index = setdiff(1:n,test_index)
    return(list(train_index=train_index,test_index=test_index))
  })
  return(indice)
}

# function to do cross validation to choose penalty parameter and estimate noise scale sd
cv = function(A, X, Y, nLamb, k_fold, gamma){
  n = length(Y)
  cv_index = KFold(n,k_fold)
  cv_sse_be = rep(0,nLamb)
  cv_sse_de = rep(0,nLamb)

  Lamb = seq(0.1,3,length.out = nLamb)
  
  for (i in 1:nLamb){
    sse_be = rep(NA,k_fold)
    sse_de = rep(NA,k_fold)
    for(k in seq(k_fold)){
      n_train = length(cv_index[[k]]$train_index)
      n_test = length(cv_index[[k]]$test_index)
      Y_train = Y[cv_index[[k]]$train_index,]
      X_train = X[cv_index[[k]]$train_index,]
      Y_test = Y[cv_index[[k]]$test_index,]
      X_test = X[cv_index[[k]]$test_index,]
      
      A_train = A[cv_index[[k]]$train_index,cv_index[[k]]$train_index]
      A_test = A[cv_index[[k]]$test_index,cv_index[[k]]$test_index]
      A_cross = A[cv_index[[k]]$test_index,cv_index[[k]]$train_index]
      A_left = rbind(A_train,A_cross)
      A_right = rbind(t(A_cross),A_test)
      fullA = cbind(A_left,A_right)
      
      D = diag(colSums(fullA))
      L = D-fullA
      L_train = L[1:n_train,1:n_train]
      L_test = L[(n_train+1):(n_train+n_test),(n_train+1):(n_train+n_test)]
      L_cross = L[(n_train+1):(n_train+n_test),1:n_train]
      
      debias_mat_train = debias_mat(A_train, X_train)
      re = netreg_eta(A_train,X_train,Y_train,debias_mat_train,Lamb[i])
      
      beta_pre_be = re$beta_be
      alpha_pre_be = -pinv(L_test)%*%L_cross%*%re$alpha_be
      Y_pre_be = X_test%*%beta_pre_be+alpha_pre_be
      
      beta_pre_de = re$beta_de
      alpha_pre_de = -pinv(L_test)%*%L_cross%*%re$alpha_de
      Y_pre_de = X_test%*%beta_pre_de+alpha_pre_de
      
      sse_k_be = mean((Y_test-Y_pre_be)^2)
      sse_k_de = mean((Y_test-Y_pre_de)^2)
    
      sse_be[k] = sse_k_be
      sse_de[k] = sse_k_de
    }
    # cv_sse_be[i] = mean(sse_be[1])
    # cv_sse_de[i] = mean(sse_de[2])
    cv_sse_be[i] = mean(sse_be)
    cv_sse_de[i] = mean(sse_de)
  }
  return(list('sigma2_be'=cv_sse_be,'sigma2_de'=cv_sse_de,'lambs'=Lamb,'lamb_sel'=Lamb[which.min(cv_sse_de)]))
}

# function to calculate matrices used in debiasing process
# like the pseudo inverse of hessian, projection matrix of L, L^(1/2), L^(-1/2)
debias_mat = function(A,X){
  n = dim(X)[1]
  p = dim(X)[2]
  
  D = diag(colSums(A))
  L = D-A
  
  L_svd = svd(L)
  L_rank = sum(L_svd$d>.Machine$double.eps^(2/3))
  L_svd$d[(L_rank+1):n] = 0
  
  L_half = L_svd$u%*%diag((L_svd$d^0.5))%*%t(L_svd$v)
  P_L = L_svd$u[1:n,1:L_rank]%*%t(L_svd$u[1:n,1:L_rank])
  
  L_sv_inv_half = c((L_svd$d[1:L_rank])^-0.5,rep(0,n-L_rank))
  L_inv_half = L_svd$v%*%diag(L_sv_inv_half)%*%t(L_svd$u)
  
  rsvds = min(L_svd$d[1:L_rank])/L_svd$d[1:L_rank]
  rcond = (rsvds[which.max(rsvds[2:L_rank]-rsvds[1:(L_rank-1)])]+rsvds[which.max(rsvds[2:L_rank]-rsvds[1:(L_rank-1)])+1])/2
  
  sel = which(min(L_svd$d[1:L_rank])/L_svd$d[1:L_rank]<=rcond)
  L_svd$d[sel] = 0
  hessian_inv = L_svd$u%*%diag(L_svd$d)%*%t(L_svd$v)
  
  return(list('P_L'=P_L,'L_half'=L_half, 'L_inv_half'=L_inv_half, 'hessian_inv'=hessian_inv,'rcond' = rcond))
}

# function to solve the optimization problem and also to debias
netreg_eta = function(A, Xp, Y, debias_mat, lambda){
  n = dim(Xp)[1]
  p = dim(Xp)[2]
  
  Y = matrix(Y,n)
  
  ml_be = rncreg(A,lambda = lambda,Y = Y,X = Xp, gamma = 0, model = 'linear',cv = NULL)
  beta_be = ml_be$beta
  alpha_be = ml_be$alpha
  eta_be = debias_mat$L_half%*%alpha_be
  err_be = Y-Xp%*%beta_be-alpha_be
  
  theta_first = -(debias_mat$L_inv_half)%*%err_be
  
  eta_bias = debias_mat$hessian_inv%*%theta_first
  
  alpha_de = (diag(n)-debias_mat$P_L)%*%alpha_be + debias_mat$L_inv_half%*%(eta_be - eta_bias)
  beta_de = solve(crossprod(Xp))%*%t(Xp)%*%(Y-alpha_de)
  err_de = Y-Xp%*%beta_de-alpha_de

  return(list('alpha_be'=alpha_be,'beta_be'=beta_be,'err_be'=err_be,'alpha_de'=alpha_de,'beta_de'=beta_de,'err_de'=err_de))
}

# function to generate fiducial samples
fiducial_sampler = function(A, X, Y, debias_mat1, sigma, lambda, sample_size){
  n = length(Y)
  p = ncol(X)
  
  # generate fiducial samples before and after debiasing
  u = matrix(rnorm(n*sample_size),n)
  newY = apply(u*sigma,2,'+',Y)
  re_samples = apply(X=newY, MARGIN=2,FUN = netreg_eta,A = A,Xp = X,debias_mat = debias_mat1,
                     lambda = lambda)
  
  # calculate loss for each fiducial sample and select those with relatively small losses
  # losses of samples w/o debiasing
  loss_be = sapply(seq(sample_size),function(i){return(sum((re_samples[[i]]$err_be-2*sigma*u[,i])^2))})
  quantiles_be = quantile(loss_be)
  thre_be = (quantiles_be[4]-quantiles_be[2])*1.5+quantiles_be[4]
  sel_be = loss_be < thre_be
  
  # losses of samples w/ debiasing
  loss_de = sapply(seq(sample_size),function(i){return(sum((re_samples[[i]]$err_de-2*sigma*u[,i])^2))})
  quantiles_de = quantile(loss_de)
  thre_de = (quantiles_de[4]-quantiles_de[2])*1.5+quantiles_de[4]
  sel_de = loss_de < thre_de
  
  # get fiducial samples of alpha, beta and mu
  alpha_be = sapply(re_samples[sel_be],function(z){return(z$alpha_be)})
  alpha_de = sapply(re_samples[sel_de],function(z){return(z$alpha_de)})
  beta_be = sapply(re_samples[sel_be],function(z){return(z$beta_be)})
  beta_de = sapply(re_samples[sel_de],function(z){return(z$beta_de)})
  mu_be =X%*%beta_be+alpha_be
  mu_de =X%*%beta_de+alpha_de
  
  return(list(alpha_be = alpha_be, alpha_de = alpha_de, beta_be = beta_be, beta_de = beta_de, mu_be = mu_be, mu_de = mu_de))
}

# find the point estimate for alpha and beta
point_est = function(samples){
  mean_alpha_be = rowMeans(samples$alpha_be)
  mean_alpha_de = rowMeans(samples$alpha_de)
  mean_beta_be = rowMeans(samples$beta_be)
  mean_beta_de = rowMeans(samples$beta_de)
  return(list(alpha_be_est= mean_alpha_be, alpha_de_est = mean_alpha_de, beta_be_est = mean_beta_be, beta_de_est = mean_beta_de))
}

# function to find the ci for alpha and beta at certain confidence level
ci = function(samples, ci_level = 0.95){
  lb_alpha_be = apply(samples$alpha_be,1,quantile,(1-ci_level)/2)
  ub_alpha_be = apply(samples$alpha_be,1,quantile,1-(1-ci_level)/2)
  
  lb_alpha_de = apply(samples$alpha_de,1,quantile,(1-ci_level)/2)
  ub_alpha_de = apply(samples$alpha_de,1,quantile,1-(1-ci_level)/2)
  #coverage_alpha_de_90 = mean(alpha >= lb_alpha_de_90 & alpha <= ub_alpha_de_90)

  lb_beta_be = apply(samples$beta_be,1,quantile,(1-ci_level)/2)
  ub_beta_be = apply(samples$beta_be,1,quantile,1-(1-ci_level)/2)

  lb_beta_de = apply(samples$beta_de,1,quantile,(1-ci_level)/2)
  ub_beta_de = apply(samples$beta_de,1,quantile,1-(1-ci_level)/2)

  return(list(lb_alpha_be = lb_alpha_be, lb_alpha_de = lb_alpha_de, 
              lb_beta_be = lb_beta_be, lb_beta_de = lb_beta_de,
              ub_alpha_be = ub_alpha_be, ub_alpha_de = ub_alpha_de, 
              ub_beta_be = ub_beta_be, ub_beta_de = ub_beta_de))
}