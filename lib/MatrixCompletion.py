import numpy as np
from sklearn.model_selection import KFold
import jax.numpy as jnp
from jax import grad, linearize, jacfwd, jacrev, jit
# from jax.config import config
# config.update('jax_platform_name', 'cpu')

class MatrixCompletion():
    """
    A class used to solve noise matrix completion problem: M = A@t(B) + sd*U.

    Parameters
    ----------
    Y : array-like of shape (m, n)
        Observed matrix with missing entries filled up by 0.
    Omega : array-like of shape(m, n), the same shape of Y
        Binary matrix indicates which entry is observed. 1 for observed and 0 for missing.
    r : int
        Rank of the true matrix
    p : float or bool.
        Observation rate. If it's known, type in the number otherwise using the default value "False" and estimate it later. 

    Attributes
    ----------
    Y : array-like of shape (m, n)
        Observed matrix with missing entries filled up by 0.
    Omega : array-like of shape(m, n), the same shape of Y
        Binary matrix indicates which entry is observed. 1 for observed and 0 for missing.
    m : int
        Row dimension of the observed matrix Y
    n : int 
        Column dimension of the observed matrix Y 
    r : int
        Rank of the true matrix
    p : float
        (Estimated) observation rate.
    theta : array-like of shape ((m+n)*r,)
        Vector of all estimated parameters before debiasing.
    theta_debiased : array-like of shape ((m+n)*r,)
        Vector of all estimated parameters after debiasing.
    sd : float
        True or estimated noise scale.

    Methods
    -------
    theta_to_target
        Convert theta to the parameters in target form.
    risk
        Calculate risk given the parameter theta.
    fit
        Fit the matrix completion model.
    debias
        Debias the estimated parameters given by fit.
    sd_est
        Estimate the noise scale.
    """

    def __init__(self, Y, Omega, r, p = False):
        self.Y = Y
        self.m, self.n = self.Y.shape
        self.Omega = Omega
        self.r = r
        if not p:
            self.p = jnp.sum(self.Omega)/self.m/self.n
        else:
            self.p = p
    
    def theta_to_target(self, theta):
        """Calculate the matrix in interest by theta.

        Args:
            theta (array-like of shape (r*(m+n),))
            Vector of all unknown parameters, (vec(A),vec(B))

        Returns:
            array-like of shape (m, n): A@B.T
        """
        return theta[0:self.m*self.r].reshape((self.m, self.r))@theta[self.n*self.r:].reshape((self.n, self.r)).T

    def risk(self, theta):
        """Calculate risk/loss given the parameter theta.

        Args:
            theta (array-like of shape ((m+n)*r, ))
            Vector of all unknown parameters.
        Returns:
            float: risk or loss
        """
        Y_hat = self.theta_to_target(theta)*self.Omega
        return jnp.sum((self.Y-Y_hat)**2)
    
    def fit(self, lamb, eta=0.01, tol=1e-6, maxiter=20000):
        """Fit the matrix completion model.

        Args:
            lamb (float): penalty parameter.
            eta (float): learning rate. Defaults to 0.01.
            tol (float): tolerance. Defaults to 1e-6.
            maxiter (int): maximum iteration. Defaults to 20000.

        Returns:
            array-like of shape (m, n), same shape of the observed matrix: estimated whole matrix before debiasing based on observations. 
        """
        # make sure not observed entries have value 0
        self.Y = self.Y * self.Omega
        # initialization
        u, d, vt = jnp.linalg.svd(self.Y/self.p)    
        A = u[:, 0:self.r]*(d[0:self.r]**0.5)
        B = (vt[0:self.r, :].T)*(d[0:self.r]**0.5)
        Y_t = (A@B.T)*self.Omega
        grad_x = 1/self.p*((Y_t-self.Y)@B + lamb*A)
        grad_y = 1/self.p*((Y_t-self.Y).T@A + lamb*B)
        loss_curr = jnp.sum((self.Y-Y_t)**2) + lamb*(jnp.sum(A**2)+jnp.sum(B**2)) 
        loss_prev = jnp.finfo(np.float64).max
        # keep info of the current minimum loss
        loss_argmin_A = A
        loss_argmin_B = B
        loss_min = loss_curr
        
        itcounter = 0
        while itcounter < maxiter and abs(loss_prev-loss_curr) > tol:
            loss_prev = loss_curr            
            A =  A - eta*grad_x
            B = B - eta*grad_y

            Y_t = (A@B.T)*self.Omega
            diff = Y_t-self.Y
            grad_x = 1/self.p*(diff@B + lamb*A)
            grad_y = 1/self.p*(diff.T@A + lamb*B)            
            loss_curr = jnp.sum(diff**2) + lamb*(jnp.sum(A**2)+jnp.sum(B**2))
            
            # if loss goes to infinity, decrease the step rate and start from the point which gives the minimun loss.
            if np.isinf(loss_curr):
                print('learning step is too large and loss goes to infinity!')
                eta = eta/5
                A = loss_argmin_A
                B = loss_argmin_B
                loss_curr = loss_min
                loss_prev = jnp.finfo(np.float64).max
                Y_t = (A@B.T)*self.Omega
                grad_x = 1/self.p*((Y_t-self.Y)@B + lamb*A)
                grad_y = 1/self.p*((Y_t-self.Y).T@A + lamb*B)
            
            # update the minimum loss info.                    
            if loss_curr < loss_min:
                loss_min = loss_curr
                loss_argmin_A = A
                loss_argmin_B = B
            
            itcounter += 1

        self.theta = jnp.hstack((A.flatten(), B.flatten()))

        return A@B.T

    def debias(self, rcond):
        """Debias for the parameters.

        Args:
            rcond (float): threshold for finding pseudo inverse of the hessian matrix

        Returns:
            array-like of shape (m, n), same shape of the observed matrix: estimated whole matrix after debiasing based on observations. 
        """
        self.rcond = rcond

        indice = jnp.nonzero(self.theta)[0]

        # gradient of the risk function
        jacrev_risk = jacrev(self.risk)
        grad_risk = jacrev_risk(self.theta)[indice]

        # hessian matrix of the risk function
        # 2nd way is to avoid memory limit exceeded for very large data
        try:
            hessian_risk = jacfwd(jacrev_risk)
            hessian_risk = jit(hessian_risk)
            hessian = hessian_risk(self.theta)[indice][:, indice]
        except:
            def hessian_risk(x):
                _, hvp = linearize(grad(self.risk), x)
                hvp = jit(hvp)
                basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
                return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
            hessian = hessian_risk(self.theta)[indice][:, indice]
    
        pinv = jnp.linalg.pinv(hessian, rcond=rcond)
        bias = jnp.dot(pinv, grad_risk)
        self.theta_debiased = self.theta.at[indice].set(
            self.theta[indice] - bias)
         
        return self.theta_to_target(self.theta_debiased)
    
    def sd_est(self):
        """ Estimate noise scale by MAD

        Returns:
            float: estimated noise scale
        """
        obs_index = jnp.where(self.Omega == 1)
        Y_hat = self.theta_to_target(self.theta_debiased)
        z = (self.Y-Y_hat)[obs_index[0], obs_index[1]]
        self.sd = jnp.median(jnp.abs(z-jnp.median(z)))/0.67449   
        return self.sd
    
class MatrixCompletion_CV():
    """Class used to find penalty parameter by cross validation.

    Parameters
    ----------
    Y : array-like of shape (m, n)
        Observed matrix with missing entries filled up by 0.
    Omega : array-like of shape(m, n), the same shape of Y
        Binary matrix indicates which entry is observed. 1 for observed and 0 for missing.
    r : int
        Rank of the true matrix
    p : float or bool.
        Observation rate. If it's known, type in the number otherwise using the default value "False" and estimate it later.
    n_lamb : int
        Number of parameters to try. Defaults to 20.
    k_fold : int
        Number of fold. Defaults to 10.
    tol : float
        Tolerance for the fitting procedure. Defaults to 1e-6. 
    debias : bool
        Whether debiasing or not. If True then debias. Defaults to True.
    rcond : float
        Threshold for finding pseudo inverse of the hessian matrix. Defaults to 0.05.
    
    Attributes
    ----------
    Y : array-like of shape (m, n)
        Observed matrix with missing entries filled up by 0.
    Omega : array-like of shape(m, n), the same shape of Y
        Binary matrix indicates which entry is observed. 1 for observed and 0 for missing.
    Omega_index : tuple of format (array-like of shape (m,), array-like of shape (n,))
        Indices of the observed entries.
    m : int
        Row dimension of the observed matrix Y
    n : int 
        Column dimension of the observed matrix Y 
    r : int
        Rank of the true matrix
    p : float
        (Estimated) observation rate.
    k_fold : int
        Number of fold. Defaults to 10.
    k_index : list(tuple(ndarray, ndarray))
        List of tuples of training set indices and testing set indices.
    tol : float
        Tolerance for the fitting procedure. Defaults to 1e-6. 
    debias : bool
        Whether debiasing or not. If True then debias. Defaults to True.
    rcond : float
        Threshold for finding pseudo inverse of the hessian matrix. Defaults to 0.05.
    lamb : array-like of shape (n_lamb,)
        Penalty parameters tried.
    lamb_loss : array-like of shape (n_lamb,) 
        Test set loss of each penalty parameter.
    lamb_sel : float
        Penalty parameter with the minimum loss.
    

    Methods
    -------
    cv_k
        Fit the model on the k-th fold and calculate the loss.
    """

    def __init__(self, Y, Omega, r, p = False, n_lamb = 20, k_fold = 10, tol = 1e-6, debias = True, rcond = 0.05):
        self.Y = Y
        self.Omega = Omega
        self.Omega_index = jnp.where(self.Omega == 1)
        self.m, self.n = self.Y.shape
        self.r = r
        if not p:
            self.p = len(self.Omega_index[0])/self.m/self.n
        else:
            self.p = p

        self.k_fold = k_fold
        kf = KFold(n_splits=k_fold, shuffle=True)
        self.k_index = [x for x in kf.split(self.Omega_index[0])]

        self.rcond = rcond
        self.debias = debias
        self.tol = tol

        self.lamb = 10**np.arange(-2, -4, -2/(n_lamb-1))*self.n*self.p
        self.lamb_loss = []
        for lamb in self.lamb:
            self.lamb_loss.append(np.mean([self.cv_k(k, lamb, self.debias) for k in range(self.k_fold)]))                
        self.lamb_sel = self.lamb[np.argmin(self.lamb_loss)]

    def cv_k(self, k, lamb, debias):
        """Fit the model on the k-th fold and calculate the loss.

        Args:
            k (int): k-th fold
            lamb (float): penalty parameter
            debias (bool): whether to debias, if True then debiasing is applied

        Returns:
            float: loss or risk on the k-th fold
        """
        train_index, test_index = self.k_index[k]
        Omega_index_train = (
            self.Omega_index[0][train_index], self.Omega_index[1][train_index])
        Omega_index_test = (
            self.Omega_index[0][test_index], self.Omega_index[1][test_index])

        Y_train = jnp.zeros((self.m, self.n))
        Y_train = Y_train.at[Omega_index_train[0], Omega_index_train[1]].set(
            self.Y[Omega_index_train[0], Omega_index_train[1]])
        
        Omega_train = jnp.zeros((self.m, self.n))
        Omega_train = Omega_train.at[Omega_index_train[0], Omega_index_train[1]].set(
            1)
        
        ft = MatrixCompletion(Y_train, Omega_train, r=self.r, p = self.p)
        Y_hat = ft.fit(lamb=lamb, tol=self.tol)
        if debias:
            Y_hat = ft.debias(self.rcond)
        
        err = jnp.sum(((Y_hat-self.Y)[Omega_index_test[0], Omega_index_test[1]])**2)
        return err