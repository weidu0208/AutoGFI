from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, linearize, grad
# from jax.config import config
# config.update('jax_platform_name', 'cpu')

class TensorRegression():
    """
    A class used to solve noise matrix completion problem: M = A@t(B) + sd*U.

    Parameters
    ----------
    Y : array-like of shape (m, n)
        Observed matrix with missing entries filled up by 0.
    X : array-like of shape(n, p1, p2)
        Tensor predictor.
    r : int
        Rank of the tensor predictor.

    Attributes
    ----------
    Y : array-like of shape (n, )
        Response vector.
    X : array-like of shape(n, p1, p2)
        Tensor predictor.
    n : int
        Sample size.
    p1 : int 
        Row dimension of the tensor predictor. 
    p2 : int
        Column dimension of the tensor predictor. 
    r : int
        Rank of the tensor predictor.
    theta : array-like of shape ((p1+p2)*r,)
        Vector of all estimated parameters before debiasing.
    theta_debiased : array-like of shape ((p1+p2)*r,)
        Vector of all estimated parameters after debiasing.

    Methods
    -------
    theta_to_target
        Convert theta to the parameters in target form.
    risk
        Calculate risk given the parameter theta.
    fit
        Fit the tensor regression model.
    debias
        Debias the estimated parameters given by fit.
    sd_est
        Estimate the noise scale.
    """
    def __init__(self, X, Y, r):
        self.X = X
        self.Y = Y
        self.n, self.p1, self.p2 = X.shape
        self.r = r

    def theta_to_target(self,theta):
        """Calculate the tensor in interest by theta.

        Args:
            theta (array-like of shape ((p1+p2)*r,))
            Vector of all unknown parameters.

        Returns:
            array-like of shape (p1, p2): tensor predictor formed by theta
        """
        B1 = theta[0:self.p1*self.r].reshape((self.r, self.p1))
        B2 = theta[self.p1*self.r:].reshape((self.r, self.p2))
        B = jnp.einsum('ri,rj->ij', B1, B2)
        return B
    
    def risk(self, theta):
        """Calculate risk/loss given the parameter theta.

        Args:
            theta (array-like of shape ((p1+p2)*r, ))
            Vector of all unknown parameters.
        Returns:
            float: risk or loss
        """
        B = self.theta_to_target(theta)
        Y_hat = jnp.tensordot(self.X, B, axes=2)
        return jnp.mean((self.Y-Y_hat)**2)

    def fit(self, lamb, tol=1e-6, maxiter=20000):
        """Fit the matrix completion model.

        Args:
            lamb (float): penalty parameter.
            tol (float): tolerance. Defaults to 1e-6.
            maxiter (int): maximum iteration. Defaults to 20000.

        Returns:
            array-like of shape (p1, p2): estimated tensor coefficient before debiasing. 
        """
        B1 = np.random.normal(size=self.r*self.p1).reshape((self.r, self.p1))
        B2 = np.random.normal(size=self.r*self.p2).reshape((self.r, self.p2))

        eps = 1
        t = 0
        l_new = 1
        while t < maxiter and np.max(np.abs(B2)) > 1e-16 and eps > tol:
            t += 1
            l_old = l_new

            X1 = np.einsum('nij,rj->nri', self.X, B2).reshape((self.n, -1))
            fit = Lasso(alpha=lamb/self.n, fit_intercept=False,
                        max_iter=10000).fit(X1, self.Y)
            B1 = fit.coef_.reshape((self.r, self.p1))

            X2 = np.einsum('nij,ri->nrj', self.X, B1).reshape((self.n, -1))
            fit = Lasso(alpha=lamb/self.n, fit_intercept=False,
                        max_iter=10000).fit(X2, self.Y)
            B2 = fit.coef_.reshape((self.r, self.p2))

            l_new = sum((self.Y-X2.dot(B2.flatten()))**2)/2+lamb * \
                (np.sum(np.abs(B1))+np.sum(np.abs(B2)))
            eps = abs(l_new-l_old)

        B = np.einsum('ri,rj->ij', B1, B2)

        self.theta = jnp.array(np.hstack((B1.flatten(), B2.flatten())))

        return B
    
    
    def debias(self,rcond):
        """Debias for the parameters.

        Args:
            rcond (float): threshold for finding pseudo inverse of the hessian matrix

        Returns:
            array-like of shape (p1, p2): estimated tensor coefficient after debiasing based on observations. 
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
        """ Estimate noise scale.

        Returns:
            float: estimated noise scale.
        """        
        self.sd = np.sqrt(self.risk(self.theta_debiased))
        return self.sd


class TensorRegression_CV():
    """Class used to find penalty parameter by cross validation.

    Parameters
    ----------
    Y : array-like of shape (n, )
        Response vector.
    X : array-like of shape(n, p1, p2)
        Tensor predictor.
    r : int
        Rank of the tensor predictor.
    n_lamb : int
        Number of penalty parameters to try. Defaults to 20.
    k_fold : int
        Number of folds for cross validation. Defaults to 10.
    tol : float
        Tolerance for the fitting procedure. Defaults to 1e-6.
    debias : bool
        Whether debiasing or not. If True then debias. Defaults to True.
    rcond : float
        Threshold for finding pseudo inverse of the hessian matrix. Defaults to 0.05.
    
    Attributes
    ----------
    Y : array-like of shape (n, )
        Response vector.
    X : array-like of shape(n, p1, p2)
        Tensor predictor.
    n : int
        Sample size.
    p1 : int 
        Row dimension of the tensor predictor. 
    p2 : int
        Column dimension of the tensor predictor. 
    r : int
        Rank of the tensor predictor.
    tol : float
        Tolerance for the fitting procedure. Defaults to 1e-6. 
    debias : bool
        Whether debiasing or not. If True then debias. Defaults to True.
    rcond : float
        Threshold for finding pseudo inverse of the hessian matrix. Defaults to 0.05.
    k_fold : int
        Number of fold. Defaults to 10.
    k_index : list(tuple(ndarray, ndarray))
        List of tuples of training set indices and testing set indices.
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

    def __init__(self, X, Y, r, n_lamb=20, k_fold=10, tol=1e-3, debias = True, rcond=0.05):
        self.X = X
        self.Y = Y
        self.n, self.p1, self.p2 = self.X.shape
        self.r = r
        
        self.tol = tol
        self.debias = debias
        self.rcond = rcond


        self.k_fold = k_fold
        kf = KFold(n_splits=self.k_fold, shuffle=True)
        self.k_index = [x for x in kf.split(X)]
        
        self.lamb = 10**np.arange(-0, -1, -1/(n_lamb-1))*self.n
        self.lamb_loss = []
        for lamb in self.lamb:
            self.lamb_loss.append(np.mean([self.cv(k, lamb,self.debias) for k in range(self.k_fold)]))                
        self.lamb_sel = self.lamb[np.argmin(self.lamb_loss)]
    
    def cv(self, k, lamb, debias):
        """Fit the model on the k-th fold and calculate the loss.

        Args:
            k (int): k-th fold
            lamb (float): penalty parameter
            debias (bool): whether to debias, if True then debiasing is applied

        Returns:
            float: loss or risk on the k-th fold
        """
        train_index, test_index = self.k_index[k]
        X_train = self.X[train_index]   
        Y_train = self.Y[train_index]
        X_test = self.X[test_index]
        Y_test = self.Y[test_index]
        a = TensorRegression(X=X_train, Y=Y_train, r=self.r)
        B_hat = a.fit(lamb, tol = self.tol)
        if debias:
            B_hat = a.debias(self.rcond)
        Y_hat = jnp.tensordot(X_test, B_hat, axes=2)
        return jnp.mean((jnp.array(Y_test)-Y_hat)**2)
        


