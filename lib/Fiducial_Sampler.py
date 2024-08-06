import os
import time
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from lib.utils import convert_to_preferred_format, progress_bar

class Fiducial_Sample_Generator():
    """
    Class used to generate fiducial samples.

    Parameters
    ----------
    model : instance of a class which fits the model
        An instance of the class which fits the model. For example, class like MatrixCompletion or TensorRegression.
    sd : bool or float, defaults to False
        If the noise scale is unknown then False. Otherwise type in the value of the noise scale.
    lamb :  float, defaults to 0.01
        Penalty parameter used in the fitting procedure.
    rcond : float, defaults to 0.05
        Threshold for finding pseudo inverse of the hessian matrix.
    sample_size :  int
        Number of fiducial samples to generate.

    Attributes
    ----------
    model : instance of a class which fits the model
        An instance of the class which fits the model. For example, class like MatrixCompletion or TensorRegression.
    Y : array-like
        The Observed data. For example, the response vector in linear model or the observed matrix in matrix completion.
    lamb :  float
        Penalty parameter used in the fitting procedure.
    rcond : float
        Threshold for finding pseudo inverse of the hessian matrix.
    sample_size :  int
        Number of fiducial samples generated.
    sd : float
        True or estimated noise scale.
    samples_theta_biased: array like of shape (n, p), p is the number of unkown parameters
        Fiducial samples for unkown parameters before debiasing. 
    samples_theta_debiased: array like of shape (n, p), p is the number of unkown parameters
        Fiducial samples for unkown parameters after debiasing.
    samples_target_biased: array like of shape (n,...)
        Fiducial samples for parameters in target form before debiasing.
    samples_target_debiased: array like of shape (n,...)
        Fiducial samples for parameters in target form after debiasing.
    loss_biased :  array-like of shape (n,)
        loss for each fiducial sample before debiasing
    loss_debiased :  array-like of shape (n,)
        loss for each fiducial sample after debiasing

    Methods
    -------
    sample_generator
        generate fiducial sample for a given ui
    get_samples
        generate multiple fiducial samples
    """
    def __init__(self, model, sd=False, lamb=0.01, rcond=0.05, sample_size=1000):
        self.model = model
        self.Y = model.Y

        self.lamb = lamb
        self.rcond = rcond
        self.sample_size = sample_size

        if sd:
            self.sd = sd
        else:
            self.model.fit(lamb)
            self.model.debias(rcond)
            self.sd = model.sd_est()
        print('sd is estimated or chosen as', self.sd)


    def sample_generator(self, ui):
        """generate fiducial sample for a given ui

        Args:
            ui (array of the same shape of Y): fiducial noise

        Returns:
            list[array, array]: list of fiducial samples based on ui before and after debiasing.
        """
        self.model.Y = self.Y - self.sd*ui
        self.model.fit(self.lamb)
        self.model.debias(self.rcond)
        biased_sample = self.model.theta
        debiased_sample = self.model.theta_debiased
        return [biased_sample, debiased_sample]

    def get_samples(self):
        """generate multiple fiducial samples
        """
        gfi_tis = time.time()

        # initialization
        self.samples_theta_biased = [None]*self.sample_size
        self.samples_target_biased = [None]*self.sample_size
        self.samples_theta_debiased = [None]*self.sample_size
        self.samples_target_debiased = [None]*self.sample_size
        self.loss_biased = np.empty(self.sample_size)
        self.loss_debiased = np.empty(self.sample_size)
        
        # generate fiducial samples
        for i in range(self.sample_size):
            ui = np.random.normal(size=self.Y.shape)
            sample = self.sample_generator(ui)
            self.samples_theta_biased[i] = sample[0]
            self.samples_target_biased[i] = self.model.theta_to_target(sample[0]).flatten()
            self.samples_theta_debiased[i] = sample[1]
            self.samples_target_debiased[i] = self.model.theta_to_target(sample[1]).flatten()
            self.loss_biased[i] = self.model.risk(sample[0])
            self.loss_debiased[i] = self.model.risk(sample[1])
            progress_bar(self.sample_size, self.sample_size//30, i)
        
        # list to ndarray
        self.samples_theta_biased = np.array(self.samples_theta_biased)
        self.samples_target_biased = np.array(self.samples_target_biased)
        self.samples_theta_debiased = np.array(self.samples_theta_debiased)
        self.samples_target_debiased = np.array(self.samples_target_debiased)

        gfi_tit = time.time()
        # print('Generating %d biased and debiased samples needs %s time \n' %
        #       (self.sample_size, convert_to_preferred_format(gfi_tit-gfi_tis)))


class Anal_Samples():
    """
    Class used to analyze fiducial samples.

    Parameters
    ----------
    fsg : An instance of the class Fiducial_Sample_Generator

    Attributes
    ----------
    Y : array-like
        The Observed data. For example, the response vector in linear model or the observed matrix in matrix completion.
    sd : float
        True or estimated noise scale.
    samples_biased: array like of shape (n, ...)
        Fiducial samples for parameters in target form before debiasing.
    samples_debiased: array like of shape (n, ...)
        Fiducial samples for parameters in target form after debiasing.
    loss_biased :  array-like of shape (n,)
        Loss for each fiducial sample before debiasing
    loss_debiased :  array-like of shape (n,)
        Loss for each fiducial sample after debiasing

    Methods
    -------
    anal_samples
        Find point estimate and confidence intervals based on the fiducial samples and give evaluation scores, like rmse, empirical coverage given the true target parameter.
    summary_report
        Summary the performance of the fiducial methods in different cases.
    """
    def __init__(self, fsg):
        self.Y = fsg.Y
        self.sd = fsg.sd

        # fiducial samples:
        self.samples_biased = fsg.samples_target_biased
        self.samples_debiased =  fsg.samples_target_debiased
        self.loss_biased =  fsg.loss_biased
        self.loss_debiased =  fsg.loss_debiased

    def anal_samples(self, M, ci_level = 0.95, debiased = True, thre = True, show_figures = False, sparse_case = True):
        """Find point estimate and confidence intervals based on the fiducial samples and give evaluation scores, like rmse, empirical coverage given the true target parameter.

        Args:
            M (array-like): True target parameter.
            ci_level (float): Confidence level. Defaults to 0.95.
            debiased (bool): If True then analyze fiducial samples with debiasing process otherwise analyze fiducial samples without debiasing. Defaults to True.
            thre (bool): If True, analyze fiducial samples with losses less than thre otherwise analyze all fiducial samples. Defaults to True.
            show_figures (bool): If True, plot the point estimate and the histogram of the losses. Defaults to False.
            sparse_case (bool): If True, evaluate scores contain scores measured on both significant parameters and zero parameters. Defaults to True.

        Returns:
            dict: dict of point estimate, confidence intervals and evaluation scores of rmse, coverage...
        """
                                                                                                            
        if debiased:
            samples = self.samples_debiased
            loss = self.loss_debiased
        else:
            samples = self.samples_biased
            loss = self.loss_biased

        if thre:
            if type(thre) == bool:
                q3, q1 = np.quantile(loss, np.array([0.75, 0.25]))
                thre_val = q3+(q3-q1)*1.5
            inds = jnp.where(loss < thre_val)[0]
            samples = samples[inds, :]
            sample_size = len(inds)
        else:
            sample_size = samples.shape[0]
        
        case = ''.join([str(ci_level*100),'%',' ','w debiasing' if debiased else 'w/o debiasing', ' ', 'w thre' if thre else 'w/o thre'])
                        
        B_mean = np.nanmean(samples, axis=0).reshape(M.shape)

        q = (1-ci_level)/2
        lower, upper = np.nanquantile(samples, q= np.array([q,1-q]), axis=0)

        cov = (M.flatten() >= lower) * (M.flatten() <= upper)
        width = upper - lower

        if sparse_case:
            ind_sig = np.nonzero(M.flatten())
            ind_0 = np.where(M.flatten() == 0)
            cov_sig = cov[ind_sig]
            width_sig = width[ind_sig]
            cov_0 = cov[ind_0]
            width_0 = width[ind_0]

        if show_figures:
            fig, ax = plt.subplots(ncols=2, constrained_layout=True,figsize=(15, 8))
            im = ax[0].imshow(B_mean, cmap='hot')
            bar = fig.colorbar(im, ax = ax[0])
            bar.set_label('Value')
            ax[1].hist(loss, bins=100)
            if thre:
                ax[1].axvline(x = thre_val, color = 'r', label = 'epsilon')
            fig.suptitle(case)
            plt.show()

        return {'B_est': B_mean, 'lower': lower, 'upper': upper, 'scores':{'rmse': np.mean((B_mean-M)**2)**0.5, 'cov_prop': np.mean(cov), 'width': np.mean(width), 'cov_prop_sig': np.mean(cov_sig), 'width_sig': np.mean(width_sig), 'cov_prop_0': np.mean(cov_0), 'width_0': np.mean(width_0), 'sample_size': sample_size}} if sparse_case else {'B_est': B_mean, 'lower': lower, 'upper': upper, 'scores':{'rmse': np.mean((B_mean-M)**2)**0.5, 'cov_prop': np.mean(cov), 'width': np.mean(width), 'sample_size': sample_size}} 
    
    def summary_report(self, M, folder, sparse_case = True):
        """Summary the performance of the fiducial methods in different cases: w or w/o debiasing, coverage for 0.9, 0.95, 0.99 confidence level

        Args:
            M (array-like): True target parameter.
            folder (str): Which folder to save the result.
            sparse_case (bool): If True, evaluate scores contain scores measured on both significant parameters and zero parameters. Defaults to True.

        Returns:
            pd.DataFrame
        """
        df_summary = pd.DataFrame()
        for ci_level in [0.9, 0.95, 0.99]:
            for debiased in [True, False]:
                for thre in [True]:
                    case = ''.join([str(ci_level*100),'%',' ','w debiasing' if debiased else 'w/o debiasing'])
                    re = self.anal_samples(M, ci_level, debiased, thre, show_figures=False, sparse_case = sparse_case)
                    df = pd.DataFrame(re['scores'], index = [case], dtype = np.float64)
                    df_summary = pd.concat((df_summary, df))
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        df_summary.T.to_csv(folder + '/' + 'GFI_df.csv', index=True)

        with pd.option_context('display.precision', 3, 'display.width', None, 'display.colheader_justify', 'center'):
            print(df_summary.T)

        return df_summary.T

