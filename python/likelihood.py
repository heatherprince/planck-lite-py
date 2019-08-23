import numpy as np
from plik_cmbonly import plik_lite


def logprob(values, names, prior_bound_dict, prior_gaussian_dict, like_obj, class_obj, class_param_dict, model):
    '''
    values - an array the length of names (things that are varying)
    names - a list of parameter names in the same order as theta_p
    prior_bound_dict - a dictionary with upper and lower bounds for flat priors
    prior_gaussian_dict - a dictionary with mu and sigma for Gaussian priors
    model - a function that takes a parameter dictionary and returns the power spectrum
    class_param_dict - a dictionary with all of the necessary CLASS inputs apart from those given in names and values
    like_obj - plik-lite object
    '''
    #priors first to make sure tau is in a sensible range and avoid errors in CLASS
    lp = logprior(values, names, prior_bound_dict, prior_gaussian_dict)
    if not np.isfinite(lp):
        return -np.inf

    #get cl from CLASS
    thetas=class_param_dict.copy()
    for i, n in enumerate(names):
        thetas[n]=values[i]
    #print('dictionary for CLASS', thetas)

    ELLMIN=1
    Dltt, Dlee, Dlte=model(thetas, class_obj, ELLMIN)
    ls = np.arange(Dltt.shape[0]+ELLMIN)[ELLMIN:]

    ll = like_obj.loglike(ls, Dltt, Dlte, Dlee, ELLMIN)
    #combine likelihood and priors
    return lp + ll

def logprior(values, names, prior_bound_dict, prior_gaussian_dict):
    '''
    places uniform priors from datafile
    places Gaussian priors from datafile

    values is an array the length of params (things that are varying)
    names is a list of parameter names in the same order as values
    prior_bound_dict is a dictionary of lists, the keys are the names and the lists are the lower and upper bounds for uniform priors
    prior_gaussian_dict is a dictionary of lists, the keys are the names and the lists are the mean and width for Gaussian priors
    '''
    for i, n in enumerate(names):
        val = values[i]
        low_bound = prior_bound_dict[n][0]
        up_bound = prior_bound_dict[n][1]
        assert low_bound < up_bound
        if not low_bound <= val <= up_bound:
            return -np.inf

    # this part will only run if all parameters are within the ranges allowed by the priors file
    log_prior_val=0
    for name in prior_gaussian_dict:
        val = values[names.index(name)]
        mu = prior_gaussian_dict[name][0]
        sigma = prior_gaussian_dict[name][1]
        log_prior_val += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(val-mu)**2/sigma**2

    return log_prior_val
