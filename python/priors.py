import numpy as np
from plik_cmbonly import plik_lite

def logprob(theta_p, model, class_param_dict, params, like_obj):
    '''
    theta_p - an array the length of params (things that are varying)
    model - a function that takes a parameter dictionary and returns the power spectrum
    params - a list of parameter names in the same order as theta_p
    class_param_dict - a dictionary with all of the necessary CLASS inputs.
        - we modify just the parameters we are sampling over
    like_obj - plik-lite object
    '''
    ELLMIN=1

    #priors first to make sure tau is in a sensible range and avoid errors in CLASS
    lp = logprior(theta_p, params)
    if not np.isfinite(lp):
        return -np.inf

    #get cl from CLASS
    thetas=class_param_dict.copy()
    for i, p in enumerate(params):
        thetas[p]=theta_p[i]
    print('dictionary for CLASS', thetas)
    Dltt, Dlee, Dlte=model(thetas, ELLMIN)
    ls = np.arange(Dltt.shape[0]+ELLMIN)[ELLMIN:]

    #combine likelihood and priors
    return lp + like_obj.loglike(ls, Dltt, Dlte, Dlee, ELLMIN)


# include some way to read priors from sampling ini file
def logprior(theta_p, params):
    '''
    places uniform priors on all LCDM parameters except tau_reio
    places Gaussian prior on tau
    theta_p is an array the length of params (things that are varying)
    params is a list of parameter names in the same order as theta_p
    '''
    #do this more neatly
    try:
        h=theta_p[params.index('h')]
    except ValueError:
        h=0.7   #in prior range

    try:
        omega_b=theta_p[params.index('omega_b')]
    except ValueError:
        omega_b=0.02   #in prior range

    try:
        omega_cdm=theta_p[params.index('omega_cdm')]
    except ValueError:
        omega_cdm=0.2

    try:
        tau_reio=theta_p[params.index('tau_reio')]
    except ValueError:
        tau_reio=0.08

    try:
        A_s=theta_p[params.index('A_s')]
    except ValueError:
        A_s=1e-9

    try:
        n_s=theta_p[params.index('n_s')]
    except:
        n_s=0.96

    try:
        N_ur=theta_p[params.index('N_ur')]
    except:
        N_ur=2.046

    #flat priors on all except tau, impose reasonable tau bounds to avoid errors when calling CLASS
    if not(0.60<=h<=0.80 and 0.001<=omega_b<=0.05 and 0.001<=omega_cdm<=0.5 and 1e-10<=A_s<=1e-8 and 0.9<=n_s<=1 and 0.01<tau_reio<0.17 and 1<N_ur<4):
        return -np.inf

    #gaussian prior on tau
    mu=0.064  #0.053  #0.078
    sigma=0.022  #0.019  #0.019
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(tau_reio-mu)**2/sigma**2

    #Gaussian prior on 10^9 A exp(-2tau)
    # mu=1.865
    # sigma=0.019
    # x=1e9*A_s*np.exp(-2*tau_reio)
    # return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(x-mu)**2/sigma**2
