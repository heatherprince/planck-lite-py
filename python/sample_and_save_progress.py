import matplotlib
matplotlib.use('Agg')
import numpy as np
import emcee
import time
import os
import sys

from plik_cmbonly import plik_lite
import priors
import cmb_power

#CLASS in parallel, emcee in serial
def sample_likelihood(with_low_ell):
    # things that need to be passed to likelihood function
    model=cmb_power.get_theoretical_TT_TE_EE_unbinned_power_spec_D_ell
    class_basic_dict={
            'output': 'tCl,pCl,lCl',
            'l_max_scalars': 3000,
            'lensing': 'yes',
            'N_ur':2.03066666667, #2.046, #1 massive neutrino to match camb
            'N_ncdm': 1,
            'omega_ncdm' : 0.0006451439,
            'non linear' : 'halofit',
            'YHe':0.245341 }
    param_list=['h', 'omega_b', 'omega_cdm', 'tau_reio', 'A_s', 'n_s'] #read in with compression vectors/from separate datafile
    like_obj=plik_lite(with_low_ell)

    # emcee setup
    starting_ind=0
    nwalkers=64
    nsteps=100
    nloops=50
    ndim=len(param_list)
    #nthreads=20

    # directory to save chains
    if with_low_ell:
        save_dir='../chains/LambdaCDM/all_ell/'
    else:
        save_dir='../chains/LambdaCDM/high_ell/'


    #initial walker positions
    if starting_ind==0:
        p0=theta_p_fid=np.array([0.6731, 0.02222, 0.1197, 0.078, np.exp(3.089)/1e10, 0.9655])
        size_ball=theta_p_fid*1e-6
        pos=emcee.utils.sample_ball(p0, size_ball, size=nwalkers)
    else:
        pos=np.loadtxt(save_dir+'pos_after_'+str(starting_ind*nsteps)+'_steps.txt')

    # emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, priors.logprob, args=(model, class_basic_dict, param_list, like_obj))


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    #run emcee
    print('starting mcmc sampling')
    for i in range(starting_ind, starting_ind+nloops):
        start = time.time()
        pos, prob, state=sampler.run_mcmc(pos, nsteps)
        end = time.time()
        print("emcee sampling took ", (end-start), "seconds for ", nsteps, " steps in loop ", i)
        np.savetxt(save_dir+'flatchain_loop_'+str(i)+'.txt', sampler.flatchain)
        np.savetxt(save_dir+'lnprob_loop_'+str(i)+'.txt', sampler.flatlnprobability)
        np.savetxt(save_dir+'pos_after_'+str((i+1)*nsteps)+'_steps.txt', pos)
        sampler.reset()

    # np.savetxt(save_dir+'flatchain_'+str(nwalkers)+'_'+str(nsteps)+'.txt', sampler.flatchain)
    # np.savetxt(save_dir+'lnprob_'+str(nwalkers)+'_'+str(nsteps)+'.txt', sampler.flatlnprobability)
    # np.savetxt(save_dir+'pos_'+str(nwalkers)+'_'+str(nsteps)+'.txt', pos)


def read_sampling_ini_file():
    pass

def str_to_bool(s):
    return s.lower() in ("true", "yes", "t", "1")

if __name__=='__main__':
    with_low_ell=str_to_bool(sys.argv[1])
    sample_likelihood(with_low_ell)
