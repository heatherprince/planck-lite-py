#python version of plik-lite to check binning etc before including in cosmoped
#Fortran to python slicing: a:b becomes a-1:b
#used some stuff from Zack's code for the ACT likelihood (esp for reading covmat): https://github.com/xzackli/actpols2_like_py/blob/master/act_like.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import scipy.linalg

import cmb_power

SHOW_PLOTS=False

#make it a class so that objects can be initialised once easily?
class plik_lite:
    def __init__(self, year=2015, spectra='TT', use_low_ell_bins=False):
        self.use_low_ell_bins=use_low_ell_bins #False matches Plik_lite - just l<=30
        if self.use_low_ell_bins:
            self.nbintt_low_ell=2
            self.plmin=2

        else:
            self.nbintt_low_ell=0
            self.plmin=30
        self.plmax=2508
        self.calPlanck=1

        if year==2015:
            self.data_dir='../cmb_data/planck2015_plik_lite/'
            version=18
        elif year==2018:
            self.data_dir='../cmb_data/planck2018_plik_lite/'
            version=22

        if spectra=='TT':
            self.use_tt=True
            self.use_ee=False
            self.use_te=False
        elif spectra=='TTTEEE':
            self.use_tt=True
            self.use_ee=True
            self.use_te=True

        self.nbintt_hi = 215 #30-2508   #used when getting covariance matrix
        self.nbinte = 199 #30-1996
        self.nbinee = 199 #30-1996
        self.nbin_hi=self.nbintt_hi+self.nbinte+self.nbinee

        self.nbintt=self.nbintt_hi+self.nbintt_low_ell #mostly want this if using low ell
        self.nbin_tot=self.nbintt+self.nbinte+self.nbinee

        self.like_file = self.data_dir+'cl_cmb_plik_v'+str(version)+'.dat'
        self.cov_file  = self.data_dir+'c_matrix_plik_v'+str(version)+'.dat'
        self.blmin_file = self.data_dir+'blmin.dat'
        self.blmax_file = self.data_dir+'blmax.dat'
        self.binw_file = self.data_dir+'bweight.dat'

        # read in binned ell value, C(l) TT, TE and EE and errors
        # use_tt etc to slice? when should this happen?
        self.bval, self.X_data, self.X_sig=np.genfromtxt(self.like_file, unpack=True)
        self.blmin=np.loadtxt(self.blmin_file).astype(int)
        self.blmax=np.loadtxt(self.blmax_file).astype(int)
        self.bin_w=np.loadtxt(self.binw_file)

        if self.use_low_ell_bins:
            self.data_dir_low_ell='../cmb_data/planck'+str(year)+'_low_ell/'
            self.bval_low_ell, self.X_data_low_ell, self.X_sig_low_ell=np.genfromtxt(self.data_dir_low_ell+'CTT_bin_low_ell_'+str(year)+'.dat', unpack=True)
            self.blmin_low_ell=np.loadtxt(self.data_dir_low_ell+'blmin_low_ell.dat').astype(int)
            self.blmax_low_ell=np.loadtxt(self.data_dir_low_ell+'blmax_low_ell.dat').astype(int)
            self.bin_w_low_ell=np.loadtxt(self.data_dir_low_ell+'bweight_low_ell.dat')

            self.bval=np.concatenate((self.bval_low_ell, self.bval))
            self.X_data=np.concatenate((self.X_data_low_ell, self.X_data))
            self.X_sig=np.concatenate((self.X_sig_low_ell, self.X_sig))

            self.blmin=np.concatenate((self.blmin_low_ell, self.blmin+len(self.bin_w_low_ell)))
            self.blmax=np.concatenate((self.blmax_low_ell, self.blmax+len(self.bin_w_low_ell)))
            self.bin_w=np.concatenate((self.bin_w_low_ell, self.bin_w))


        self.fisher=self.get_inverse_covmat()

        if SHOW_PLOTS:
            plt.plot(blmin[1:]-blmin[:-1], 'o', linestyle='none')
            plt.show()

    def get_inverse_covmat(self):
        #read full covmat
        f = FortranFile(self.cov_file, 'r')
        covmat = f.read_reals(dtype=float).reshape((self.nbin_hi,self.nbin_hi))
        for i in range(self.nbin_hi):
            for j in range(i,self.nbin_hi):
                covmat[i,j] = covmat[j,i]

        #select relevant covmat
        if self.use_tt and not(self.use_ee) and not(self.use_te):
            #just tt
            bin_no=self.nbintt_hi
            start=0
            end=start+bin_no
            cov=covmat[start:end, start:end]
        elif not(self.use_tt) and not(self.use_ee) and self.use_te:
            #just te
            bin_no=self.nbinte
            start=self.nbintt_hi
            end=start+bin_no
            cov=covmat[start:end, start:end]
        elif not(self.use_tt) and self.use_ee and not(self.use_te):
            #just ee
            bin_no=self.nbinee
            start=self.nbintt_hi+self.nbinte
            end=start+bin_no
            cov=covmat[start:end, start:end]
        elif self.use_tt and self.use_ee and self.use_te:
            #use all
            bin_no=self.nbin_hi
            cov=covmat
        else:
            print("not implemented")

        if self.use_low_ell_bins:
            print('Including low ell TT in covariance matrix')
            bin_no += self.nbintt_low_ell

            covmat_with_lo=np.zeros(shape=(bin_no, bin_no))
            cov_lo=np.diag(self.X_sig_low_ell**2)
            covmat_with_lo[0:self.nbintt_low_ell, 0:self.nbintt_low_ell]=cov_lo
            covmat_with_lo[self.nbintt_low_ell:, self.nbintt_low_ell:]=cov

            cov=covmat_with_lo

        #invert covariance matrix (cholesky decomposition should be faster)
        fisher=scipy.linalg.cho_solve(scipy.linalg.cho_factor(cov), np.identity(bin_no))
        #zack transposes it (because fortran indexing works differently?) but I don't think this should make a difference? check!
        fisher=fisher.transpose()
        if SHOW_PLOTS:
            fisher2=np.linalg.inv(cov)
            plt.subplot(131)
            plt.pcolormesh(fisher)
            plt.colorbar()
            plt.subplot(132)
            plt.pcolormesh(fisher2)
            plt.colorbar()
            plt.subplot(133)
            plt.pcolormesh(fisher-fisher2)
            plt.colorbar()
            plt.show()

            plt.subplot(131)
            plt.pcolormesh(fisher)
            plt.colorbar()
            plt.subplot(132)
            plt.pcolormesh(fisher.transpose())
            plt.colorbar()
            plt.subplot(133)
            plt.pcolormesh(fisher-fisher.transpose())
            plt.colorbar()
            plt.show()

        return fisher

    def loglike(self, ls, Dltt, Dlte, Dlee, ellmin=1):
        #l should start at 1 for consistency with plik_lite??
        #convert model Dl's to Cls then bin them
        fac=ls*(ls+1)/(2*np.pi)
        Cltt=Dltt/fac
        Clte=Dlte/fac
        Clee=Dlee/fac


        #indexing here is a bit odd. need to subtract 1 to use 0 indexing for cl, then add one for weights because fortran includes top value
        #how does it work in fortran when i=1 and blmin(i)=0?
        Cltt_bin=np.zeros(self.nbintt)
        #import IPython; IPython.embed()
        for i in range(self.nbintt):
            #if i==0:
            Cltt_bin[i]=np.sum(Cltt[self.blmin[i]+self.plmin-ellmin:self.blmax[i]+self.plmin+1-ellmin]*self.bin_w[self.blmin[i]:self.blmax[i]+1]) #what happens in Fortran when blmin is 0?
            #else:
            #    Cltt_bin[i]=np.sum(Cltt[self.blmin[i]+self.plmin-1:self.blmax[i]+self.plmin]*self.bin_w[self.blmin[i]-1:self.blmax[i]]) #testing!

        #shouldn't I be using a different part of blmin, blmax and bin_w??
        Clte_bin=np.zeros(self.nbinte)
        for i in range(self.nbinte):
            Clte_bin[i]=np.sum(Clte[self.blmin[i]+self.plmin-1:self.blmax[i]+self.plmin]*self.bin_w[self.blmin[i]:self.blmax[i]+1])

        #shouldn't I be using a different part of blmin, blmax and bin_w??
        Clee_bin=np.zeros(self.nbinee)
        for i in range(self.nbinee):
            Clee_bin[i]=np.sum(Clee[self.blmin[i]+self.plmin-1:self.blmax[i]+self.plmin]*self.bin_w[self.blmin[i]:self.blmax[i]+1])

        X_model=np.zeros(self.nbin_tot)
        X_model[:self.nbintt]=Cltt_bin/self.calPlanck**2
        X_model[self.nbintt:self.nbintt+self.nbinte]=Clte_bin/self.calPlanck**2
        X_model[self.nbintt+self.nbinte:]=Clee_bin/self.calPlanck**2

        Y=self.X_data-X_model

        #choose relevant bits based on whether using TT, TE, EE
        if self.use_tt and not(self.use_ee) and not(self.use_te):
            #just tt
            bin_no=self.nbintt
            start=0
            end=start+bin_no
            diff_vec=Y[start:end]
        elif not(self.use_tt) and not(self.use_ee) and self.use_te:
            #just te
            bin_no=self.nbinte
            start=self.nbintt
            end=start+bin_no
            diff_vec=Y[start:end]
        elif not(self.use_tt) and self.use_ee and not(self.use_te):
            #just ee
            bin_no=self.nbinee
            start=self.nbintt+self.nbinte
            end=start+bin_no
            diff_vec=Y[start:end]
        elif self.use_tt and self.use_ee and self.use_te:
            #use all
            bin_no=self.nbin_tot
            diff_vec=Y
        else:
            print("not implemented")

        #why is -lnlike returned in plik_lite? --> WMAP convention
        return -0.5*diff_vec.dot(self.fisher.dot(diff_vec))







if __name__=='__main__':
    ELLMIN=1
    class_dict={
            'output': 'tCl,pCl,lCl',
            'l_max_scalars': 3000,
            'lensing': 'yes',
            'N_ur':2.03066666667, #2.0328 #1 massive neutrino to match camb
            'N_ncdm': 1,
            'omega_ncdm' : 0.0006451439,
            # 'm_ncdm': 0.06,
            'non linear' : 'halofit',
            'YHe':0.245341 }
    logA=3.089
    A=np.exp(logA)/1e10
    theta={"h":0.6731, "omega_b":0.02222, "omega_cdm":0.1197, "tau_reio":0.078, "A_s":A, "n_s":0.9655}
    class_dict.update(theta)
    Dltt, Dlte, Dlee=cmb_power.get_theoretical_TT_TE_EE_unbinned_power_spec_D_ell(class_dict, ELLMIN)
    ls = np.arange(Dltt.shape[0]+ELLMIN)[ELLMIN:]

    # compare high ell TT 2015 vs 2018
    like_obj=plik_lite(year=2015)
    print ('plik-lite-py with CMB spectra from CLASS, planck 2015 high ell temperature data: ', -1*like_obj.loglike(ls, Dltt, Dlte, Dlee))

    like_obj2018=plik_lite(year=2018)
    print ('plik-lite-py with CMB spectra from CLASS, planck 2018 high ell temperature data: ', -1*like_obj2018.loglike(ls, Dltt, Dlte, Dlee))

    like_obj2018_temp_and_pol=plik_lite(year=2018, spectra='TTTEEE')
    print ('plik-lite-py with CMB spectra from CLASS, planck 2018 high ell temperature and polarization data: ', -1*like_obj2018_temp_and_pol.loglike(ls, Dltt, Dlte, Dlee))

    like_obj=plik_lite(use_low_ell_bins=True)
    print ('plik-lite-py with CMB spectra from CLASS: ', -1*like_obj.loglike(ls, Dltt, Dlte, Dlee))
