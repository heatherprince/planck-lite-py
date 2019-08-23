from classy import Class
import numpy as np

def get_theoretical_TT_unbinned_power_spec_C_ell(class_params, cosmo, ellmin=2, ellmax=2508, T_cmb=2.7255):
    cosmo.set(class_params)
    cosmo.compute()
    cls = cosmo.lensed_cl(3000)

    #get in units of microkelvin squared
    T_fac=(T_cmb*1e6)**2

    Cltt=(T_fac*cls['tt'])[ellmin:ellmax+1]
    return Cltt

def get_theoretical_TT_TE_EE_unbinned_power_spec_D_ell(class_params, cosmo, ellmin=2, ellmax=2508, T_cmb=2.7255):
    cosmo.set(class_params)
    cosmo.compute()
    cls = cosmo.lensed_cl(3000)

    #get in units of microkelvin squared
    T_fac=(T_cmb*1e6)**2

    ell=cls['ell']
    D_fac=ell*(ell+1.)/(2*np.pi)

    Dltt=(T_fac*D_fac*cls['tt'])[ellmin:ellmax+1]
    Dlte=(T_fac*D_fac*cls['te'])[ellmin:ellmax+1]
    Dlee=(T_fac*D_fac*cls['ee'])[ellmin:ellmax+1]
    return Dltt, Dlte, Dlee
