import numpy as np
from planck_lite_py import PlanckLitePy

# create a PlanckLitePy object
TTTEEE2018_lowTTbins=PlanckLitePy(data_directory='data', year=2018, spectra='TTTEEE', use_low_ell_bins=True)

# read some spectra to pass to the likelihood (can use CAMB/CLASS to generate these)
ls, Dltt, Dlte, Dlee = np.genfromtxt('data/Dl.dat', unpack=True)
ellmin=int(ls[0])

# call the likelihood function
loglike=TTTEEE2018_lowTTbins.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
print('Planck-lite-py likelihood (2018 high-l TT, TE, EE + low-l TT bins):', loglike)

# suppose we only want the high-ell temperature
TT2015=PlanckLitePy(data_directory='data', year=2015, spectra='TT', use_low_ell_bins=False)
loglike=TT2015.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
print('Planck-lite-py likelihood (2015 high-l TT):', loglike)
