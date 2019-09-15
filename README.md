# Planck-lite-py

A Python implementation of *Planck*'s plik-lite code (see the [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) or the [arXiv version](https://arxiv.org/abs/1507.02704), and the [*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875)) with the option of including two approximately Gaussian low-l temperature bins to replace the temperature low-l likelihood. Includes 2015 data and 2018 data as options. 

Note, a Python implementation of the plik-lite part also exists in Cobaya ('planck_2018_highl_plik.[TT|TTTEEE]_lite_native'), see https://cobaya.readthedocs.io/en/latest/likelihood_planck.html

# required packages
* numpy
* scipy 

# usage

```python
# import the PlanckLitePy class
from planck_lite_py import PlanckLitePy

# create a PlanckLitePy object
TT2018 = PlanckLitePy(data_directory='data', year=2018, spectra='TT', use_low_ell_bins=False)

# call the log likelihood function with TT, TE and EE spectra
loglike=TT2018.loglike(Dltt, Dlte, Dlee, ellmin) 
```

When initializing the PlanckLitePy object you can specify:
* path: from where you are running the code to the data directories
* year: 2015 or 2018 to use the *Planck* 2015 or 2018 data releases
* spectra: 'TT' for just temperature, or 'TTTEEE' for TT, TE and EE spectra
* use_low_ell: True to use two low-l temperature bins, False to use just l>=30 data

Notes on the PlanckLitePy log likelihood function:
* the log likelihood function expects the spectra in the form D=l(l+1)/2&pi; C 
* Dltt, Dlte and Dlee should all cover the same l range, usually from a minimum l value of 0 or 2
* ellmin=2 by default; if your spectra start at l=0 then specify this with ellmin=0

# please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because the high ell Planck-lite-py code is based on the public *Planck* plik-lite likelihood code and the datafiles belong to *Planck*.

Our paper: arXiv link coming soon


