# Planck-lite-py

A Python implementation of *Planck*'s plik-lite code (see the [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) or the [arXiv version](https://arxiv.org/abs/1507.02704)) with the option of including two approximately Gaussian low-l temperature bins to replace the temperature low-l likelihood. Includes 2015 data and 2018 data as options. 

Note, a Python implementation of the plik-lite part also exists in Cobaya ('planck_2018_highl_plik.[TT|TTTEEE]_lite_native'), see https://cobaya.readthedocs.io/en/latest/likelihood_planck.html

# required packages
* numpy
* scipy 

# usage

see planck_lite_py_example.py

# please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because the high ell Planck-lite-py code is based on the public *Planck* plik-lite likelihood code and the datafiles belong to *Planck*.

Our paper: arXiv link coming soon


