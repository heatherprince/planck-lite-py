# plik-lite-py

A Python implementation of *Planck*'s plik-lite code (see the [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html)) with the option of including two approximately Gaussian low-l temperature bins to replace the temperature low-l likelihood. Currently just 2015 data, stay tuned for 2018.

# required packages

In addition to numpy, scipy and matplotlib you will need
* [CLASS](http://class-code.net/) and its [python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper)
* [emcee](https://emcee.readthedocs.io/en/latest/user/install/) if you want to use the sampling code (version 3 up recommended to use hdf5 backend)

# please cite

[*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) because the high ell plik-lite-py code is based on the public *Planck* plik-lite likelihood code and the datafiles belong to *Planck*.

Our paper: arXiv link coming soon


