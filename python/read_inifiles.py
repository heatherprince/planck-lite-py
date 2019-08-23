import numpy as np

def read_sampling_inifile(filename='../inifiles/sample_compressed_LCDM.ini'):
    # read in and split rows that aren't comments
    f = open(filename, 'r+')
    data = [line.strip().split('=') for line in f.readlines()
                if not (line.startswith('#') or line.startswith('\n'))]
    f.close()

    # remove spaces before or after =
    data = [[item.strip() for item in line] for line in data]

    # remove comments at the end of rows
    for row in data:
        for i, val in enumerate(row):
            if '#' in val:
                row=row[:i]
                break

    names = [row[0] for row in data]
    vals =  [row[1] for row in data]

    dict={}
    for i, n in enumerate(names):
        dict[n]=vals[i]
    # deal with which are floats/ints when initialising sampling variables
    return dict

def read_values(filename='../inifiles/values_LCDM.ini'):
    # read in and split rows that aren't comments
    f = open(filename, 'r+')
    data = [line.strip().split() for line in f.readlines()
                if not (line.startswith('#') or line.startswith('\n'))]
    f.close()

    # remove comments at the end of rows
    for row in data:
        for i, val in enumerate(row):
            if '#' in val:
                row=row[:i]
                break

    # read all
    param_names = np.array([row[0] for row in data])
    param_lower = np.array([float(row[1]) if len(row)>3 else float('nan') for row in data])
    param_upper = np.array([float(row[3]) if len(row)>3 else float('nan') for row in data])
    param_values = np.array([float(row[1]) if len(row)<3 else float(row[2]) for row in data])

    # divide into categories:
    # dictionary with fiducial values for params we aren't sampling over
    param_dict_fiducial={}
    for i, n in enumerate(param_names):
        if np.isnan(param_lower[i]):
            param_dict_fiducial[n]=param_values[i]

    # dictionary with starting values for params we are sampling over
    param_dict_sample_start={}
    for i, n in enumerate(param_names):
        if np.isfinite(param_lower[i]):
            param_dict_sample_start[n]=param_values[i]

    # dictionary with upper and lower bounds for priors
    param_dict_prior_bounds={}
    for i, n in enumerate(param_names):
        if np.isfinite(param_lower[i]):
            param_dict_prior_bounds[n]=[param_lower[i], param_upper[i]]


    return param_dict_fiducial, param_dict_sample_start, param_dict_prior_bounds

def read_gaussian_priors(filename='../inifiles/tau_prior.ini'):
    f = open(filename, 'r+')
    data = [line.strip().split() for line in f.readlines()
                if not (line.startswith('#') or line.startswith('\n'))]
    f.close()

    # remove comments at the end of rows
    for row in data:
        for i, val in enumerate(row):
            if '#' in val:
                row=row[:i]
                break

    param_names = np.array([row[0] for row in data])
    mean = np.array([float(row[1]) for row in data])
    sigma = np.array([float(row[2]) for row in data])

    param_dict_gaussian_priors={}
    for i, n in enumerate(param_names):
        param_dict_gaussian_priors[n]=[mean[i], sigma[i]]

    return param_dict_gaussian_priors

if __name__=='__main__':
    read_sampling_inifile()
    # fix float rounding errors?
    fid, start, bounds = read_values()
    print(fid,'\n\n',start, '\n\n', bounds, '\n\n')
    gauss = read_gaussian_priors()
    print(gauss)
