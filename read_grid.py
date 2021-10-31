from glob import glob
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import simpson
from lmfit import Parameters, minimize
from astropy.io import fits


def fit_poly(wave,cext):
    ranges = [(2.5,2.7),(3.7,4.1)]
    fit_params = Parameters()
    fit_params.add('c0',value=1.0)
    fit_params.add('c1',value=1.0)
    fit_params.add('c2',value=0.0)
    fit_params.add('c3',value=0.0, min=0,max=1e-10)
        
    gsubs = []
    for range in ranges:
        gsubs = gsubs + np.where((wave>range[0]) & (wave<range[1]))[0].tolist()
        
    cext_subs = cext[gsubs]
        
    out = minimize(residual, fit_params, args=(wave[gsubs],),kws={'data':cext_subs})
    fit = residual(out.params, wave)
    '''
    plt.plot(wave,fit)    
    plt.plot(wave,cext)    
    plt.show()
    '''
    return fit
    
    
    
    
    
def residual(pars,x,data=None):
    cont = pars['c0'] + pars['c1']*x + pars['c2']*x**2 + pars['c3']*x**3
    if data is None:
        return cont
    else:
        return cont-data


path = 'opacities/'
h2o_range = (2.8,3.8)
cont_ranges = [(2.5,2.7),(3.7,4.0)]

allfiles = glob(path+"*.*")
nfiles = len(allfiles)

ratios = np.zeros(nfiles)
strengths = np.zeros(nfiles)
amaxs = np.zeros(nfiles)
alphas = np.zeros(nfiles)

for ii,file in enumerate(allfiles):
    opac,hdr = fits.getdata(file,1, header=True)
    wave = opac['wavelength']
    cext = opac['cext']
    ratios[ii] = hdr['ratio']
    amaxs[ii]  = hdr['amax']
    alphas[ii] = hdr['alpha']

    cont = fit_poly(wave,cext)
    isubs = np.where((wave>h2o_range[0]) & (wave<h2o_range[1]))
    strengths[ii] = simpson(cext[isubs]-cont[isubs],x=wave[isubs]) 
    print(ii)

plt.scatter(amaxs,strengths/ratios,s=np.abs(alphas)**3)
plt.xscale('log')
plt.yscale('log')
plt.show()
    
#    plt.plot(opac['wavelength'],opac['cext'])

#plt.show()
breakpoint()