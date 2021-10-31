import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from astropy import units as u
from astropy.modeling.models import BlackBody
from lmfit import Parameters, minimize

'''
Class to fit continua for ice spectra and derive optical depths

'''

class IceCont():
    def __init__(self,datafile,opacfile):
        self.read_data(datafile)
        self.read_opac(opacfile)
        self.fit_poly()
        
    def read_data(self,datafile):
        data_raw = fits.getdata(datafile)
        self.wave = data_raw['WAVELENGTH'].flatten()
        self.fd = data_raw['FLUX'].flatten()
    
    def read_opac(self,opacfile):
        data_raw = fits.getdata(opacfile)
        wave_opac = data_raw['wavelength']
        cext      = data_raw['cext']
        
        ssubs = np.argsort(wave_opac)
        wave_opac = wave_opac[ssubs]
        cext = cext[ssubs]
        self.cext_int  = np.interp(self.wave,wave_opac,cext)
        
        
    def fit_poly(self):
        ranges = [(2.85,3.4),(3.6,4.0),(4.1,4.2)]
        fit_params = Parameters()
        fit_params.add('c0',value=1.0)
        fit_params.add('c1',value=1.0)
        fit_params.add('c2',value=0.0)
        fit_params.add('c3',value=0.0)
        fit_params.add('nd',value=1e-3)
        
        gsubs = []
        for range in ranges:
            gsubs = gsubs + np.where((self.wave>range[0]) & (self.wave<range[1]))[0].tolist()
        
        #self.cext_int -= np.min(self.cext_int[gsubs])
        self.cext_subs = self.cext_int[gsubs]
        
        out = minimize(self.residual, fit_params, args=(self.wave[gsubs],), kws={'data':self.fd[gsubs]})
        fit = self.residual(out.params, self.wave)
        cont_params = out.params
        cont_params['nd'].value = 0.0
        cont = self.residual(cont_params, self.wave)
        plt.plot(self.wave,self.fd)
        plt.plot(self.wave,fit)
        plt.plot(self.wave,cont)
        
        plt.show()
        
    def residual(self,pars,x,data=None,cext=None):
        cont = pars['c0'] + pars['c1']*x + pars['c2']*x**2 + pars['c3']*x**3
        if data is None:
            model = cont*np.exp(-self.cext_int*pars['nd'])
            return model
        else:
            model = cont*np.exp(-self.cext_subs*pars['nd'])
        return model-data

    def fit_bb(self):
        ranges = [(2.85,3.2),(3.65,4.0)]
        fit_params = Parameters()
        fit_params.add('const',value=1e7)
        fit_params.add('const_star',value=1e7)        
        fit_params.add('temp',value=1000,min=100,max=4000)
        fit_params.add('nd',value=1e-3,max=3e-3,min=0)
        
        gsubs = []
        for range in ranges:
            gsubs = gsubs + np.where((self.wave>range[0]) & (self.wave<range[1]))[0].tolist()
        
        #self.cext_int -= np.min(self.cext_int[gsubs])
        self.cext_subs = self.cext_int[gsubs]
        
        out = minimize(self.residual_bb, fit_params, args=(self.wave[gsubs],), kws={'data':self.fd[gsubs]})
        fit = self.residual_bb(out.params, self.wave)
        cont_params = out.params
        cont_params['nd'].value = 0.0
        cont = self.residual_bb(cont_params, self.wave)
        plt.plot(self.wave,self.fd)
        plt.plot(self.wave,fit)
        plt.plot(self.wave,cont)
        
        plt.show()
        
    def residual_bb(self,pars,x,data=None,cext=None):
        bb = BlackBody(temperature=pars['temp']*u.K)
        cont = pars['const']*bb(x*u.micron).value
        bb = BlackBody(temperature=4000*u.K)
        star = pars['const_star']*bb(x*u.micron).value
        if data is None:
            model = (star+cont)*np.exp(-self.cext_int*pars['nd'])
            return model
        else:
            model = (star+cont)*np.exp(-self.cext_subs*pars['nd'])
        return model-data
        
                
path = '/astro/pontoppi/DATA/ESO/ISAAC_ARCHIVE_FITS/'
ice_cont = IceCont(path+'RNO91_L.fits','opacity.fits')
#ice_cont = IceCont(path+'RE50_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRS43_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRS44_L.fits','opacity.fits')
#ice_cont = IceCont(path+'CRBR2422_L.fits','opacity.fits')
#ice_cont = IceCont(path+'Elias32_L.fits','opacity.fits')
#ice_cont = IceCont(path+'L1489_L.fits','opacity.fits')
#ice_cont = IceCont(path+'WL12_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRS42_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRS46_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRS51_L.fits','opacity.fits')
#ice_cont = IceCont(path+'IRAS08448_L.fits','opacity.fits')
