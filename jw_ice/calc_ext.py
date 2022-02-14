import os
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import simps as simpson
import json
from astropy.io import ascii,fits
from astropy.table import Table
import PyMieScatt as pms


class OpacityModel():
    def __init__(self,amin=None,amax=None,alpha=None,ice_thick=None,carbon_frac=None,outname=None):
        
        self.rootpath = os.path.abspath(os.path.dirname(__file__))
        
        self.read_setup()
        
        #override default config with passed parameter
        if amin:
            self.config['dust']['amin'] = amin
        if amax:
            self.config['dust']['amax'] = amax
        if alpha:
            self.config['dust']['dist']['alpha'] = alpha
        if ice_thick:
            self.config['dust']['ice_thick'] = ice_thick
            self.ice_thick = ice_thick
        if carbon_frac:
            self.config['properties']['carbon_frac'] = carbon_frac
        if outname:
            self.config['io']['outname'] = outname

        self.read_ocs()
        self.create_grids()
        self.calc_distribution(self.distname)
        self.run_model()
        self.write_opac()
        #self.plot_cext()

    def run_model(self):

        volumes = (4.0*np.pi/3.0)*(self.acores*1e-4/2)**3.0
        
        for iw,wave in enumerate(self.waves):
            for ia,acore in enumerate(self.acores):

                out = pms.MieQCoreShell(self.core_index_int[iw], self.mant_index_int[iw], wave*1e3, acore*1e3, (acore+self.ice_thick)*1e3, asDict=True)

                self.qexts[ia,iw] = out['Qext']
                self.qscas[ia,iw] = out['Qsca']
                self.gscas[ia,iw] = out['g']
        
                '''
                Converting from Q (dimensionless quantity measured relative to the
                geometric cross section of a grain) to cross section per gram [cm^2/g]
                                    Q*(pi*(dia/2)^2) 
                cross =         -----------------------      =   Q*(3/2)/dens_av/dia
                                (4*pi/3)*(dia/2)^3*dens_av
                '''
                
                #normalizing to _refractory_ component to preserve gas-to-dust ratio during freeze-out
                self.sigma_exts[ia,iw] = self.qexts[ia,iw]*(1.5/self.dens_core)/(acore*1e-4) 
                self.sigma_scas[ia,iw] = self.qscas[ia,iw]*(1.5/self.dens_core)/(acore*1e-4) 
 
                self.int_exts[ia,iw] = self.sigma_exts[ia,iw]*self.dens_core*self.fdist[ia]*volumes[ia]
                self.int_scas[ia,iw] = self.sigma_scas[ia,iw]*self.dens_core*self.fdist[ia]*volumes[ia]
                self.int_gsca[ia,iw] = self.gscas[ia,iw]*self.sigma_scas[ia,iw]*self.dens_core*self.fdist[ia]*volumes[ia]
               
            #logarithmically integrate over distribution       
            self.sigma_exts_tot[iw] = simpson(self.int_exts[:,iw]*self.acores,x=self.lnacores)
            self.sigma_scas_tot[iw] = simpson(self.int_scas[:,iw]*self.acores,x=self.lnacores)
            self.gsca_tot[iw]       = simpson(self.int_gsca[:,iw]*self.acores,x=self.lnacores)
            
        #Weighing g-factor with scattering cross section
        self.gsca_tot /= self.sigma_scas_tot
        #and normalize to total core mass
        self.calc_icerock()
        self.sigma_exts_tot /= self.ma_tot
        self.sigma_scas_tot /= self.ma_tot
                    
    def calc_icerock(self):
        
        volumes = (4.0*np.pi/3.0)*(self.acores*1e-4/2)**3.0
        
        for ia,acore in enumerate(self.acores):
            #Ma = mass in one bin in the grain mass distribution
            self.mas[ia]     = self.dens_core*self.fdist[ia]*volumes[ia]
            self.mas_ice[ia] = self.dens_ice*self.fdist[ia]*(4/3)*np.pi*(((acore+self.ice_thick)*1e-4/2)**3 - (acore*1e-4/2)**3)
             
                
                #integrate over distribution
        self.ma_tot    = simpson(self.mas*self.acores,x=self.lnacores)
        self.maice_tot = simpson(self.mas_ice*self.acores,x=self.lnacores)
        #print('ice/rock mass ratio:',self.maice_tot/self.ma_tot)
        
                
    def read_setup(self):

        with open(os.path.join(self.rootpath,'config.json'), 'r') as file:
            self.config = json.load(file)
        
        self.ocpath    = os.path.join(self.rootpath,self.config['ocs']['ocpath'])
        self.nwave     = self.config['spectrum']['nwave']
        self.nsize     = self.config['dust']['nsize']
        self.distname  = self.config['dust']['dist']['name']
        self.dens_core = self.config['properties']['silicate_dens']
        self.dens_ice  = self.config['properties']['ice_dens']
        self.ice_thick = self.config['dust']['ice_thick']
        self.carbon_frac = self.config['properties']['carbon_frac']
    
    def read_ocs(self):
        print(self.ocpath+self.config['ocs']['core_oc_sil'][0])
        self.core_oc_sil = ascii.read(self.ocpath+self.config['ocs']['core_oc_sil'][0])
        self.core_oc_car = ascii.read(self.ocpath+self.config['ocs']['core_oc_car'][0])
        core_index_sil = self.core_oc_sil['col2']+self.core_oc_sil['col3']*1j
        core_index_car = self.core_oc_car['col2']+self.core_oc_car['col3']*1j

        core_wave_sil  = self.core_oc_sil['col1']*1e4
        core_wave_car  = self.core_oc_car['col1']
        # The interpolation requires that the wavelengths are monotonically increasing
        ssubs = np.argsort(core_wave_sil)
        core_wave_sil = core_wave_sil[ssubs]
        core_index_sil = core_index_sil[ssubs]

        ssubs = np.argsort(core_wave_car)
        core_wave_car = core_wave_car[ssubs]
        core_index_car = core_index_car[ssubs]

        # and interpolating to the silicate grid
        core_index_car = np.interp(core_wave_sil, core_wave_car, core_index_car)
        # Calculating effective medium optical constants for spherical carbon inclusions in the silicate.
        if self.carbon_frac>0:
            f = self.carbon_frac
            e_mat =  core_index_sil**2
            e_inc =  core_index_car**2
            e_eff = e_mat*(1+3*f*((e_inc-e_mat)/(e_inc+2*e_mat))/(1-f*((e_inc-e_mat)/(e_inc+2*e_mat))))
            self.core_index = np.sqrt(e_eff)
        else:
            self.core_index = core_index_sil
        
        self.core_wave = core_wave_sil

        # Then handle the mantle refractive index
        self.mant_oc = ascii.read(self.ocpath+self.config['ocs']['mantle_oc'][0])
        self.mant_wave  = 1e4/self.mant_oc['col1']
        self.mant_index = self.mant_oc['col2']+self.mant_oc['col3']*1j

        # The interpolation requires that the wavelengths are monotonically increasing
        ssubs = np.argsort(self.mant_wave)
        self.mant_wave = self.mant_wave[ssubs]
        self.mant_index = self.mant_index[ssubs]

    def create_grids(self):
        
        amax = self.config['dust']['amax']
        amin = self.config['dust']['amin']
        wmax = self.config['spectrum']['wmax']
        wmin = self.config['spectrum']['wmin']
        
        self.waves = np.logspace(np.log10(wmin),np.log10(wmax),self.nwave)
        self.lnacores    = np.arange(self.nsize)/(self.nsize-1)*(np.log(amax)-np.log(amin))+np.log(amin)
        self.acores = np.exp(self.lnacores)

        self.core_index_int = np.interp(self.waves, self.core_wave,self.core_index)
        self.mant_index_int = np.interp(self.waves, self.mant_wave,self.mant_index)
        
        self.qexts          = np.zeros((self.nsize,self.nwave))
        self.qscas          = np.zeros((self.nsize,self.nwave))
        self.sigma_exts     = np.zeros((self.nsize,self.nwave))
        self.sigma_scas     = np.zeros((self.nsize,self.nwave))
        self.gscas          = np.zeros((self.nsize,self.nwave))
        self.int_exts       = np.zeros((self.nsize,self.nwave))
        self.int_scas       = np.zeros((self.nsize,self.nwave))
        self.int_gsca       = np.zeros((self.nsize,self.nwave))

        self.sigma_exts_tot = np.zeros(self.nwave)
        self.sigma_scas_tot = np.zeros(self.nwave)
        self.gsca_tot       = np.zeros(self.nwave)
        
        self.mas     = np.zeros(self.nsize)
        self.mas_ice = np.zeros(self.nsize)
        
    def calc_distribution(self,distname):
        if distname == 'wd':
            pass
            #make_dist_wd,RCores,fdist, alpha_s=alpha, ats=ats,bc=bc,acg=acg,beta_s=beta_s
        elif distname == 'power':
            self.fdist = self.make_dist_power(self.acores, alpha=self.config['dust']['dist']['alpha'])
        
    def make_dist_power(self, a, alpha=-3.5):
        const = 1.
        fdist = const*a**alpha
        return fdist
        
    def plot_cext(self):
        plt.plot(self.waves, self.sigma_exts_tot)
        plt.show()
        
    def write_opac(self):
        outname =  self.config['io']['outname'] 
        
        c1 = fits.Column(name='wavelength',format='E', array=self.waves)
        c2 = fits.Column(name='cext',format='E', array=self.sigma_exts_tot)
        c3 = fits.Column(name='csca',format='E', array=self.sigma_scas_tot)
        c4 = fits.Column(name='g',format='E', array=self.gsca_tot)
        
        cols = fits.ColDefs([c1,c2,c3,c4])
        
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['amin'] = self.config['dust']['amin']
        hdu.header['amax'] = self.config['dust']['amax']
        hdu.header['alpha'] = self.config['dust']['dist']['alpha']
        hdu.header['icethick'] = self.config['dust']['ice_thick']
        hdu.header['ratio'] = self.maice_tot/self.ma_tot

        hdu.writeto(outname, overwrite=True)


