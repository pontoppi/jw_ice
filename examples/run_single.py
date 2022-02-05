import numpy as np
from jw_ice.calc_ext import OpacityModel

outname = 'opacity.fits'
model = OpacityModel(amin=0.1,amax=1.0,alpha=-3.5,ice_thick=0.05,outname=outname)
