import numpy as np
import matplotlib.pylab as plt
import itertools
from multiprocessing import Pool
from calc_ext import OpacityModel

counter = 0
nworkers = 1

def run_model(pars_and_id):
    id = pars_and_id['id']
    pars = pars_and_id['pars']
    outname = 'opacities/opacity_'+str(id)+'.fits'
    model = OpacityModel(amin=pars[0],amax=pars[1],alpha=pars[2],ice_thick=pars[3],outname=outname)
    print(outname)
    
thick_range = (0.1,1.0,10)
alpha_range = (-2.5,-3.5,10)
amin_range = (0.01,0.01,1)
amax_range = (0.011,10.0,30)
model_index = np.arange(thick_range[2]*alpha_range[2]*amin_range[2]*amax_range[2])

ice_thicks = np.linspace(thick_range[0],thick_range[1],thick_range[2])
alphas = np.linspace(alpha_range[0],alpha_range[1],alpha_range[2])
amins = np.logspace(np.log10(amin_range[0]),np.log10(amin_range[1]),amin_range[2])
amaxs = np.logspace(np.log10(amax_range[0]),np.log10(amax_range[1]),amax_range[2])

all_pars = [amins,amaxs,alphas,ice_thicks]
master_iter = list(itertools.product(*all_pars))
master_iter_withid = [{'pars':pars,'id':ii} for ii,pars in enumerate(master_iter)]


if __name__ == "__main__":
    pool = Pool(nworkers)
    pool.map(run_model,master_iter_withid)
                