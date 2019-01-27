import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.time import Time
import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)




def gpfinder(wfall_file, thresh):

    data = np.load(wfall_file)
    deltat = float(data['tbin_microsecond'])*u.microsecond
    tstart = Time(float(data['t0']),format='mjd')
    wfall = data['I']

    #wfall = wfall[2000:-5000,:,0]+wfall[2000:-5000,:,-1]
    wfall = wfall[:,0,0,:] + wfall[:,0,-1,:]
    wfall = wfall.T
    wfall = wfall/wfall.mean(axis=1,keepdims=True) -1
    med_std = np.nanmedian(np.std(wfall,axis=1))
    wfall[np.abs(np.std(wfall,axis=1)-med_std)>0.01,:]=np.nan
    
    power = np.nanmean(wfall,axis=0)
    
    gps = []
    sns = []
    
    for i in range(len(power)):
        sn = ((power[i] - np.median(power[max(0,i-500):min(i+500,len(power))]))
              /np.std(power[max(0,i-500):min(i+500,len(power))]))
        if sn > thresh:
            gps += [i]
            sns += [sn]       
        
    sns,gps = zip(*sorted(zip(sns, gps),reverse=True))
    with open("{0}_gps.txt".format(tstart.isot),'w') as f,  PdfPages('{0}_gps.pdf'.format(tstart.isot)) as pdf:
        f.write('Time\tS/N\n')
        for i in range(len(sns)//6):
            fig, ax = plt.subplots(2,3,figsize=(6*3,6*2))
            ax = ax.flatten()
            for j in range(len(ax)):
                if i*6+j < len(sns):
                    pdata = wfall[:,max(gps[i*6+j]-50,0):min(gps[i*6+j]+50,wfall.shape[1])]
                    pdata = np.nanmean(pdata.reshape(128,-1,pdata.shape[1]),axis=1)
                    ax[j].imshow(pdata,aspect='auto',origin='upper',interpolation='none',vmin=-0.1,vmax=0.1,
                                 extent=[(-50*deltat).to(u.ms).value,(+50*deltat).to(u.ms).value,
                                         400,800], cmap=plt.get_cmap('viridis'))
                    ax[j].set_xlabel('time (ms)')
                    ax[j].set_ylabel('freq (MHz)')
                    ax[j].set_title('{0}  S/N {1:.2f}'.format((gps[i*6+j]*deltat+tstart).isot,sns[i*6+j]))
                    f.write('{0}\t{1}\n'.format((gps[i*6+j]*deltat+tstart).mjd,sns[i*6+j]))
            pdf.savefig()
            plt.close()
    return len(gps)


if __name__=='__main__':
    files = glob.glob('2018*.npz')
    #deltat = 163.84*u.microsecond
    thresh = 3

    for wfall_file in files:
        print('Starting search of {0}'.format(wfall_file))
        ndetect = gpfinder(wfall_file, thresh)
        print('Found {0} giant pulses'.format(ndetect))

