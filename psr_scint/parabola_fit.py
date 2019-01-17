import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.time import Time
from tqdm import tqdm
from scipy.fftpack import fft2,fftshift,ifft2,ifftshift,fftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

################################################################
# parabola_fit.py
# Dana Simard, January 2019
# This module contains code for fitting a parabola to a secondary spectrum.
# It allows both the curvature of the parabola and the location of the apex 
# along the f_D or delay rate axis to vary.  
################################################################

def __init__():
    return

def gauss(x, *p):
    a, b, c, d = p
    y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d
    return y

def prep_for_fitting_parabola(If,doppler,delay,doppler_extent=15*u.mHz,rebin_delay=4):
    # Prepares the secondary spectrum for fitting.  In particular, this 
    # creates the secondary spectrum, zooms in to the part that is appropriate for 
    # fitting, and bins the spectrum in the delay axis.
    # Parameters:
    # -----------
    # If:             conjugate spectrum, numpy array, axes should be (delay,Doppler)
    # doppler:        Doppler frequencies, 1d numpy array, units of frequency
    # delay:          delays, 1d numpy array, units of time
    # doppler_extent: extent of Doppler frequencies you want to include in 
    #                 cropped array, Quantity, units of frequency
    # rebin_delay:    number of pixels to bin the delay by, integer
    # Returns:
    #------------
    # Ifit:           the array to perform fitting on, numpy array, axes (delay, Doppler)
    # do_fit:         Doppler frequencies, 1d numpy array, units frequency
    # de_fit:         delays, 1d numpy array, units time
    #
    Ifit = If*np.conjugate(If)
    Ifit = np.abs(Ifit.reshape((Ifit.shape[0]//rebin_delay,rebin_delay,Ifit.shape[1])).mean(axis=1))
    Ifit = Ifit[Ifit.shape[0]//2:,:]
    do_fit = doppler[np.argmin(np.abs(doppler+doppler_extent)):np.argmin(np.abs(doppler-doppler_extent))]
    de_fit = delay[len(delay)//2::rebin_delay]
    return Ifit, do_fit, de_fit
    
def fit_eta(Ifit,de_fit,do_fit,etas,deta, x0,icutoff):
    # Transforms the secondary spectrum to a 1D spectrum: flux vs eta, the curvature of the secondary 
    # spectrum, while holding x0, the apex of the parabola along the Doppler axis, constant.
    # Parameters:
    # -----------
    # Ifit:    the array to perform fitting on, numpy array, axes (delay, Doppler)
    # do_fit:  Doppler frequencies, 1d numpy array, units frequency
    # de_fit:  delays, 1d numpy array, units time
    # etas:    numpy array, the values of curvature to fit for, units of T^3
    # deta:    Quantity, the separation between values of eta, units of T^3
    # x0:      Quantity, the location of the apex of the parabola along the Doppler axis,
    #          units of Frequency
    # icutoff: integer, the index of the minimum delay to include flux from in the fit
    # Returns:
    # -----------
    # sums:    the sum of flux in each parabola, 1d array, same length as etas
    # counts:  the number of bins used to calculate the sum in each parabola
    #          1d array, same length as etas
    #
    sums   = np.zeros((etas.shape[0]))
    counts = np.zeros((etas.shape[0]))
    for i,eta in tqdm(enumerate(etas)):
        mask = np.zeros(Ifit.shape)
        for j in np.arange(len(do_fit)):
            y1 = (eta-deta/2)*(np.abs(do_fit[j]-x0))**2
            y2 = (eta+deta/2)*(np.abs(do_fit[j]-x0))**2
            k1 = np.searchsorted(de_fit,y1)-1
            k2 = np.searchsorted(de_fit,y2)-1
            if k1 < k2:
                mask[k1:k2+1,j] = 1
            else:
                mask[k2:k1+1,j] = 1

        mask[:icutoff,:] = 0
        sums[i]          = (np.abs(Ifit*mask)).sum()
        counts[i]        = mask.sum()
    return sums, counts

def fit_x0(Ifit,de_fit,do_fit,eta,deta,x0s,icutoff):
    # Transforms the secondary spectrum to a 1D spectrum: flux vs x0, the apex of the parabola along the Doppler 
    # axis, while holding eta, the curvature of the secondary 
    # spectrum, constant.
    # Parameters:
    # -----------
    # Ifit:    the array to perform fitting on, numpy array, axes (delay, Doppler)
    # do_fit:  Doppler frequencies, 1d numpy array, units frequency
    # de_fit:  delays, 1d numpy array, units time
    # eta:     Quantity, the value of curvature, units of T^3
    # deta:    Quantity, the separation between values of eta, units of T^3
    # x0s:     1d array, the location of the apex of the parabola along the Doppler axis
    #          to fit for, units of Frequency
    # icutoff: integer, the index of the minimum delay to include flux from in the fit
    # Returns:
    # -----------
    # sums:    the sum of flux in each parabola, 1d array, same length as x0s
    # counts:  the number of bins used to calculate the sum in each parabola
    #          1d array, same length as x0s
    #
    sums   = np.zeros((x0s.shape[0]))
    counts = np.zeros((x0s.shape[0]))

    for i,x0 in tqdm(enumerate(x0s)):
        mask = np.zeros(Ifit.shape)
        for j in np.arange(len(do_fit)):
            y1 = (eta-deta/2)*(np.abs(do_fit[j]-x0))**2
            y2 = (eta+deta/2)*(np.abs(do_fit[j]-x0))**2
            k1 = np.searchsorted(de_fit,y1)-1
            k2 = np.searchsorted(de_fit,y2)-1
            if k1 < k2:
                mask[k1:k2+1,j] = 1
            else:
                mask[k2:k1+1,j] = 1

        mask[:icutoff,:] = 0
        sums[i]          = (np.abs(Ifit*mask)).sum()
        counts[i]        = mask.sum()
    return sums, counts

def fit_curvature_offset(Ifit, de_fit, do_fit, icutoff, etas, deta, x0, x0s,
                         p_eta = [1200,2,0.5,4500],p_x0 = [2200,0.25,0.5,3500]):
    # Performs one iteration of fitting first for eta, the curvature of the parabola,
    # and then for x0, the location of the apex of the parabola along the Doppler 
    # frequency axis.
    # Parameters:
    # -----------
    # Ifit:    the array to perform fitting on, numpy array, axes (delay, Doppler)
    # de_fit:  delays, 1d numpy array, units time
    # do_fit:  Doppler frequencies, 1d numpy array, units frequency
    # icutoff: integer, the index of the minimum delay to include flux from in the fit
    # etas:    1d array, the values of curvature to fit for, units of T^3
    # deta:    Quantity, the separation between values of eta, units of T^3
    # x0:      Quantity, the initial value of x0 to use in the fit
    # x0s:     1d array, the location of the apex of the parabola along the Doppler axis
    #          to fit for, units of Frequency
    # p_eta:   list, initial values for Gaussian fit to flux v curvature, 
    #          amplitude, mean, standard deviation, y offset
    # p_x0:    list, initial values for Gaussian fit to flux v x0, 
    #          amplitude, mean, standard deviation, y offset
    # Returns:
    # -----------
    # eta:          the best fit value for eta
    # eeta:         the error from curve_fit on the fit to eta
    # x0:           the fitted value of x0
    # ex0:          the error from curve_fit on the fit to x0
    # avg_flux_eta: the avg flux in each pixel for parabolas during the fit for eta
    # avg_flux_x0:  the avg flux in each pixel for parabolas during the fit for x0
    #
    sums_eta, counts_eta = fit_eta(Ifit, de_fit, do_fit, etas, deta, x0, icutoff)
    popt, pcov = curve_fit(gauss, etas, 
                (sums_eta/counts_eta), p0=p_eta)
    eta=popt[1]*u.s**3
    eeta = np.sqrt(pcov[1,1])*u.s**3
    sums_x0,counts_x0 = fit_x0(Ifit, de_fit, do_fit, eta, deta, x0s, icutoff)
    popt, pcov = curve_fit(gauss, x0s.value, 
                       (sums_x0/counts_x0), p0=p_x0)
    x0 = popt[1]*u.mHz
    ex0 = np.sqrt(pcov[1,1])*u.mHz
    return eta, eeta, x0, ex0, sums_eta/counts_eta, sums_x0/counts_x0

def plot_fit_parabola(Ifit, do_fit, de_fit, eta, eeta, x0, ex0):
    # Plots a parabola over the secondary spectrum
    # Parameters:
    # -----------
    # Ifit:    the array to plot, numpy array, axes (delay, Doppler)
    # de_fit:  delays, 1d numpy array, units time
    # do_fit:  Doppler frequencies, 1d numpy array, units frequency
    # eta:          the value of eta to plot
    # eeta:         the uncertainty on eta
    # x0:           the value of x0 to plot
    # ex0:          the uncertainty on x0
    #
    plt.figure()
    plt.imshow(np.log10(np.abs(Ifit)),aspect='auto',origin='lower',interpolation='none',
              extent=[do_fit[0].value,do_fit[-1].value,
                      de_fit[0].value,de_fit[-1].value],vmin=3,vmax=8)
    plt.autoscale(False)
    plt.plot(do_fit,eta*(do_fit-x0)**2,color='white',ls=':')
    plt.colorbar()
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ ($\mu$s)')
    plt.title(r'$\eta$: {0:.3f} $\pm$ {1:.3f}; x0: {2:.3f} $\pm$ {3:.3f}'.format(eta,eeta,x0,ex0))