import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.fftpack import fftshift, fft2, ifftshift, ifft2, fftfreq

def __init__(self):
    ''' A module for generating secondary cross spectra.'''
    return

class spectrum():
    ''' 
    A class for containing dynamic spectra and calculating their 
    secondary spectra
    '''
    def __init__(self,I,f,t):
        ''' 
        Parameters:
        -----------
        I:  array of dynamic spectra.  If a single dynamic spectrum is passed, 
            it is assumed that it is a dynamic cross spectrum between two stations.
            If two dynamic spectra are passed, it is assumed that they are two auto
            dynamic spectra from two stations.  If three are passed, it is assumed
            that they are I_00 , I_11, I_01.  Note that changing the order will change
            the comparitive mapping of angles. 
        f:  array, the frequencies of each pixel along the frequency dimension 
            of the dynamic spectrum.  Can be floats or astropy quantity.
        t:  an array, the times of each pixel along the time dimension of the dynamic
            spectrum.  Can be floats or astropy quantity or time.
        '''
        self.f = f
        self.t = t
        
        if I.shape[1] == len(f) and I.shape[0] == len(t):
            I = I.T

        assert (I.shape[0] == len(f) and I.shape[1] == len(t)), \
            'Dynamic spectra must have shape (len(f),len(t)) or (len(t),len(f))'
        
        self.I = I

    def prepare_dynamic_spectrum(self,nfsplits,ntsplits,nbins_tau,nbins_fd):
        '''
        MAINLY UNTESTED

        Prepares the original dynamic spectra for the fft by padding and reshaping
        to the shape required in order to perform the fft and removing any nan's from
        the dynamic spectrum.  In future, may also want this to make sure all 
        of the arrays have the same statistics.
        Parameters:
        -----------
        nfsplits:  int, the number of spectra to divide the dynamic spectrum into 
                   along the frequency axis before calculating the secondary spectrum
        ntsplits:  int, the number of spectra to divide the dynamic spectrum into 
                   along the time axis before calculating the secondary spectrum
        nbins_tau: int, the number of pixels along the delay axis in the output 
                   secondary spectrum
        nbins_fd:  int, the number of pixels along the Doppler frequency (fringe 
                   rate) axis in the output secondary spectrum
        '''
        mean = np.nanmean(self.I)
        
        Iprep = np.copy(self.I)
        fprep = self.f.copy()
        tprep = self.t.copy()
            
        nf = Iprep.shape[0]
        nt = Iprep.shape[1]

        if (nt//ntsplits)%nbins_fd != 0:
            ntpad = (nt//ntsplits//nbins_fd+1)*ntsplits*nbins_fd-nt
        else:
            ntpad = 0
            
        if (nf//nfsplits)%nbins_tau != 0:
            nfpad = (nf//nfsplits//nbins_tau+1)*nfsplits*nbins_tau-nf
        else:
            nfpad = 0

        Iprep = np.pad(Iprep,((0,nfpad),(0,ntpad)),mode='constant',
                        constant_values=(np.nan,))
        # Test if this will break for an array of times
        tprep = np.concatenate((tprep.value,tprep[-1].value+np.median(np.diff(tprep.value))
                                *(np.arange(ntpad)+1)))*tprep.unit
        fprep = np.concatenate((fprep.value,fprep[-1].value+np.median(np.diff(fprep.value))
                                *(np.arange(nfpad)+1)))*fprep.unit

        Iprep[Iprep!=Iprep] = mean
        return Iprep,fprep,tprep
        
    def secondary_spectrum(self,nfsplits=1,ntsplits=1,nbins_tau=None,
                           nbins_fd=None):
        '''
        MAINLY UNTESTED

        Generates the secondary spectrum from the dynamic spectrum.
        Parameters:
        -----------
        nfsplits:  int, the number of spectra to divide the dynamic spectrum into 
                   along the frequency axis before calculating the secondary spectrum
        ntsplits:  int, the number of spectra to divide the dynamic spectrum into 
                   along the time axis before calculating the secondary spectrum
        nbins_tau: int, the number of pixels along the delay axis in the output 
                   secondary spectrum
        nbins_fd:  int, the number of pixels along the Doppler frequency (fringe 
                   rate) axis in the output secondary spectrum
        '''
        if nbins_tau is None:
            nbins_tau = len(self.f)//nfsplits*2
        if nbins_fd is None:
            nbins_fd  = len(self.t)//ntsplits*2
        
        Iprep,fprep,tprep = self.prepare_dynamic_spectrum(nfsplits,ntsplits,nbins_tau,nbins_fd)
        nf   = Iprep.shape[0]//nfsplits
        nt   = Iprep.shape[1]//ntsplits
        mean = Iprep.mean()
        
        self.Ic = np.zeros((nfsplits,ntsplits,nf*2,nt*2),dtype=complex)
        for i in range(nfsplits):
            for j in range(ntsplits):
                self.Ic[i,j,...] = fftshift(fft2(np.pad(
                            Iprep[i*nf:(i+1)*nf,j*nt:(j+1)*nt],
                            ((0,nf),(0,nt)),
                            mode='constant',constant_values=(mean,))))
        self.delay = (fftshift(fftfreq(nf*2))*1/np.median(np.diff(fprep))
                      ).to(u.microsecond)
        self.doppler = (fftshift(fftfreq(nt*2))*1/np.median(np.diff(tprep))
                        ).to(u.mHz)
        
        nbinf = nf*2//nbins_tau
        nbint = nt*2//nbins_fd

        self.Ic = np.mean(self.Ic.reshape((nfsplits,ntsplits,nbins_tau,nbinf,nt*2)),
                          axis=3)
        self.Ic = np.mean(self.Ic.reshape((nfsplits,ntsplits,nbins_tau,nbins_fd,
                                           nbint)),axis=-1)
        
        self.delay   = self.delay[::nbinf]
        self.doppler = self.doppler[::nbint]

        
def get_conjugate_spectra(I,t,f,ntsplit=1,nrebin_fd=1,nrebin_tau=1):
    '''
    Calculate the conjugate spectrum from the dynamic spectrum. 
    Parameters:
    ------------
    I:          numpy array, complex or real, the dynamic spectrum
    t:          numpy array, astropy quantity, the time of each pixel in the 
                dynamic spectrum
    f:          numpy array, astropy quantity, the frequency of each pixel in the 
                dynamic spectrum
    ntsplit:    int, optional, the number of time chunks to split the data into 
                before calculating the secondary spectra, default is 1
    nrebin_fd:  int, optional, the number of bins to rebin the secondary spectrum 
                along the Doppler axis by, default is 1
    nrebin_tau: int, optional, the number of bins to rebind the secondary spectrum 
                along the delay axis by, default is 1
    Returns:
    ------------
    Ic:         numpy array, complex, the conjugate spectra
    doppler:    numpy array, astropy quantity, the values along the Doppler axis of 
                the conjugate spectrum
    delay:      numpy array, astropy quantity, the values along the delay axis of 
                the conjugate spectrum
    '''
    assert I.shape[0] == t.shape[0] and I.shape[1] == f.shape[0]

    I  = I.reshape((ntsplit,-1,I.shape[-1]))
    Ic = fftshift(fft2(np.pad(I,((0,0),(0,I.shape[1]),(0,I.shape[2])),
                              mode='constant',constant_values=(0.,))))
    Ic = Ic.reshape((ntsplit,-1,nrebin_fd,Ic.shape[-1])).mean(axis=2)
    Ic = Ic.reshape((ntsplit,Ic.shape[1],-1,nrebin_tau)).mean(axis=-1)

    doppler = (fftshift(fftfreq(I.shape[1]*2))*1/np.median(np.diff(t))).to(u.mHz)
    delay   = (fftshift(fftfreq(I.shape[2]*2))*1/np.median(np.diff(f))
               ).to(u.microsecond)
    
    doppler = doppler[::nrebin_fd]
    delay   = delay[::nrebin_tau]
    return Ic, doppler, delay

def get_secondary_cross_spectrum(Ic,doppler,delay,nrebin_fd=1,nrebin_tau=1):
    '''
    Calculate the secondary cross spectrum from the conjugate spectrum of the VLBI 
    visibility.  Can process multiple visibilities at once, concatenated along 
    the 0th axis.
    Parameters:
    ------------
    Ic:         numpy array, 2 or 3 dimensions, complex, the conjugate spectrum.  
                The last two dimensions are Doppler frequency and delay.
    doppler:    numpy array, astropy quantity, the values along the Doppler axis 
                of the conjugate spectrum
    delay:      numpy array, astropy quantity, the the values along the delay axis 
                of the conjugate spectrum
    nrebin_fd:  int, optional, the number of bins to rebin the secondary spectrum 
                along the Doppler axis by, default is 1
    nrebin_tau: int, optional, the number of bins to rebind the secondary spectrum 
                along the delay axis by, default is 1
    Returns:
    ------------
    Icross_vlbi:numpy array, complex same dimensions as the input array, the 
                secondary cross spectra
    doppler:    numpy array, astropy quantity, the values along the Doppler axis 
                of the secondary spectrum
    delay:      numpy array, astropy quantity, the values along the delay axis 
                of the secondary spectrum
    '''
    ndim = Ic.ndim
    
    Ic = Ic.reshape((-1,Ic.shape[-2],Ic.shape[-1]))
    assert Ic.shape[1] == doppler.shape[0] and Ic.shape[2] == delay.shape[0]
    
    rollx = 1-Ic.shape[1]%2
    rolly = 1-Ic.shape[2]%2
    
    Icross_vlbi = np.zeros(Ic.shape,dtype=np.complex)
    for i in range(Icross_vlbi.shape[0]):
        Icross_vlbi[i,...] = (Ic[i,...]*np.roll(np.roll(np.fliplr(np.flipud(
                            Ic[i,...])),rollx,axis=0),rolly,axis=1))

    Icross_vlbi = Icross_vlbi.reshape((Icross_vlbi.shape[0],-1,nrebin_fd,
                                       Icross_vlbi.shape[2])).mean(axis=2)
    Icross_vlbi = Icross_vlbi.reshape((Icross_vlbi.shape[0],Icross_vlbi.shape[1],
                                       -1,nrebin_tau)).mean(axis=-1)

    if ndim==2:
        Icross_vlbi = Icross_vlbi.squeeze()
    
    doppler = doppler[::nrebin_fd]
    delay = delay[::nrebin_tau]

    return Icross_vlbi,doppler,delay

def get_cross_secondary_spectrum(Ic0,Ic1,doppler,delay,nrebin_fd=1,nrebin_tau=1):
    '''
    Calculate the cross secondary spectrum, Ic0*np.conjugate(Ic1), from the 
    single dish conjugate spectra.  
    Can process multiple spectra at once, concatenated along the 0th axis.
    Parameters:
    ------------
    Ic0:        numpy array, 2 or 3 dimensions, complex, the conjugate 
                spectrum for station 0.  The last two dimensions are Doppler 
                frequency and delay.
    Ic1:        numpy array, 2 or 3 dimensions, complex, the conjugate spectrum 
                for station 1.  The last two dimensions are Doppler frequency 
                and delay.
    doppler:    numpy array, astropy quantity, the values along the Doppler axis 
                of the conjugate spectrum
    delay:      numpy array, astropy quantity, the the values along the delay axis 
                of the conjugate spectrum
    nrebin_fd:  int, optional, the number of bins to rebin the secondary spectrum 
                along the Doppler axis by, default is 1
    nrebin_tau: int, optional, the number of bins to rebind the secondary spectrum 
                along the delay axis by, default is 1
    Returns:
    ------------
    Icross_single:numpy array, complex same dimensions as the input array, 
                the secondary cross spectra
    doppler:    numpy array, astropy quantity, the values along the Doppler axis 
                of the secondary spectrum
    delay:      numpy array, astropy quantity, the values along the delay axis 
                of the secondary spectrum
    '''
    ndim = Ic0.ndim

    Ic0 = Ic0.reshape((-1,Ic0.shape[-2],Ic0.shape[-1]))
    Ic1 = Ic1.reshape((-1,Ic1.shape[-2],Ic1.shape[-1]))
    assert Ic0.shape[1] == doppler.shape[0] and Ic0.shape[2] == delay.shape[0]
    assert Ic1.shape == Ic0.shape

    Icross_single = Ic0*np.conjugate(Ic1)
    Icross_single = Icross_single.reshape((Icross_single.shape[0],-1,
                                           nrebin_fd,Icross_single.shape[2])
                                          ).mean(axis=2)
    Icross_single = Icross_single.reshape((Icross_single.shape[0],
                                           Icross_single.shape[1],
                                           -1,nrebin_tau)).mean(axis=-1)

    if ndim==2:
        Icross_single = Icross_single.squeeze()
    
    doppler = doppler[::nrebin_fd]
    delay   = delay[::nrebin_tau]


    return Icross_single,doppler,delay


def thetas_dot_baseline(Icross_single,Icross_vlbi,lamb,add_phases=False):
    '''
    Calculate theta \cdot baseline for theta1 and theta2 from 
    the single-dish cross secondary spectrum and the vlbi secondary
    cross spectrum.
    Parameters:
    -----------
    Icross_single: numpy array, complex, the single-dish cross secondary
                   spectrum
    Icross_vlbi:   numpy array, complex, the vlbi secondary cross spectrum
    lamb:          astropy quantity, the observing wavelength
    add_phases:    boolean, if True will calculate the phases from the two spectra
                   and add them.  If False, will multiply the spectra together
                   and then calculate the phase.  Default is False.
    Returns:
    ----------
    theta1b :      numpy array, astropy quantity (dimensions length), 
                   theta1 \cdot b
    theta2b:       numpy array, astropy quantity (dimensions length),
                   theta2 \cdot b
    '''
    assert Icross_single.shape == Icross_vlbi.shape 

    if add_phases:
        theta1b = (np.angle(Icross_single)+np.angle(Icross_vlbi))/2*(lamb/(2*np.pi))
        theta2b = (np.angle(Icross_vlbi)-np.angle(Icross_single))/2*(lamb/(2*np.pi))
    else:
        theta1b = np.angle(Icross_single*Icross_vlbi)/2*(lamb/(2*np.pi))
        theta2b = np.angle(Icross_vlbi*np.conjugate(Icross_single))/2*(lamb/(2*np.pi))
    return theta1b, theta2b

def theta_lm(thetab1,thetab2,u1,v1,u2,v2):
    '''
    Convert theta projected along two different baselines into theta in l,m 
    coordinates using the baseline orientation in the uv plane.
    Parameters:
    ------------
    thetab1: numpy array, astropy quantity, dimensions length, theta \cdot b1
    thetab2: numpy array, astropy quantity, dimensions length, theta \cdot b2
    u1:      astropy quantity, dimensions length, the u component of the baseline 
             b1
    v1:      astropy quantity, dimensions length, the v component of the baseline 
             b1
    u2:      astropy quantity, dimensions length, the u component of the baseline
             b2
    v2:      astropy quantity, dimensions length, the v component of the baseline
             b2
    Returns: 
    ------------
    thetal:  numpy array, astropy quantity, dimensions angle, the l component of 
             the theta vector
    thetam:  numpy array, astropy quantity, dimensions angle, the m component of 
             the theta vector
    '''
    assert thetab1.shape == thetab2.shape

    thetam = (thetab2 - thetab1*u2/u1)/(v2-v1*u2/u1)
    thetal = (thetab1 - thetam*v1)/u1
    thetam = (thetam*u.rad).to(u.mas)
    thetal = (thetal*u.rad).to(u.mas)
    return thetal, thetam

def get_deff(theta1l,theta1m,theta2l,theta2m,delay):
    '''
    Calculate the effective distance from the two angles, theta1 and theta2,
    and the delay.
    Parameters:
    -----------
    theta1l: numpy array, astropy quantity, dimension angle, the component of
             theta1 in the l direction
    theta1m: numpy array, astropy quantity, dimension angle, the component of
             theta1 in the m direction
    theta2l: numpy array, astropy quantity, dimension angle, the component of
             theta2 in the l direction
    theta2m: numpy array, astropy quantity, dimension angle, the component of
             theta2 in the m direction
    delay:   numpy array, astropy quantity, dimension time, the values along 
             the delay axis
    Returns:
    -----------
    deff:    numpy array (same size as thetas), astropy quantity, dimension
             length, the effective distance calculated at each pixel
    '''
    assert theta1l.shape == theta1m.shape
    assert theta1l.shape == theta2l.shape
    assert theta2l.shape == theta2m.shape

    assert theta1l.shape[1] == delay.shape[0]

    deltatheta_squared = (theta2l**2+theta2m**2)-(theta1l**2+theta1m**2)
    deff = (2*c.c*delay[np.newaxis,:]/(deltatheta_squared/u.rad**2)).to(u.pc)
    return deff

def get_s(deff,dpsr,edeff=None,edpsr=None):
    '''
    Calculate s at every pixel in the secondary spectrum
    Parameters:
    -----------
    deff: numpy array, astropy quantity, dimension
          length, the effective distance calculated at each pixel
    dpsr: astropy quantity, dimension length, the distance to the pulsar
    Returns:
    ----------
    s:    numpy array, the fractional distance of the screen between the 
          observer and the pulsar.  s=0 for a screen at the pulsar, s=1 
          for a screen at the observer
    '''
    s = (1/(deff/dpsr+1)).to(u.dimensionless_unscaled).value
    if edeff is None and edpsr is None:
        return s
    else:
        if edeff is None:
            edeff = 0.*u.pc
        if edpsr is None:
            edpsr = 0.*u.pc
        es = s**2 * np.sqrt(
            (edeff/dpsr)**2 + (deff/dpsr**2 * edpsr)**2
            ).to(u.dimensionless_unscaled).value
        return s,es

def get_veff_parallel(theta1l,theta1m,theta2l,theta2m,doppler,lamb):
    '''
    Calculate the effective distance from the two angles, theta1 and theta2,
    and the delay.
    Parameters:
    -----------
    theta1l: numpy array, astropy quantity, dimension angle, the component of
             theta1 in the l direction
    theta1m: numpy array, astropy quantity, dimension angle, the component of
             theta1 in the m direction
    theta2l: numpy array, astropy quantity, dimension angle, the component of
             theta2 in the l direction
    theta2m: numpy array, astropy quantity, dimension angle, the component of
             theta2 in the m direction
    doppler: numpy array, astropy quantity, dimension 1/time, the values along 
             the doppler axis
    lamb:    astropy quantity, dimension lenght, the observing wavelength
    Returns:
    -----------
    veff_parallel:numpy array (same size as thetas), astropy quantity, dimension
             length/time, the effective velocity projected along the direction of
             scattering calculated at each pixel
    '''
    assert theta1l.shape == theta1m.shape
    assert theta1l.shape == theta2l.shape
    assert theta2l.shape == theta2m.shape

    assert theta1l.shape[0] == doppler.shape[0]

    delta_theta = np.sqrt((theta2l-theta1l)**2+(theta2m-theta1m)**2)
    veff_parallel = lamb*doppler[:,np.newaxis]/(delta_theta/u.rad)
    return veff_parallel.to(u.km/u.s)


def calculate_alphab1s(m1,em1,m2,em2,alphab1,alphab2,b1,b2):
    '''
    Calculates alpha_b1 - alpha_s, the angle between baseline 1 and the 
    scattering axis in the uv plane (measured from the 
    positive v axis towards the positive u axis) from the slopes of \phi vs f_D
    measured for cross (single dish) secondary spectra along two different baselines.
    Parameters:
    ------------
    m1:      astropy quantity, the slope of \phi vs f_D measured for baseline 1
    em1:     astropy quantity, the error in the slope of \phi vs f_D measured
             for baseline 1
    m2:      astropy quantity, the slope of \phi vs f_D measured for baseline 2
    em2:     astropy quantity, the error in the slope of \phi vs f_d measured
             for baseline 2
    alphab1: astropy quantity, the angle of baseline 1 in the uv plane
    alphab2: astropy quantity, the angle of baseline 2 in the uv plane
    b1:      astropy quantity, the length of baseline 1
    b2:      astropy quantity, the length of baseline 1
    Returns:
    ------------
    alphab1s:astropy quantity, the angle of between baseline 1 and the scattering
             axis
    ealphab1s:astropy quantity, the variance in alphab1s
    '''
    alphab21 = alphab2 - alphab1
    alphab1s = np.arctan(1/np.sin(alphab21)*(np.cos(alphab21)-b1/b2*m2/m1)).to(u.deg)
    ealphab1s = (np.abs(
            1/(((np.cos(alphab21) - b1/b2*m2/m1)/np.sin(alphab21))**2+1)*
            (np.abs(b1/b2*(np.sqrt((em2/m1)**2 + 
                                   (m2/m1**2*em1)**2))
                    /np.sin(alphab21)))*u.rad).to(u.deg))
    return alphab1s, ealphab1s
    
def calc_veff_from_alphab1s(alphab1s,ealphab1s,m1,em1,b1):
    '''
    Calculates Veff parallel to the scattering axis from the angle
    between the scattering axis and a baseline and the slope 
    of \phi vs f_D measured from the single-dish cross spectrum for that 
    baseline.
    Parameters:
    ------------
    alphab1s: astropy quantity, the angle between the baseline and the
              scattering axis
    ealphab1s: astropy quantity, the uncertainty on alphab1s
    m1:       astropy quantity, the slope of \phi vs f_D measured along
              the baseline
    em1:      astropy quantity, the error in m1
    b1:       astropy quantity, the length of the projected baseline
    Returns:
    ------------
    veff_ll:  astropy quantity, the effective velocity parallel to the 
              scattering axis
    eveff_ll: astropy quantity, the uncertainty in veff_ll
    '''
    veff_ll = 2*np.pi*b1*np.cos(alphab1s)/m1
    eveff_ll = 2*np.pi*b1*np.sqrt(
        (np.cos(alphab1s)/m1**2 * em1)**2 + 
        (np.sin(alphab1s)/m1 * ealphab1s/u.rad)**2)
    return veff_ll.to(u.km/u.s),eveff_ll.to(u.km/u.s)

def calculate_alphas_veff(m1,m2,alphab1,alphab2,b1,b2):
    '''
    Calculates alpha_s, the angle of scattering in the uv plane (measured from the 
    positive v axis towards the positive u axis) from the slopes of \phi vs f_D
    measured for cross (single dish) secondary spectra along two different baselines.
    Parameters:
    ------------
    m1:      astropy quantity, the slope of \phi vs f_D measured for baseline 1
    m2:      astropy quantity, the slope of \phi vs f_D measured for baseline 2
    alphab1: astropy quantity, the angle of baseline 1 in the uv plane
    alphab2: astropy quantity, the angle of baseline 2 in the uv plane
    b1:      astropy quantity, the length of baseline 1
    b2:      astropy quantity, the length of baseline 1
    Returns:
    ------------
    alphas:  astropy quantity, the angle of scattering in the uv plane
    veff_ll: astropy quantity, V_eff parallel to the scattering direction
    '''
    alphab21 = alphab2 - alphab1
    alphab1s = np.arctan(1/np.sin(alphab21)*(np.cos(alphab21)-b1/b2*m2/m1))
    alphas  = alphab1 - alphab1s
    
    veff_ll = 2*np.pi*b1*np.cos(alphab1s)/m1

    return alphas,veff_ll.to(u.km/u.s)

def calc_deff_from_eta_veff(eta,veff_ll,lamb,eeta=None,eveff_ll=None):
    '''
    Calculated D_eff from the curvature, eta, and Veff parallel 
    to the scattering direction, veff_ll.
    Parameters:
    ------------
    eta:     astropy quantity, the curvature 
    veff_ll: astropy quantity, the effective velocity parallel 
             to the scattering direction
    lamb:    astropy quantity, the observing wavelength
    Returns:
    ------------
    deff:    the effective distance of the screen
    '''
    deff = (veff_ll**2*eta*2*c.c/lamb**2).to(u.pc)
    if eeta is None and eveff_ll is None:
        return deff
    else:
        if eeta is None:
            eeta = 0.*u.s**3
        if eveff_ll is None:
            eveff_ll = 0.*u.km/u.s
        edeff = (2*c.c/lamb**2 * np.sqrt(
            (veff_ll**2 * eeta)**2 + 
            (2*veff_ll*eta * eveff_ll)**2)).to(u.pc)
        return deff, edeff


