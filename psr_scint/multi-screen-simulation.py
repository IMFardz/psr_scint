import numpy as np
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq,fft,ifft
import astropy.units as u
import astropy.constants as c
from tqdm import tqdm
import sys

def __init__():
    return

def get_parameters():
    # Grid parameters
    npixf = 2048
    npixt = 1024
    tmax  = 6000.*u.s
    fmin  = 310.*u.MHz
    bwidth= 1.*u.MHz

    # Baseline parameters
    b     = 3000*u.km

    # Lensing geometry parameters
    dpsr   = 1*u.kpc
    dlens1 = 0.4*dpsr # Closer to the observer
    dlens2 = 0.8*dpsr # Closer to the pulsar
    vpsr   = 200*u.km/u.s
    vobs   = 0*u.km/u.s
    vlens1 = 0*u.km/u.s
    vlens2 = 0*u.km/u.s

    # Image parameters
    nimages1 = 200 # Number of lensed images on screen 1
    nimages2 = 200 # Number of lensed images on screen 2
    resolved = False # Whether or not screen 1 resolves screen 2
    
    return npixf,npixt,tmax,fmin,bwidth,b,dpsr,dlens1,dlens2,vpsr,vobs,vlens1,vlens2,nimages1,nimages2,resolved

def inverted_parabola(x,x0,y0):
    a = -y0/x0**2
    return y0+a*(x-x0)**2

def parabola(x,x0,y0):
    a = np.abs(y0/x0**2)
    return a*x**2

def get_dthetaref(mag_images,theta_images):
    return (np.abs(mag_images)*3/2)**(3/5)*np.abs(theta_images)

def multiscreen_fd(vpsr,v2,v1,vobs,dpsr,d2,d1,lamb,theta2,theta1):
    '''
    Returns the Doppler shift of an image after being lensed by two different screens.
    Screen 2 is closer to the pulsar, and screen 1 is closer to the observer.
    '''
    return (((vpsr*theta2*d2/(dpsr-d2) + vobs*theta1 - v2*theta2*dpsr/(dpsr-d2) + 
        v1*theta2*d2/(d2-d1) - v1*theta1*d2/(d2-d1))/lamb)/u.rad).to(u.mHz)

def multiscreen_tau(d1,d2,dpsr,theta1,theta2):
    '''
    Returns the delay of an image after being lensed by two different screens.
    Screen 2 is closer to the pulsar, and screen 1 is closer to the observer.
    '''
    return (((1/2*d1*d2/(d2-d1)*(theta1**2 - 2*theta1*theta2 + 
                               d2/d1*(dpsr-d1)/(dpsr-d2)*theta2**2))/c.c)/u.rad**2).to(u.microsecond)

def multiscreen_mag(mag_images1,mag_images2,theta_images1,theta_images2):
    return mag_images2[i]*2/3*np.abs(dthetaref(mag_images1,theta_images1)/(theta_images2[i]-theta_images1[j]))**(5/3)

def singlescreen_tau(dlens,dpsr,theta):
    return ((1/(2*c.c)*theta**2*dlens*dpsr/(dpsr-dlens))/u.rad**2).to(u.microsecond)

def singlescreen_fd(vpsr,vlens,vobs,dpsr,dlens,lamb,theta):
    s = 1 - dlens/dpsr
    return ((theta/lamb*(vpsr*(1-s)/s+vobs-vlens/s))/u.rad).to(u.mHz)

def singlescreen_mag(theta,dthetaref):
    return 2/3*(np.abs(dthetaref/theta))**(5/3)

def set_up_spectra(npixt,tmax,npixf,bwidth,fmin):
    t = np.arange(npixt)*tmax/npixt
    f = fmin + np.arange(npixf)*bwidth/(npixf)

    doppler = (fftshift(fftfreq(npixt))*1/np.median(np.diff(t))).to(u.mHz)
    delay   = (fftshift(fftfreq(npixf))*1/np.median(np.diff(f))).to(u.microsecond)

    f0 = np.median(f)
    return t,f,doppler,delay,f0

def set_up_geometry(dlens,dpsr,vpsr,vobs,vlens,f0,alphavs=0*u.rad):
    s     = 1-dlens/dpsr
    Deff  = dpsr*(1-s)/s
    Veff  = (vpsr*(1-s)/s + vobs - vlens/s)*np.cos(alphavs)
    eta   = (Deff/(2*c.c)*(c.c/f0)**2/(Veff)**2).to(u.s**3)
    return s,Deff,Veff,eta

def set_up_images(doppler_max,delay_max,f0,Veff,Deff,nimages,
                 vpsr,vlens,vobs,dpsr,dlens,s):
    dtheta_max     = np.min([np.abs(doppler_max*c.c/f0/Veff*u.rad).to(u.mas).value,
                          np.abs(np.sqrt(2*c.c*delay_max/Deff)*u.rad).to(u.mas).value])*u.mas
    theta_images   = np.abs(np.random.rand(nimages))*dtheta_max*2-dtheta_max

    doppler_images = singlescreen_fd(vpsr,vlens,vobs,dpsr,dlens,c.c/f0,theta_images)
    delay_images   = singlescreen_tau(dlens,dpsr,theta_images)
    mag_images     = np.abs(np.random.rand(nimages))*np.exp((-np.abs(theta_images)/(dtheta_max/3)))
    dthetaref      = get_dthetaref(mag_images,theta_images)
    #dthetaref      = (np.abs(np.random.rand(nimages))*s/np.sqrt(2))**(2/5)*10.*u.mas
    #mag_images     = singlescreen_mag(theta_images,dthetaref)
    phi_images     = ((2*np.pi*f0 * delay_images).to(u.dimensionless_unscaled)*u.rad) #+ (np.random.rand((nimages))*2*np.pi-np.pi)*u.rad)#%(2*np.pi*u.rad)
    return theta_images,dthetaref,doppler_images,delay_images,mag_images,phi_images

def set_up_images_resolved(nimages1,nimages2,mag_images2,dthetaref1,theta_images2,theta_images1,dlens1,dlens2,dpsr,vpsr,vlens2,vlens1,vobs,phi_images2,phi_images1,f0):
    nimages = nimages1*nimages2

    doppler_images = np.zeros(nimages)*u.mHz
    delay_images   = np.zeros(nimages)*u.microsecond
    mag_images     = np.zeros(nimages)
    phi_images     = np.zeros(nimages)*u.rad
    theta_images   = np.zeros(nimages)*u.mas

    for i in range(nimages2):
        for j in range(nimages1):
            mag_images[i*nimages1+j]     = mag_images2[i]*2/3*np.abs(dthetaref1[j]/(theta_images2[i]-theta_images1[j]))**(5/3)
            delay_images[i*nimages1+j]   = multiscreen_tau(dlens1,dlens2,dpsr,theta_images1[j],theta_images2[i])
            doppler_images[i*nimages1+j] = multiscreen_fd(vpsr,vlens2,vlens1,vobs,dpsr,dlens1,dlens2,
                                                            c.c/f0,theta_images2[i],theta_images1[j])
            phi_images[i*nimages1+j]     = phi_images2[i]+phi_images1[j]
            theta_images[i*nimages1+j]   = theta_images1[j]
    
    return nimages,doppler_images,delay_images,mag_images,phi_images,theta_images

def add_images(Ie0,Ie1,nimages,doppler,doppler_images,delay,delay_images,mag_images,phi_images,f0,b,theta_images):
    for i in range(nimages):
        idoppler = np.argmin(np.abs(doppler-doppler_images[i]))
        idelay   = np.argmin(np.abs(delay-delay_images[i]))
        Ie0[idelay,idoppler] += (1e4*mag_images[i])*np.exp(1.0j*phi_images[i].value)
        Ie1[idelay,idoppler] += (1e4*mag_images[i])*np.exp(1.0j*phi_images[i].value +(2.0j*np.pi*f0/c.c*theta_images[i]*b).to(u.rad).value)
    return Ie0,Ie1

def add_freq_dependence(Ie0_2,Ie1_2,f):
    N  = Ie0_2.shape[-1]
    n  = np.arange(N)
    k0 = n.reshape((N,1))
    If0 = np.zeros(Ie0_2.shape,dtype=complex)
    If1 = np.zeros(Ie1_2.shape,dtype=complex)
    for i in tqdm(range(len(f))):
        k = np.where(k0>=N//2,k0-N,k0)
        k = k*(f[0]/f[i]).to(u.dimensionless_unscaled).value
        k = np.where(k<0,k+N,k)  
        k = k%N
        M = np.exp(-2j*np.pi*k*n/N)

        Itemp = np.dot(M,Ie0_2[i,:])
        If0[i,:] = ifft(Itemp)
        If0[i,:] = np.roll(ifft(Itemp),-int(round((f[0]/f[i]-1).to(u.dimensionless_unscaled).value*N))//2,axis=-1)
        Itemp = np.dot(M,Ie1_2[i,:])
        If1[i,:] = ifft(Itemp)
        If1[i,:] = np.roll(ifft(Itemp),-int(round((f[0]/f[i]-1).to(u.dimensionless_unscaled).value*N))//2,axis=-1)
    return If0,If1

def simulate_spectra(npixf,npixt,tmax,fmin,bwidth,b,dpsr,dlens1,dlens2,vpsr,
                    vobs,vlens1,vlens2,nimages1,nimages2,resolved):
    
    # Set up the dynamic and secondary spectra
    t,f,doppler,delay,f0 = set_up_spectra(npixt,tmax,npixf,bwidth,fmin)
    
    # Set up the lensing geometry
    s1,Deff1,Veff1,eta1 = set_up_geometry(dlens1,dpsr,vpsr,vobs,vlens1,f0)
    s2,Deff2,Veff2,eta2 = set_up_geometry(dlens2,dpsr,vpsr,vobs,vlens2,f0)

    # Set up the images
    doppler_max = doppler[-1]/2
    delay_max   = delay[-1]/2
    theta_images1,dthetaref1,doppler_images1,delay_images1,mag_images1,phi_images1 = \
        set_up_images(doppler_max,delay_max,f0,Veff1,Deff1,nimages1,vpsr,vlens1,vobs,dpsr,dlens1,s1)
    theta_images2,dthetaref2,doppler_images2,delay_images2,mag_images2,phi_images2 = \
        set_up_images(doppler_max,delay_max,f0,Veff2,Deff2,nimages2,vpsr,vlens2,vobs,dpsr,dlens2,s2)

    if resolved:
        nimages3,doppler_images3,delay_images3,mag_images3,phi_images3,theta_images3 = \
            set_up_images_resolved(nimages1,nimages2,mag_images2,dthetaref1,theta_images2,theta_images1,
                                        dlens1,dlens2,dpsr,vpsr,vlens2,vlens1,vobs,phi_images2,phi_images1,f0)
        
    # Populate the electric fields in secondary space
    Ie0 = np.zeros((npixf,npixt),dtype=complex)
    Ie1 = np.zeros((npixf,npixt),dtype=complex)
    Ie0,Ie1 = add_images(Ie0,Ie1,nimages1,doppler,doppler_images1,delay,
                              delay_images1,mag_images1,phi_images1,f0,b,theta_images1)
    Ie0,Ie1 = add_images(Ie0,Ie1,nimages2,doppler,doppler_images2,delay,
                              delay_images2,mag_images2,phi_images2,f0,b,theta_images2)
    if resolved:
        Ie0,Ie1 = add_images(Ie0,Ie1,nimages3,doppler,doppler_images3,delay,
                                  delay_images3,mag_images3,phi_images3,f0,b,theta_images3)

    # Fourier transform to real space
    
    Ie0_2 = ifft2(ifftshift(Ie0))
    Ie1_2 = ifft2(ifftshift(Ie1))
    

    # Add frequency dependence of the fringe rate
    # Add noise
    If0,If1 = add_freq_dependence(Ie0_2,Ie1_2,f)
    If0     = If0 + np.random.normal(loc=0,scale=np.std(If0)/4,size=If0.shape) + np.random.normal(loc=0,scale=np.std(If0)/10,size=If0.shape)*1.0j
    If1     = If1 + np.random.normal(loc=0,scale=np.std(If1)/4,size=If1.shape) + np.random.normal(loc=0,scale=np.std(If1)/10,size=If1.shape)*1.0j

    
    # Calculate the auto and cross spectra
    I0   = If0*np.conjugate(If0)
    I1   = If1*np.conjugate(If1)
    I01  = If0*np.conjugate(If1)

    # Save the results
    if not resolved:
        np.savez('images.npz',theta1_mas=theta_images1.to(u.mas).value,theta2_mas=theta_images2.to(u.mas).value,
             mag1=mag_images1.value,mag2=mag_images2.value)
    else:
        np.savez('images.npz',theta1_mas=theta_images1.to(u.mas).value,theta2_mas=theta_images2.to(u.mas).value,
             mag1=mag_images1.value,mag2=mag_images2.value,theta3_mas=theta_images3.to(u.mas).value,mag3=mag_images3)
    
    np.savez('secondary_spectra.npz',I00=I0,I11=I1,I01=I01)
    
    fi = open('parameters.txt','w')
    fi.write('# Grid parameters')
    fi.write('npixf  = {0}'.format(npixf))
    fi.write('npixt  = {0}'.format(npixt))
    fi.write('tmax   = {0}'.format(tmax))
    fi.write('fmin   = {0}'.format(fmin))
    fi.write('bwidth = {0}'.format(bwidth))
    fi.write('')
    fi.write('# Baseline parameters')
    fi.write('b      = {0}'.format(b))
    fi.write('')
    fi.write('# Lensing geometry parameters')
    fi.write('dpsr   = {0}'.format(dpsr))
    fi.write('dlens1 = {0}'.format(dlens1))
    fi.write('dlens2 = {0}'.format(dlens2))
    fi.write('vpsr   = {0}'.format(vpsr))
    fi.write('vobs   = {0}'.format(vobs))
    fi.write('vlens1 = {0}'.format(vlens1))
    fi.write('vlens2 = {0}'.format(vlens2))
    fi.write('')
    fi.write('# Image parameters')
    fi.write('nimages1= {0}'.format(nimages1))
    fi.write('nimages2= {0}'.format(nimages2))
    fi.write('resolved= {0}'.format(resolved))
    fi.write('')
    fi.close()

    return I0,I1,I01,t,f

if __name__=='__main__':
    npixf,npixt,tmax,fmin,bwidth,b,dpsr,dlens1,dlens2,vpsr,vobs,vlens1,vlens2,nimages1,nimages2,resolved = get_parameters()
    simulate_spectra(npixf,npixt,tmax,fmin,bwidth,b,dpsr,dlens1,dlens2,vpsr,
                    vobs,vlens1,vlens2,nimages1,nimages2,resolved)   
