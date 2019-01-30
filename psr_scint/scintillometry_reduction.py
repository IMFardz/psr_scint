import astropy.units as u
import astropy.constants as c
from baseband import vdif
from scintillometry import dm, dispersion, channelize, functions, integration, fourier
from astropy.time import Time
import numpy as np
from pulsar import predictor

class Waterfall:

    def __init__(self, fh, dispersion_measure, frequency,
                 sideband, polyco_file, nint, tbin, 
                 polarization, fullpol, nthreads):

        self.dispersion_measure = dispersion_measure
        self.frequency    = frequency
        self.sideband     = sideband
        self.polyco_file  = polyco_file
        self.nint         = nint
        self.tbin         = tbin
        self.polarization = polarization
        self.fullpol      = fullpol
        self.nthreads     = nthreads

        self.integrator = self.initialize_pipeline(fh)

        # Note that you instead of the number of bins that you want to integrate together
        # you can pass a time or a number of cycles. If you pass a number of cycles, the
        # integrator.time() function currently doesn't work
        # eg. if
        # tbin         = (1/128)*u.cycle  
        # then the integrator function should look like
        # integrator  = integration.Integrate(power, tbin, psr_polyco) 
          
    def initialize_pipeline(self,fh):
        dispersion_measure = dm.DispersionMeasure(self.dispersion_measure)
        psr_polyco         = predictor.Polyco(self.polyco_file)
        #FFT                = fourier.get_fft_maker('pyfftw',threads=self.nthreads)
    
        # Build the pipeline
        dedisperser = dispersion.Dedisperse(fh, dispersion_measure, 600*u.MHz,
                                            frequency=self.frequency,sideband=self.sideband, samples_per_frame=2**20)
        channelizer = channelize.Channelize(dedisperser, 1)
        if self.fullpol:
            power   = functions.Power(channelizer,polarization=self.polarization)
        else:
            power   = functions.Square(channelizer,polarization=self.polarization)
        integrator = integration.Integrate(power, self.nint)
        return integrator

    def integrate_and_save(self,output):
        tstart = self.integrator.time
        print('Starting at time {0}'.format(tstart.isot))
        self.integrator.read(out=output)
        # Note - calculating the time doesn't currently work for the
        # integrator if you are using bins in phase.
        np.savez('{0}'.format(tstart.isot),
                I=output,t0=tstart.mjd,tbin_microsecond=self.tbin.to_value(u.microsecond))

    










