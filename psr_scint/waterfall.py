import astropy.units as u
import astropy.constants as c
from baseband import vdif
from astropy.time import Time
import numpy as np
import sys
import os
import scintillometry_reduction as sr

fdir = sys.argv[1]
first = int(sys.argv[2])
last = int(sys.argv[3])

dispersion_measure = 56.7613 * u.pc / u.cm**3
frequency    = 800.*u.MHz - (np.arange(1024))*(400*u.MHz/1024)
sideband     = -1
polyco_file  = './B0531+21_58458_polyco.dat'
nperiods     = 330 # the number of periods to reduce in each file
nint         = 64 # number of samples to integrate.  Can also use atime in eg. seconds.
tbin         = (nint*1/(400*u.MHz/1024)).to(u.microsecond)
polarization = [['X'],['Y']]
fullpol      = True
nthreads     = os.getenv('SLURM_CPUS_PER_TASK',1)

# Open the files
files = ['{0}/{1:07d}.vdif'.format(fdir, fnumber)
         for fnumber in range(first, last+1)]
fh = vdif.open(files,'rs')
print('Opened vdif stream from {0} to {1}'.format(fh.start_time.isot, fh.stop_time.isot))

# Initialize waterfall integrator
WF = sr.Waterfall(fh, dispersion_measure, frequency, 
                  sideband, polyco_file, nint, tbin,
                polarization, fullpol, nthreads)

# Run over sample of the data until we reach the end of the file
nsamples            = WF.integrator.shape[0]
nsamples_per_output = 2**16 #2**13
output              = np.zeros((nsamples_per_output,WF.integrator.shape[1],WF.integrator.shape[2],WF.integrator.shape[3]))


while WF.integrator.tell() < nsamples - nsamples_per_output:
    WF.integrate_and_save(output)
    # Go back the number of samples that are corrupt by the dedispersion
    # Removed because Jing says the code already deals with this
    # WF.integrator.seek(-WF.ncorrupt,1) 
    
nsamples_per_output = WF.integrator.shape[0]-WF.integrator.tell()
output              = np.zeros((nsamples_per_output,WF.integrator.shape[1], WF.integrator.shape[2],WF.integrator.shape[3]))
integrator = WF.integrate_and_save(output)
WF.integrator.close()
