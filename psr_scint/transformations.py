from astropy.coordinates import EarthLocation, AltAz
import numpy as np
import astropy.units as u

def __init__():
    return

def calculate_uvw(source, time, s1,s2):
    '''
    Calculates the projected uvw track for the baseline s2-s1.
    Negative u points West, Negative v points South.
    Parameters:
    -------------
    source: astropy.coordinates.SkyCoord object, 
            the phase center of the observation
    time:   astropy.time.Time object, the time of the
            observation
    s1:     astropy.coordinates.EarthLocation object,
            station 1
    s2:     astropy.coordinates.EarthLocation object,
            station 2
    Returns:
    ------------
    uvw, a list of the u, v and w tracks 
    '''
    gw    = EarthLocation(lon=0,lat=0)
    b     = s2-s1
    altaz = source.transform_to(AltAz(obstime=time,location=gw))
    H     = time.sidereal_time('mean',longitude=gw.lon) - source.ra   
    d     = source.dec
    trans = np.array([[ np.sin(H),           np.cos(H),                   0],
                      [-np.sin(d)*np.cos(H), np.sin(d)*np.sin(H),np.cos(d)],
                      [ np.cos(d)*np.cos(H),-np.cos(d)*np.sin(H),np.sin(d)]])
    return np.dot(trans,b)

def calculate_deltara_deltadec(l,m,source):
    '''
    Calculates the offset (delta_ra, delta_dec) of a position given by (l,m)
    from a phase center source.
    Parameters:
    ------------
    l:         astropy Quantity, the l coordinate (radians or equivalent)
    m:         astropy Quantity, the m coordinate (radians or equivalent)
    source:    astropy.coordinates.SkyCoord, the phase center
    Returns:
    ------------
    delta_ra:  astropy Quantity, the offset right ascension from the 
               phase center
    delta_dec: astropy Quantity, the offset declination from the phase center
    '''
    l = l.to(u.rad).value
    m = m.to(u.rad).value
    n          = np.sqrt(1 - l**2 - m**2)
    delta_dec  = (np.arcsin(m*np.cos(source.dec) + n*np.sin(source.dec)) - source.dec)
    delta_ra   = np.arctan(l / (np.cos(source.dec)*n - m*np.sin(source.dec)))
    return delta_ra,delta_dec
