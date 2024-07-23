import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
from astropy.io import fits


class PolData(object):

    def __init__(self, polevents,polrsp=None, specrsp=None,reference_time=0.0):
        """
        container class that converts raw POLAR fits data into useful python
        variables

        This can build both the polarimetric and spectral data
        
        :param poevents: path to polarisation data event file
        :param specrsp: path to spectral responce file
        :param polrsp: path to polarisation responce file
                             it will use SABOUNDS to bin you SA data in 'polar_events'
                             if 'NONE', we assume 'SA' data is already binned
        :param reference_time: reference time of the events (in SECOND)

        """
        ebounds=None
        if specrsp is not None:
                hdu_spec = fits.open(specrsp)   
                # This gets the spectral response
                mc_low = hdu_spec['MATRIX'].data.field('ENERG_LO')
                mc_high = hdu_spec['MATRIX'].data.field('ENERG_HI')
                ebounds = np.append(mc_low, mc_high[-1])
                matrix = hdu_spec['MATRIX'].data.field('MATRIX')
                matrix = matrix.transpose()

                # build the POLAR response
                mc_energies = np.append(mc_low, mc_high[-1])
                self.rsp = InstrumentResponse(matrix=matrix, ebounds=ebounds, monte_carlo_energies=mc_energies)
        else :
            pass
    
        # open the event file
        hdu_evt = fits.open(polevents)
        
        # Extract mission and instrument info
        self.mission = hdu_evt['POLEVENTS'].header['TELESCOP']
        self.instrument = hdu_evt['POLEVENTS'].header['INSTRUME']
        pha = hdu_evt['POLEVENTS'].data.field('ENERGY')

        # non-zero ADC channels and correct energy range. Also bin the pha if using spectral response
        if ebounds in locals():  # check if ebounds was defined
            pha_mask = (pha >= 0) & (pha <= ebounds.max()) & (pha >= ebounds.min())
            
            # bin the ADC channels
            self.pha = np.digitize(pha[pha_mask], ebounds)
            self.n_channels= len(self.rsp.ebounds) - 1
        else:
            pha_mask = (pha >= 0)
        
        # get the dead time fraction
        self.dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[pha_mask]

        # get the arrival time, in SECOND
        self.time = (hdu_evt['POLEVENTS'].data.field('TIME'))[pha_mask] - reference_time

        # now do the scattering angles
        scattering_angles = hdu_evt['POLEVENTS'].data.field('SA')[pha_mask]

        # clear the bad scattering angles
        scat_angle_mask = scattering_angles != -1

        self.scattering_angle_time = (hdu_evt['POLEVENTS'].data.field('TIME'))[scat_angle_mask] - reference_time
        self.scattering_angle_dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[scat_angle_mask]
        self.scattering_angles = scattering_angles[scat_angle_mask]

        # bin the scattering_angles
        if polrsp is not None:
                hdu_polrsp=fits.open(polrsp)
                samin = hdu_polrsp['SABOUNDS'].data.field('SA_MIN')
                samax = hdu_polrsp['SABOUNDS'].data.field('SA_MAX')
                scatter_bounds = np.append(samin, samax[-1])

                self.scattering_edges = scatter_bounds
                self.scattering_angles = np.digitize(self.scattering_angles, scatter_bounds)
                self.n_scattering_bins= len(self.scattering_edges) - 1

        else:
            self.scattering_edges = None
            self.scattering_angles = None