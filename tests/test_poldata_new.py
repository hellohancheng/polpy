import numpy as np
from astropy.io import fits
from polpy import PolData

def test_poldata(polevents_file, polrsp_file, reference_time=0.):
    # Initialize the PolData object with the given files
    pol_data = PolData(polevents_file, polrsp=polrsp_file, reference_time=reference_time)

    # Open the polar events file and extract expected values
    with fits.open(polevents_file) as hdu_evt:
        expected_mission = hdu_evt['POLEVENTS'].header['TELESCOP']
        expected_instrument = hdu_evt['POLEVENTS'].header['INSTRUME']
        expected_pha = hdu_evt['POLEVENTS'].data.field('ENERGY')
        expected_dead_time_fraction = hdu_evt['POLEVENTS'].data.field('DEADFRAC')
        expected_time = hdu_evt['POLEVENTS'].data.field('TIME') - reference_time
        expected_scattering_angles = hdu_evt['POLEVENTS'].data.field('SA')

        # Filter non-zero ADC channels
        idx = expected_pha >= 0
        expected_pha = expected_pha[idx]
        expected_dead_time_fraction = expected_dead_time_fraction[idx]
        expected_time = expected_time[idx]

        # Filter valid scattering angles
        idx = expected_scattering_angles != -1
        expected_scattering_angles = expected_scattering_angles[idx]
        expected_scattering_angle_time = (hdu_evt['POLEVENTS'].data.field('TIME'))[idx] - reference_time
        expected_scattering_angle_dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[idx]

    # Open the polar response file and extract expected values
    with fits.open(polrsp_file) as hdu_pol:
        samin = hdu_pol['SABOUNDS'].data.field('SA_MIN')
        samax = hdu_pol['SABOUNDS'].data.field('SA_MAX')
        scatter_bounds = np.append(samin, samax[-1])

    assert pol_data.mission == expected_mission, "Mission does not match"
    assert pol_data.instrument == expected_instrument, "Instrument does not match"
    assert np.allclose(pol_data.pha, np.digitize(expected_pha, pol_data.rsp.ebounds)), "PHA bins do not match"
    assert np.allclose(pol_data.time, expected_time), "Event times do not match"
    assert np.allclose(pol_data.dead_time_fraction, expected_dead_time_fraction), "Dead time fractions do not match"
    assert np.allclose(pol_data.scattering_angles, np.digitize(expected_scattering_angles, pol_data.scattering_bins)), "Scattering angles bins do not match"
    assert np.allclose(pol_data.scattering_angle_time, expected_scattering_angle_time), "Scattering angle times do not match"
    assert np.allclose(pol_data.scattering_angle_dead_time_fraction, expected_scattering_angle_dead_time_fraction), "Scattering angle dead time fractions do not match"
    assert np.allclose(pol_data.scattering_bins, scatter_bounds), "Scattering bins do not match"

    # Print the results if all tests pass
    print("All tests passed!")

# Paths to the event and polar response files in the data folder
polevents_file = 'data/POLAR_170114A.pevt'
polrsp_file = 'data/POLAR_170114A.prsp'

# Run the test
test_poldata(polevents_file, polrsp_file)
