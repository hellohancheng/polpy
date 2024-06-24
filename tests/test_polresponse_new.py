import numpy as np
from astropy.io import fits
from polpy import PolResponse  # Replace 'your_module' with the actual module name where PolResponse is defined

def test_polresponse(response_file):
    # Initialize the PolResponse object with the given file
    pol_response = PolResponse(response_file)

    # Check if the energy low bounds are correctly set
    with fits.open(response_file) as hdu:
        expected_ene_lo = np.array(hdu['INEBOUNDS'].data.field('ENERG_LO'), dtype=np.float64)
        expected_ene_hi = np.array(hdu['INEBOUNDS'].data.field('ENERG_HI'), dtype=np.float64)
        expected_energy_mid = (expected_ene_lo + expected_ene_hi) / 2
        expected_samin = np.array(hdu['SABOUNDS'].data.field('SA_MIN'), dtype=np.float64)
        expected_samax = np.array(hdu['SABOUNDS'].data.field('SA_MAX'), dtype=np.float64)
        expected_bins = np.append(expected_samin, expected_samax[-1])
        expected_bin_center = 0.5 * (expected_bins[:-1] + expected_bins[1:])

    assert np.allclose(pol_response.ene_lo, expected_ene_lo), "Energy low bounds do not match"
    assert np.allclose(pol_response.ene_hi, expected_ene_hi), "Energy high bounds do not match"
    assert np.allclose(pol_response.energy_mid, expected_energy_mid), "Energy mid points do not match"
    assert np.allclose(pol_response.scattering_bins, expected_bin_center), "Scattering bin centers do not match"
    assert np.allclose(pol_response.scattering_bins_lo, expected_bins[:-1]), "Scattering bin low bounds do not match"
    assert np.allclose(pol_response.scattering_bins_hi, expected_bins[1:]), "Scattering bin high bounds do not match"

    # Print the results if all tests pass
    print("All tests passed!")

# Path to the response file from the data folder
response_file = 'data/POLAR_170114A.prsp'

# Run the test
test_polresponse(response_file)
