import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from polpy import PolResponse, FastGridInterpolate

class TestPolResponse(unittest.TestCase):

    @patch('pol_response.fits.open')
    def test_initialization(self, mock_fits_open):
        # Mock for polarisation response FITS file
        mock_hdu_pol = MagicMock()
        mock_hdu_pol['INEBOUNDS'].data.field.side_effect = [
            np.array([10, 20], dtype=np.float64),  # ENERG_LO
            np.array([15, 25], dtype=np.float64)  # ENERG_HI
        ]
        mock_hdu_pol['INPAVALS'].data.field.side_effect = [
            np.array([0, 45, 90], dtype=np.float64)  # PA_IN
        ]
        mock_hdu_pol['SABOUNDS'].data.field.side_effect = [
            np.array([0, 30], dtype=np.float64),  # SA_MIN
            np.array([30, 60], dtype=np.float64)  # SA_MAX
        ]
        mock_hdu_pol['SPECRESP POLMATRIX'].data = np.array([
            [0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]
        ])
        mock_hdu_pol['SPECRESP UNPOLMATRIX'].data = np.array([
            [0.7, 0.8, 0.9], 
            [1.0, 1.1, 1.2]
        ])

        # Mock fits.open to return the mock_hdu_pol
        mock_fits_open.side_effect = [mock_hdu_pol]

        # Initialize PolResponse
        pol_response = PolResponse('response.prsp')

        # Assertions
        np.testing.assert_array_equal(pol_response.ene_lo, np.array([10, 20], dtype=np.float64))
        np.testing.assert_array_equal(pol_response.ene_hi, np.array([15, 25], dtype=np.float64))
        np.testing.assert_array_equal(pol_response.energy_mid, np.array([12.5, 22.5], dtype=np.float64))
        self.assertEqual(pol_response.n_scattering_bins, 2)
        np.testing.assert_array_equal(pol_response.scattering_bins, np.array([15, 45], dtype=np.float64))
        np.testing.assert_array_equal(pol_response.scattering_bins_lo, np.array([0, 30], dtype=np.float64))
        np.testing.assert_array_equal(pol_response.scattering_bins_hi, np.array([30, 60], dtype=np.float64))

        # Check interpolators
        for interpolator in pol_response.interpolators:
            self.assertIsInstance(interpolator, FastGridInterpolate)

    @patch('pol_response.fits.open')
    def test_interpolators(self, mock_fits_open):
        # Mock for polarisation response FITS file
        mock_hdu_pol = MagicMock()
        mock_hdu_pol['INEBOUNDS'].data.field.side_effect = [
            np.array([10, 20], dtype=np.float64),  # ENERG_LO
            np.array([15, 25], dtype=np.float64)  # ENERG_HI
        ]
        mock_hdu_pol['INPAVALS'].data.field.side_effect = [
            np.array([0, 45, 90], dtype=np.float64)  # PA_IN
        ]
        mock_hdu_pol['SABOUNDS'].data.field.side_effect = [
            np.array([0, 30], dtype=np.float64),  # SA_MIN
            np.array([30, 60], dtype=np.float64)  # SA_MAX
        ]
        mock_hdu_pol['SPECRESP POLMATRIX'].data = np.array([
            [0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]
        ])
        mock_hdu_pol['SPECRESP UNPOLMATRIX'].data = np.array([
            [0.7, 0.8, 0.9], 
            [1.0, 1.1, 1.2]
        ])

        # Mock fits.open to return the mock_hdu_pol
        mock_fits_open.side_effect = [mock_hdu_pol]

        # Initialize PolResponse
        pol_response = PolResponse('response.prsp')

        # Assertions for interpolators
        for interpolator in pol_response.interpolators:
            grid, values = interpolator._grid, interpolator._values
            np.testing.assert_array_equal(grid[0], np.array([12.5, 22.5], dtype=np.float64))
            np.testing.assert_array_equal(grid[1], np.array([0, 45, 90], dtype=np.float64))
            np.testing.assert_array_equal(grid[2], np.array([0., 100.], dtype=np.float64))
            self.assertEqual(values.shape, (2, 3, 2))  # Shape of the pol_matrix for each bin

if __name__ == '__main__':
    unittest.main()
