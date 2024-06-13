# test_pol_data_reader.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from polpy import PolData


class TestPolData(unittest.TestCase):

    @patch('pol_data_reader.fits.open')
    def test_initialization(self, mock_fits_open):
        # Mock for spectral response FITS file
        mock_hdu_spec = MagicMock()
        mock_hdu_spec['MATRIX'].data.field.side_effect = [
            np.array([10, 20]),  # ENERG_LO
            np.array([15, 25]),  # ENERG_HI
            np.array([[1, 2], [3, 4]])  # MATRIX
        ]

        # Mock for polarization events FITS file
        mock_hdu_evt = MagicMock()
        mock_hdu_evt['POLEVENTS'].header = {
            'TELESCOP': 'TestMission',
            'INSTRUME': 'TestInstrument'
        }
        mock_hdu_evt['POLEVENTS'].data.field.side_effect = [
            np.array([5, 15, 25]),  # ENERGY
            np.array([0.1, 0.2, 0.3]),  # DEADFRAC
            np.array([100, 200, 300]),  # TIME
            np.array([45, 90, -1])  # SA
        ]

        # Mock FITS open
        mock_fits_open.side_effect = [mock_hdu_spec, mock_hdu_evt]

        # Initialize PolData
        pol_data = PolData('polevents.fits', 'specrsp.fits', reference_time=100)

        # Assertions for spectral response
        self.assertIsInstance(pol_data.rsp, MagicMock)
        self.assertEqual(pol_data.mission, 'TestMission')
        self.assertEqual(pol_data.instrument, 'TestInstrument')
        np.testing.assert_array_equal(pol_data.pha, np.array([1, 2]))
        np.testing.assert_array_equal(pol_data.time, np.array([0, 100, 200]) - 100)
        np.testing.assert_array_equal(pol_data.dead_time_fraction, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(pol_data.scattering_angles, np.array([1, 2]))

    @patch('pol_data_reader.fits.open')
    def test_initialization_with_polrsp(self, mock_fits_open):
        # Mock for spectral response FITS file
        mock_hdu_spec = MagicMock()
        mock_hdu_spec['MATRIX'].data.field.side_effect = [
            np.array([10, 20]),  # ENERG_LO
            np.array([15, 25]),  # ENERG_HI
            np.array([[1, 2], [3, 4]])  # MATRIX
        ]

        # Mock for polarization events FITS file
        mock_hdu_evt = MagicMock()
        mock_hdu_evt['POLEVENTS'].header = {
            'TELESCOP': 'TestMission',
            'INSTRUME': 'TestInstrument'
        }
        mock_hdu_evt['POLEVENTS'].data.field.side_effect = [
            np.array([5, 15, 25]),  # ENERGY
            np.array([0.1, 0.2, 0.3]),  # DEADFRAC
            np.array([100, 200, 300]),  # TIME
            np.array([45, 90, -1])  # SA
        ]

        # Mock for polarization response FITS file
        mock_hdu_pol = MagicMock()
        mock_hdu_pol['SABOUNDS'].data.field.side_effect = [
            np.array([0, 30]),  # SA_MIN
            np.array([30, 60])  # SA_MAX
        ]

        # Mock FITS open
        mock_fits_open.side_effect = [mock_hdu_spec, mock_hdu_evt, mock_hdu_pol]

        # Initialize PolData
        pol_data = PolData('polevents.fits', 'specrsp.fits', 'polrsp.fits', reference_time=100)

        # Assertions for polarization response
        np.testing.assert_array_equal(pol_data.scattering_edges, np.array([0, 30, 60]))
        np.testing.assert_array_equal(pol_data.scattering_angles, np.array([1, 2]))
    
  
unittest.main()
