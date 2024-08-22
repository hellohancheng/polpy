#!/usr/bin/env python

#from mpi4py import MPI
from pathlib import Path
#rank = comm.Get_rank()
import numpy as np
#import pymultinest
import warnings
from astropy.io import fits
from threeML import *

# data and responses
data_path = Path(__file__).parent.parent.joinpath("data")
polevents = data_path.joinpath("POLAR_161218B.pevt")
polrsp = data_path.joinpath("POLAR_161218B.prsp")
specrsp = data_path.joinpath("POLAR_161218B.rmfarf")


trigger_time = 1482049960.65

# spectral
polar = TimeSeriesBuilder.from_pol_spectrum('polar_spec', polevents.as_posix(), specrsp.as_posix(),
                                            trigger_time=trigger_time)


#threeML_config['lightcurve']['lightcurve color'] = '#07AE44'
polar.set_active_time_interval('-0.0-25.10')
polar.set_background_interval('-20.0--0.0','50.0-100.0')
polar.create_time_bins(start=-20.0, stop=90.0, method='constant', dt=0.1)

polar_spec = polar.to_spectrumlike()
polar_spec.set_active_measurements('30-750')

polar_spec.use_effective_area_correction(0.7, 1.3)

# polarisation 
polar_polarization_ts = TimeSeriesBuilder.from_polarization('polar_pol', polevents.as_posix(), polrsp.as_posix(),
                                              specrsp, trigger_time=trigger_time)


polar_polarization_ts.set_background_interval('-20--0.0','50-100')
polar_polarization_ts.set_active_time_interval('0.0-25.10')
polar_data = polar_polarization_ts.to_polarizationlike()
polar_data.use_effective_area_correction(0.7, 1.5)



gbm_cat = FermiGBMBurstCatalog()
gbm_cat.query_sources('GRB161218356')
grb_info = gbm_cat.get_detector_information()["GRB161218356"]
gbm_detectors=grb_info['detectors']
source_interval = grb_info["source"]["fluence"]
background_interval = grb_info["background"]["full"]
best_fit_model = grb_info["best fit model"]["fluence"]
model = gbm_cat.get_model(best_fit_model, "fluence")["GRB161218356"]

dl = download_GBM_trigger_data('bn161218356',gbm_detectors)

fluence_plugins = []
time_series = {}
for det in gbm_detectors:
    ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
        det, cspec_or_ctime_file=dl[det]["cspec"], rsp_file=dl[det]["rsp"]
    )

    ts_cspec.set_background_interval(*background_interval.split(","))
    ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)

    ts_tte = TimeSeriesBuilder.from_gbm_tte(
        det,
        tte_file=dl[det]["tte"],
        rsp_file=dl[det]["rsp"],
        restore_background=f"{det}_bkg.h5",
    )

    time_series[det] = ts_tte

    ts_tte.set_active_time_interval(source_interval)

    ts_tte.view_lightcurve(-40, 120)

    fluence_plugin = ts_tte.to_spectrumlike()

    if det.startswith("b"):
        fluence_plugin.set_active_measurements("250-30000")

    else:
        fluence_plugin.set_active_measurements("9-900")

    fluence_plugin.rebin_on_background(1.0)

    fluence_plugins.append(fluence_plugin)

#Just do the spectral fit first
# modeling setup
band = Band()
band.xp.prior = Log_normal(mu=np.log(300),sigma=np.log(100))
band.xp.bounds = (None, None)

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

band.alpha.bounds = (-1.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-3.,sigma=0.6, lower_bound=-5, upper_bound=-1.5)

# pollack setup
lp = LinearPolarization(10,10)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
lp.degree.value=0.
lp.angle.value=10

sc =SpectralComponent('synch', band, lp)
ps = PointSource('polar_GRB',0,0,components=[sc])

model = Model(ps)
datalist = DataList(polar_data, polar_spec, *fluence_plugins)

# BAYES
bayes = BayesianAnalysis(model,datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(model.free_parameters)
wrapped[5] = 1

bayes.sampler.setup(n_live_points=1000,
                           resume = False,
                           importance_nested_sampling=False,
                           verbose=True,
                           wrapped_params=wrapped,
                           chain_name='chains/synch_p2')

bayes.sample()

#if rank == 0:
bayes.results.write_to("fit_results_161218B.fits", overwrite=True)
bayes.restore_median_fit()
fig = display_spectrum_model_counts(bayes)
    
fig.savefig("count_spec_161218B.pdf", bbox_inches="tight")

file=load_analysis_results('/home/polpy/polpy_test/polpy/examples/fit_results_161218B.fits')
fig=file.corner_plot()
fig.savefig('corner_plot_161218B.png')

modulationcurve = polar_data.display(show_model=True,min_rate=20)
fig=modulationcurve.savefig()