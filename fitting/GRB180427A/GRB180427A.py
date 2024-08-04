import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np
import os
np.seterr(all="ignore")
from threeML import *
silence_warnings()
# %matplotlib inline
set_threeML_style()


# reading polarization data from AstroSat CZTI and creating polarization plugin
trigger_time = 262521423.0
czti_polarization_ts = TimeSeriesBuilder.from_pol_polarization(name='czti_pol', polevents='/home/polpy/polpy_test/polpy/fitting/GRB180427A/cztipol_GRB180427A.pevt',
                                              polrsp='/home/polpy/polpy_test/polpy/fitting/GRB180427A/CZTI_POLRSP_EMIN_100_EMAX_1000_GRB180427A.prsp', specrsp=None,
                                               trigger_time=trigger_time)
czti_polarization_ts.set_background_interval("-150.--50","50.-150.")
czti_polarization_ts.set_active_time_interval('0.-22.')
czti_data = czti_polarization_ts.to_polarizationlike()
#effective area correction
#czti_data.fix_effective_area_correction(2.0)
czti_data.use_effective_area_correction(0.1,300.0)


# reading spectrum data from Fermi GBM and creating spectrum plugin
gbm_cat = FermiGBMBurstCatalog()
gbm_cat.query_sources('GRB180427442')
grb_info = gbm_cat.get_detector_information()["GRB180427442"]
gbm_detectors = grb_info["detectors"]
source_interval = grb_info["source"]["fluence"]
background_interval = grb_info["background"]["full"]
best_fit_model = grb_info["best fit model"]["fluence"]
print(source_interval,"\n", background_interval, "\n", gbm_detectors, "\n", type(gbm_detectors))
source_interval = '0.0-27.009000'
gbm_detectors = ['n4', 'n1', 'n6', 'b0']
dload = download_GBM_trigger_data("GRB180427442", detectors=gbm_detectors)
fluence_plugins = []
time_series = {}
#print(best_fit_model, "\n", source_interval, "\n", model.display())
for det in gbm_detectors:
    ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(det, cspec_or_ctime_file=dload[det]["cspec"], rsp_file=dload[det]["rsp"])
    ts_cspec.set_background_interval(*background_interval.split(","))
    ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)
    ts_tte = TimeSeriesBuilder.from_gbm_tte(
        det,
        tte_file=dload[det]["tte"],
        rsp_file=dload[det]["rsp"],
        restore_background=f"{det}_bkg.h5",
    )
    time_series[det] = ts_tte
    ts_tte.set_active_time_interval(source_interval)
    ts_tte.view_lightcurve(-40, 160)
    fluence_plugin = ts_tte.to_spectrumlike()
    if det.startswith("b"):
        fluence_plugin.set_active_measurements("250-30000")
    else:
        fluence_plugin.set_active_measurements("9-900")
    fluence_plugin.rebin_on_background(1.0)
    fluence_plugins.append(fluence_plugin)


#setting up spectrum model
band = Band()
band.xp.prior = Log_normal(mu=np.log(147),sigma=np.log(100))
band.xp.bounds = (None, None)
band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)
band.alpha.bounds = (-1.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-0.29, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)
band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.8,sigma=0.6, lower_bound=-7, upper_bound=-1.5)

#settting up polarization model
lp = LinearPolarization(60,47)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
lp.degree.value=60.
lp.angle.value=16.91

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB180427A',0,0, components = [sc])
combined_model = Model(ps)
datalist = DataList(*fluence_plugins,czti_data)

# Setting up sampler and running bayes
bayes = BayesianAnalysis(combined_model,datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(combined_model.free_parameters)
# wrapped[3] = 1
bayes.sampler.setup(n_live_points=100,
                           resume = False,
                           importance_nested_sampling=False,
                           verbose=True,
                           wrapped_params=wrapped,
                           chain_name='chains/synch_p2')
bayes.sample()
bayes.restore_median_fit()
bayes.results.write_to("AstroSat_CZTI_polarization_results_GRB180427A.fits", overwrite=True)
fig = display_spectrum_model_counts(bayes, min_rate=20)
fig.savefig("count_spec.pdf", bbox_inches="tight")
#display everthing in bayes.results
bayes.results.display()
cornerplot = bayes.results.corner_plot()
cornerplot.savefig("corner_plot.pdf", bbox_inches="tight")
a = bayes.raw_samples
np.shape(a)
deg = a[:,5]
plt.hist(deg)
degrot = np.copy(deg)
degrot[deg > 90] -= 180
plt.hist(degrot)
print("np.median(degrot)",np.median(degrot),"np.std(degrot)", np.std(degrot))
modualtioncurve = czti_data.display(show_model=True)
modualtioncurve.savefig("modulation_curve.pdf", bbox_inches="tight")