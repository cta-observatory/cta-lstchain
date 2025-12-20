#!/usr/bin/env python

"""

This script reads in the DL1 datacheck files for all runs which correspond
to a given night summary DL1 datacheck file, i.e. DL1_datacheck_YYYYMMDD.h5,
calculates some subrun-wise parameters related to the intensity spectrum of
cosmic rays (dR/dI, differential rate per unit of intensity), and writes
them in a new "cosmics_intensity_spectrum" table in the night summary file.

These parameters are very useful to assess the quality of data, and also
may eventually be used to apply corrections to the absolute light
calibration for nights with smaller-than-optimal atmospheric transmissivity

Input files will be the run-wise datacheck files in the path provided
through the input_dir command-line argument

"""

import glob
import logging
import json

import tables
import gc
import sys
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.stats import chi2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="compute Cherenkov transparency")

parser.add_argument('-u', '--update-datacheck-file', dest='file_to_update',
                    type=str, required=True,
                    help='(a DL1_datacheck_YYYYMMDD.h5 file)')

parser.add_argument('-d', '--input-dir', type=str, required=True,
                    help='Path to directory containing run-wise datacheck '
                         'files')


def main():
    args = parser.parse_args()
    output_file = args.file_to_update

    input_dir = args.input_dir
    files = glob.glob(input_dir+'/datacheck_dl1_LST-1.Run?????.h5')
    files.sort()

    if len(files) == 0:
        logging.error(f'No run-wise datacheck files found in {input_dir}!')
        sys.exit(-1)

    a = tables.open_file(files[0])

    hist_intensity_binning = \
        a.root.dl1datacheck.histogram_binning.col('hist_intensity')[0]
    hist_intensity_bin_centers = (hist_intensity_binning[1:] *
                                  hist_intensity_binning[:-1]) ** 0.5
    hist_delta_t_binning = \
        a.root.dl1datacheck.histogram_binning.col('hist_delta_t')[0]
    hist_delta_t_bin_centers = 0.5*(hist_delta_t_binning[1:] +
                                    hist_delta_t_binning[:-1])

    a.close()

    gc.disable()

    num_cosmics = []
    num_cleaned_cosmics = []
    num_pedestals = []
    num_cleaned_pedestals = []
    num_flatfield = []
    num_cleaned_flatfield = []
    num_star_affected_pixels = []
    diffuse_nsb_mean = []
    diffuse_nsb_std = []
    mean_alt_tel = []
    mean_az_tel = []
    tel_ra = []
    tel_dec = []
    dragon_time = []
    elapsed_time = []
    corrected_elapsed_time = []
    subrun_index = []
    run = []
    intensity_hist = []
    delta_t_hist = []
    picture_thresh = []
    boundary_thresh = []

    for file in tqdm(files):
        a = tables.open_file(file)

        if 'pedestals' in a.root.dl1datacheck:
            if len(a.root.dl1datacheck.cosmics.col('subrun_index')) < len(
                    a.root.dl1datacheck.pedestals.col('subrun_index')):
                print('    Warning: skipped', file,
                      'because it had fewer entries in cosmics table than in '
                      'pedestals table!')
                a.close()
                continue

        if 'flatfield' in a.root.dl1datacheck:
            if len(a.root.dl1datacheck.cosmics.col('subrun_index')) < len(
                    a.root.dl1datacheck.flatfield.col('subrun_index')):
                print('    Warning: skipped', file,
                      'because it had fewer entries in cosmics table than in '
                      'flatfield table!')
                a.close()
                continue

        num_cosmics.append(a.root.dl1datacheck.cosmics.col('num_events'))
        num_cleaned_cosmics.append(
            a.root.dl1datacheck.cosmics.col('num_cleaned_events'))
        mean_alt_tel.append(a.root.dl1datacheck.cosmics.col('mean_alt_tel'))
        mean_az_tel.append(a.root.dl1datacheck.cosmics.col('mean_az_tel'))
        tel_ra.append(a.root.dl1datacheck.cosmics.col('tel_ra'))
        tel_dec.append(a.root.dl1datacheck.cosmics.col('tel_dec'))

        dragon_time.append(
            np.mean(a.root.dl1datacheck.cosmics.col('dragon_time'), axis=1))
        elapsed_time.append(a.root.dl1datacheck.cosmics.col('elapsed_time'))

        # Datachecks processed before 2025 do not contain config information.
        # Since the reprocessed files correspond to runs with increased cleaning
        # settings and the unprocessed files to tailcut 84, we hardcode the values
        # of the picture and boundary threshold for these unprocessed runs
        try:
            config = json.loads(a.root.dl1datacheck.cosmics.attrs['config'])
            picture_thresh.append(
                config['tailcuts_clean_with_pedestal_threshold']['picture_thresh']
            )
            boundary_thresh.append(
                config['tailcuts_clean_with_pedestal_threshold']['boundary_thresh']
            )
        except:
            picture_thresh.append(8)
            boundary_thresh.append(4)

        # Compute corrected elapsed time, i.e. remove periods of inactive daq
        # (e.g. busy spikes). For this we use the timestamps (50 per subrun)
        # stored in the datacheck files. Note that it is 50 timestamps taken
        # at equal intervals of number of events - and the last one is not the
        # last event in the run, typically it is ~1000 events earlier.

        tdiff = np.diff(a.root.dl1datacheck.cosmics.col('dragon_time'), axis=1)
        # Exclude the 10 largest time jumps per subrun (among which will be the
        # possible daq hiccups) to calculate a reasonable threshold to
        # identify the jumps (a gap of 5 times the maximum "ordinary" gap in
        # the whole run):
        hiccup_threshold = 5 * np.max(np.sort(tdiff, axis=1)[:, :-10])
        tdiff_mean = np.nanmean(np.where(tdiff < hiccup_threshold, tdiff,
                                         np.nan),  axis=1)
        # Replace too big tdiffs by the mean tdiff in the subrun:
        elapsed_time_correction = np.sum(np.where(tdiff > hiccup_threshold,
                                                  tdiff_mean[:, None] - tdiff,
                                                  0),
                                         axis=1)
        newtime = (a.root.dl1datacheck.cosmics.col('elapsed_time') +
                   elapsed_time_correction)
        corrected_elapsed_time.append(newtime)

        subrun_index.append(a.root.dl1datacheck.cosmics.col('subrun_index'))
        run.append(int(file[file.find('.Run') + 4:file.find('.Run') + 9]))

        num_subruns = len(a.root.dl1datacheck.cosmics.col('subrun_index'))

        if 'pedestals' in a.root.dl1datacheck:
            if len(a.root.dl1datacheck.pedestals.col(
                    'subrun_index')) == num_subruns:  # all subruns present
                num_pedestals.append(
                    a.root.dl1datacheck.pedestals.col('num_events'))
                num_cleaned_pedestals.append(
                    a.root.dl1datacheck.pedestals.col('num_cleaned_events'))

                starmask = a.root.dl1datacheck.pedestals.col('num_nearby_stars') < 1
                diffuse_nsb_mean.append(np.nanmean(np.where(starmask,
                                                            a.root.dl1datacheck.pedestals.col(
                                                                'charge_mean'),
                                                            np.nan), axis=1))
                diffuse_nsb_std.append(np.nanmean(np.where(starmask,
                                                           a.root.dl1datacheck.pedestals.col(
                                                               'charge_stddev'),
                                                           np.nan), axis=1))

            else:  # some subruns missing. Fill the existing ones.
                dummy = np.zeros(num_subruns)
                dummy2 = np.zeros(num_subruns)
                for jj, nn, nn2 in zip(
                        a.root.dl1datacheck.pedestals.col('subrun_index'),
                        a.root.dl1datacheck.pedestals.col('num_events'),
                        a.root.dl1datacheck.pedestals.col('num_cleaned_events')):
                    new_index = np.where(a.root.dl1datacheck.cosmics.col('subrun_index') == jj)[0][0]
                    dummy[new_index] = nn
                    dummy2[new_index] = nn2
                num_pedestals.append(dummy)
                num_cleaned_pedestals.append(dummy2)

                dummy = np.array(num_subruns * [np.nan])
                dummy2 = np.array(num_subruns * [np.nan])
                for jj, nns, ch1, ch2 in zip(
                        a.root.dl1datacheck.pedestals.col('subrun_index'),
                        a.root.dl1datacheck.pedestals.col('num_nearby_stars'),
                        a.root.dl1datacheck.pedestals.col('charge_mean'),
                        a.root.dl1datacheck.pedestals.col('charge_stddev')):
                    starmask = nns < 1
                    new_index = np.where(a.root.dl1datacheck.cosmics.col('subrun_index') == jj)[0][0]
                    dummy[new_index] = np.nanmean(np.where(starmask, ch1, np.nan))
                    dummy2[new_index] = np.nanmean(np.where(starmask, ch2, np.nan))
                diffuse_nsb_mean.append(dummy)
                diffuse_nsb_std.append(dummy2)

        else:  # no pedestals table at all
            num_pedestals.append(np.zeros(num_subruns))
            num_cleaned_pedestals.append(np.zeros(num_subruns))
            diffuse_nsb_mean.append(np.array(num_subruns * [np.nan]))
            diffuse_nsb_std.append(np.array(num_subruns * [np.nan]))

        if 'flatfield' in a.root.dl1datacheck:
            if len(a.root.dl1datacheck.flatfield.col(
                    'subrun_index')) == num_subruns:  # all subruns present
                num_flatfield.append(
                    a.root.dl1datacheck.flatfield.col('num_events'))
                num_cleaned_flatfield.append(
                    a.root.dl1datacheck.flatfield.col('num_cleaned_events'))
            else:  # some subruns missing. Fill the existing ones.
                dummy = np.zeros(num_subruns)
                dummy2 = np.zeros(num_subruns)
                for jj, nn, nn2 in zip(
                        a.root.dl1datacheck.flatfield.col('subrun_index'),
                        a.root.dl1datacheck.flatfield.col('num_events'),
                        a.root.dl1datacheck.flatfield.col('num_cleaned_events')):
                    new_index = np.where(a.root.dl1datacheck.cosmics.col('subrun_index') == jj)[0][0]
                    dummy[new_index] = nn
                    dummy2[new_index] = nn2
                num_flatfield.append(dummy)
                num_cleaned_flatfield.append(dummy2)

        else:  # no flatfield table at all
            num_flatfield.append(np.zeros(num_subruns))
            num_cleaned_flatfield.append(np.zeros(num_subruns))

        num_star_affected_pixels.append(
            np.sum(a.root.dl1datacheck.cosmics.col('num_nearby_stars') > 0, axis=1))

        intensity_hist.append(a.root.dl1datacheck.cosmics.col('hist_intensity'))
        delta_t_hist.append(a.root.dl1datacheck.cosmics.col('hist_delta_t'))
        a.close()

    gc.enable()

    # x log10(intensity/p.e.); y: rate (s-1)
    #
    # min_intensity (p.e.) : if intensity is lower han this (indicating "fake
    # peak" from satellites or whatever),
    # then go for the second peak, not the first!
    #

    # Range of bins of the rate histograms to perform the power-law fit of
    # dR/dI vs. I  (I=intensity)
    bin_start = 30  # min
    bin_end = 35  # max
    intensity_mean = np.round((hist_intensity_binning[bin_start] *
                               hist_intensity_binning[bin_end]) ** 0.5)
    # geometric mean of the fit range extremes (it is, rounded, = 422 p.e.)
    rate_intensity_min_to_max = []  # = adding up rates in bins defined below

    # The all_ prefix indicates the arrays have one entry per subrun
    all_elapsed_time = []
    all_corrected_elapsed_time = []
    all_dt_exp_index = []
    all_cosmic_rates = []
    all_cosmic_cleaned_rates = []
    all_pedestal_rates = []
    all_flatfield_rates = []
    all_alt_tel = []
    all_az_tel = []
    all_tel_ra = []
    all_tel_dec = []
    all_runs = []
    all_subruns = []
    all_star_affected_pixels = []
    all_dragon_time = []
    all_diffuse_nsb_mean = []
    all_diffuse_nsb_std = []
    all_picture_thresh = []
    all_boundary_thresh = []

    all_intensity_50 = []
    all_peak_intensity = []
    all_cosmics_peak_rate = []
    all_threshold_anomaly_detected = []

    all_fit_params = []
    all_fit_errors = []
    all_p_value = []

    num_total = len(run)

    intensity_bin_widths = np.diff(hist_intensity_binning)

    # Loop over runs:
    for ncosm, ncosm2, nped, nff, x, t, t2, dth, alt, az, ra, dec, r, sr, dt, \
        p_th, b_th, nsap, dnsbm, dnsbs in tqdm(zip(
            num_cosmics, num_cleaned_cosmics, num_pedestals, num_flatfield,
            intensity_hist, elapsed_time, corrected_elapsed_time, delta_t_hist,
            mean_alt_tel, mean_az_tel,
            tel_ra, tel_dec, run, subrun_index, dragon_time,
            picture_thresh, boundary_thresh,
            num_star_affected_pixels, diffuse_nsb_mean, diffuse_nsb_std),
        total=num_total):

        all_elapsed_time.extend(t)
        all_corrected_elapsed_time.extend(t2)

        # Compute the approximate index of the exponential part of the delta_t
        # histograms (one per subrun). For robustness, instead of a fit we
        # just use the peak value of the histogram and the first point after
        # it for which the number of events is 10 times lower:
        factor = 0.1
        for dd in dth:
            max_dd = np.max(dd)
            logpeak = np.log(max_dd)
            peakpos = np.argmax(dd)
            # first bin after peakpos with < factor*peak value. If this is not
            # reached anywhere within the histogram (because of too slow
            # descent, i.e. for very small "external" Poisson rates), just
            # use the last bin of the histogram for the calculation. We
            # cannot take always the last bin, since the exponential behaviour
            # typically does not hold for the whole range of the histogram.
            index2 = -1
            if np.min(dd[peakpos:]) < factor * max_dd:
                index2 = peakpos + np.where(dd[peakpos:] <
                                            factor * max_dd)[0][0]
            slope = ((logpeak - np.log(dd[index2])) /
                     (hist_delta_t_bin_centers[index2] -
                      hist_delta_t_bin_centers[peakpos]))
            all_dt_exp_index.append(slope * 1000)  # *1000 is /ms to => /s

        all_cosmic_rates.extend(ncosm / t2)
        all_cosmic_cleaned_rates.extend(ncosm2 / t2)
        all_pedestal_rates.extend(nped / t2)
        all_flatfield_rates.extend(nff / t2)
        all_star_affected_pixels.extend(nsap)

        all_alt_tel.extend(alt)
        all_az_tel.extend(az)
        all_tel_ra.extend(ra)
        all_tel_dec.extend(dec)

        all_runs.extend(len(sr) * [r])
        all_subruns.extend(sr)
        all_dragon_time.extend(dt)
        all_diffuse_nsb_mean.extend(dnsbm)
        all_diffuse_nsb_std.extend(dnsbs)

        all_picture_thresh.extend(len(sr) * [p_th])
        all_boundary_thresh.extend(len(sr) * [b_th])

        rate_vs_intensity = x / t2[:, None]
        delta_rate_vs_intensity = x ** 0.5 / t2[:, None]

        for rvi, delta_rvi in zip(rate_vs_intensity, delta_rate_vs_intensity):

            diff_rvi = rvi / intensity_bin_widths
            delta_diff_rvi = delta_rvi / intensity_bin_widths

            xxx = np.logspace(1, 4, 1200)
            yyy = np.interp(np.log10(xxx), np.log10(hist_intensity_bin_centers),
                            diff_rvi)

            x50, ymax, x100 = find50(xxx, yyy)
            if x50 < 25:
                all_threshold_anomaly_detected.append(True)
                x50, ymax, x100 = find50(xxx, yyy, 25)
                # recalculate, try to avoid fake peak (not always possible)
            else:
                all_threshold_anomaly_detected.append(False)

            all_intensity_50.append(x50)
            all_peak_intensity.append(x100)
            all_cosmics_peak_rate.append(ymax)

            try:
                xfit = np.log(hist_intensity_bin_centers)[bin_start:bin_end]
                # Use a reference energy close to the decorrelation energy
                # (intensity_mean is the geometrical mean of the fit range's
                # extremes, =422 p.e. for the range defined above):
                xfit -= np.log(intensity_mean)
                yfit = diff_rvi[bin_start:bin_end]
                yerrfit = delta_diff_rvi[bin_start:bin_end]
                params, pcov, info, _, _ = curve_fit(expfunc, xfit, yfit,
                                                     sigma=yerrfit,
                                                     full_output=True)
                all_fit_params.append(params)
                all_fit_errors.append([pcov[0, 0]**0.5, pcov[1, 1]**0.5])
                chisq = np.sum(info['fvec'] ** 2)
                all_p_value.append(1 - chi2.cdf(chisq, (bin_end - bin_start - 2)))
            except Exception:
                all_fit_params.append([np.nan, np.nan])
                all_fit_errors.append([np.nan, np.nan])
                all_p_value.append(np.nan)

        rate_intensity_min_to_max.extend(
            np.sum(rate_vs_intensity[:, bin_start:bin_end], axis=1))

    all_elapsed_time = np.array(all_elapsed_time)
    all_corrected_elapsed_time = np.array(all_corrected_elapsed_time)
    all_dt_exp_index = np.array(all_dt_exp_index)
    all_cosmic_rates = np.array(all_cosmic_rates)
    all_cosmic_cleaned_rates = np.array(all_cosmic_cleaned_rates)
    # all_pedestal_rates = np.array(all_pedestal_rates)
    # all_flatfield_rates = np.array(all_flatfield_rates)
    all_alt_tel = np.array(all_alt_tel)
    all_runs = np.array(all_runs)
    all_subruns = np.array(all_subruns)
    all_star_affected_pixels = np.array(all_star_affected_pixels)
    all_dragon_time = np.array(all_dragon_time)
    # all_diffuse_nsb_mean = np.array(all_diffuse_nsb_mean)
    all_diffuse_nsb_std = np.array(all_diffuse_nsb_std)
    all_intensity_50 = np.array(all_intensity_50)
    all_peak_intensity = np.array(all_peak_intensity)
    all_cosmics_peak_rate = np.array(all_cosmics_peak_rate)
    all_threshold_anomaly_detected = np.array(all_threshold_anomaly_detected)
    all_fit_params = np.array(all_fit_params)
    all_fit_errors = np.array(all_fit_errors)
    all_p_value = np.array(all_p_value)

    all_yyyymmdd = time_to_yyyymmdd(all_dragon_time)

    all_coszd = np.cos(np.pi / 2 - all_alt_tel)

    #
    # Cosmics differential rate vs. log(image intensity / 422 p.e.) was fitted
    # to expfunc, with (par0, par1) parameters
    # That is, dR/dI = par0 * (intensity/422)**par1
    #
    # Zenith dependence of par0 and par1 determined from v0.9 analysis of runs
    # 10800-12000 (good quality data):
    # par0 = pol2(cos_zd, -0.47572145,  3.90206282, -1.5625303)
    # par1 = pol2(cos_zd, -2.89253876,  0.99443457, -0.34012985)
    #

    p0a = -0.44751321
    p0b = 3.62502037
    p0c = -1.43611437

    p1a = -2.89253919
    p1b = 0.99443581
    p1c = -0.34013068

    # We compute versions of the parameters "corrected to ZD=0", so that we can
    # use the result as a
    # zenith-independent proxy for data quality in terms of light yield.
    #

    # Corrections for all subruns (factor to convert to ZD=0 equivalent):
    par0_at_zenith = pol2(1, p0a, p0b, p0c)
    par0_correction = par0_at_zenith / pol2(all_coszd, p0a, p0b, p0c)

    par1_at_zenith = pol2(1, p1a, p1b, p1c)
    par1_correction = par1_at_zenith - pol2(all_coszd, p1a, p1b, p1c)

    all_corrected_fit_params = np.array([all_fit_params[:, 0] * par0_correction,
                                         all_fit_params[:, 1] + par1_correction]).T
    all_corrected_fit_errors = np.array([all_fit_errors[:, 0] * par0_correction,
                                         all_fit_errors[:, 1]]).T

    # Corrections for all subruns (cleaning dependent correction)
    # The intensity parameter depends on the cleaning settings. Consequently,
    # the derived CR intensity spectrum and the associated parameters (par0 and par1)
    # also depend on the cleaning applied to the shower images. To enable a consistent
    # comparison between runs with different cleaning settings, the effect of cleaning
    # on the CR intensity spectrum parameters must be corrected.
    # We apply an empirical correction to par0 and par1 derived from the good-quality
    # data (i.e. 20221118 - 20230214). The correction is estimated separately on par0
    # and par1 for each cleaning setting. A single correction value is applied to
    # all subruns processed with the same image cleaning.
    # The correction is defined such that the corrected par0 and par1 values match
    # the values obtained from runs with the tailcut 8 and 4, which is used for
    # the largest fraction of the data.
    # The correction for cleaning setting i is:
    #     mean(par) at tailcut 84  -  mean(par) at tailcut_i
    # Outliers for tailcut 8 and 4 were removed via 3-sigma clipping.
    par0_vs_cleaning = np.array([
        1.733,     # tailcut84
        1.578,     # tailcut1005
        1.465,     # tailcut1206
        1.390,     # tailcut1407
        1.264,     # tailcut1608
        1.174      # tailcut1809
    ])
    par1_vs_cleaning = np.array([
        -2.239,    # tailcut84
        -2.239,    # tailcut1005
        -2.235,    # tailcut1206
        -2.267,    # tailcut1407
        -2.330,    # tailcut1608
        -2.428     # tailcut1809
    ])
    list_pict_cleaning = np.array([8, 10, 12, 14, 16, 18])

    params_vs_cleaning = np.array([par0_vs_cleaning, par1_vs_cleaning]).T
    reference_params = params_vs_cleaning[0, :]

    unique_picture_threshold = np.sort(np.unique(all_picture_thresh))
    assert set(unique_picture_threshold).issubset(list_pict_cleaning)

    for pict_th in unique_picture_threshold:
        # No correction for tailcut 84
        if pict_th == 8:
            continue
        arg_i = np.flatnonzero(pict_th == list_pict_cleaning)[0]
        correction = reference_params - params_vs_cleaning[arg_i, :]
        mask = (all_picture_thresh == pict_th)
        all_corrected_fit_params[mask] += correction

    #
    # The dependece with cos zenith of the peak cosmics rate, and intensity at
    # 50% of peak rate are more complicated and we need splines. Also computed
    # from the v0.9 analysis of runs 10800-12000 (good quality data):
    #
    czd = np.array(
            [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32,
             0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56,
             0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8,
             0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.])

    peak_rate = np.array([12.99216054, 13.59193902, 14.46537829, 15.57193601,
                          16.87106985, 18.32223746, 19.88489651, 21.51850465,
                          23.18251955, 24.83639886, 26.43960025, 27.95158137,
                          29.33179989, 30.53971347, 31.53900361, 32.33011501,
                          32.93242147, 33.36545325, 33.64874059, 33.80165409,
                          33.84217475, 33.78756809, 33.6550937, 33.46201118,
                          33.22680403, 32.9786081, 32.7520441, 32.58177807,
                          32.50247603, 32.54527421, 32.71058649, 32.983008,
                          33.34700313, 33.78703628, 34.29134794, 34.88104458,
                          35.59415516, 36.4688485, 37.5432934, 38.849495,
                          40.36581152, 42.04297872, 43.83150408, 45.68189509,
                          47.54465923, 49.37030398])

    intensity50 = np.array([41.11826207, 41.15641234, 41.22466677, 41.31856549,
                            41.43364861, 41.56545627, 41.70952858, 41.86140568,
                            42.01662769, 42.17073473, 42.31926692, 42.45776439,
                            42.58176727, 42.68681567, 42.76914445, 42.83103512,
                            42.87788259, 42.9151075, 42.94813048, 42.98132214,
                            43.00991401, 43.02443194, 43.0153629, 42.97319385,
                            42.88890509, 42.75777064, 42.57727538, 42.34492242,
                            42.05821489, 41.71642663, 41.33424318, 40.93428549,
                            40.53924008, 40.17179347, 39.85245074, 39.58273022,
                            39.35437412, 39.15904383, 38.98840075, 38.83449203,
                            38.69272196, 38.56022348, 38.43414379, 38.31163009,
                            38.18982959, 38.06588949])

    all_zd_corrected_cosmics_peak_rate = (
                np.interp(1, czd, peak_rate) * all_cosmics_peak_rate /
                np.interp(all_coszd, czd, peak_rate))

    all_zd_corrected_intensity_50 = (
                np.interp(1, czd, intensity50) * all_intensity_50 /
                np.interp(all_coszd, czd, intensity50))

    all_rate_at_422 = all_fit_params[:, 0]
    all_zd_corrected_rate_at_422 = all_corrected_fit_params[:, 0]

    # Compute, by inverting the fits, the value of intensity which corresponds to
    # a certain rate, event/s/p.e
    # (after correction to ZD=0)
    target_rate = 1.74
    # aprox. mean rate at center of the fitting range for good-quality
    # data (e.g. 20221118 - 20230214)
    powerlaw_index = all_corrected_fit_params[:, 1]
    intensity_at_reference_rate = (intensity_mean *
                                   (target_rate /
                                    all_corrected_fit_params[:, 0]) **
                                   (1 / powerlaw_index))
    # The "light yield" as calculated below (relative to the intensity_mean,
    # i.e. 422 p.e.) is a factor that tells us how much shower light is
    # collected (i.e. is 1 for data which have a value that matches the mean
    # value of the good-quality period 20221118 - 20230214)
    # The value can be used to re-calibrate data, i.e. scaling all pixel
    # charges by 1/light_yield before the image cleaning and re-doing the
    # analysis. The correction with the power-law index is needed because of
    # how dR/dI transforms under a change in the amount of light.

    light_yield = ((intensity_at_reference_rate / intensity_mean) **
                   (powerlaw_index / (1 + powerlaw_index)))

    out_dict = {"yyyymmdd": all_yyyymmdd,
                "ra_tel": all_tel_ra,
                "dec_tel": all_tel_dec,
                "cos_zenith": all_coszd,
                "az_tel": all_az_tel,
                "runnumber": all_runs,
                "subrun": all_subruns,
                "time": all_dragon_time,
                "elapsed_time": all_elapsed_time,
                "corrected_elapsed_time": all_corrected_elapsed_time,
                "delta_t_exp_index": all_dt_exp_index,
                "picture_thresh": all_picture_thresh,
                "boundary_thresh": all_boundary_thresh,
                "cosmics_rate": all_cosmic_rates,
                "cosmics_cleaned_rate": all_cosmic_cleaned_rates,
                "intensity_at_half_peak_rate": all_intensity_50,
                "intensity_at_peak_rate": all_peak_intensity,
                "ZD_corrected_intensity_at_half_peak_rate": all_zd_corrected_intensity_50,
                "cosmics_peak_rate": all_cosmics_peak_rate,
                # diff cosmics rate (events/s/p.e.) at (genuine) rate peak
                "ZD_corrected_cosmics_peak_rate": all_zd_corrected_cosmics_peak_rate,
                "cosmics_rate_at_422_pe": all_rate_at_422,
                "delta_cosmics_rate_at_422_pe": all_fit_errors[:, 0],
                # ^^^ (events/s/p.e.), value and uncertainty from fit
                "ZD_corrected_cosmics_rate_at_422_pe": all_zd_corrected_rate_at_422,
                "ZD_corrected_delta_cosmics_rate_at_422_pe": all_corrected_fit_errors[:, 0],
                "cosmics_spectral_index": all_fit_params[:, 1],
                "delta_cosmics_spectral_index": all_fit_errors[:, 1],
                "ZD_corrected_cosmics_spectral_index": all_corrected_fit_params[:, 1],
                "intensity_spectrum_fit_p_value": all_p_value,
                "intensity_at_reference_rate": intensity_at_reference_rate,
                "light_yield": light_yield,
                "diffuse_nsb_std": all_diffuse_nsb_std,
                "num_star_affected_pixels": all_star_affected_pixels,
                "anomalous_low_intensity_peak": all_threshold_anomaly_detected,
                # T/F, small rate peak at very low intensity? (satellites/meteors?)
                }

    pd.DataFrame(out_dict).to_hdf(output_file, key='cosmics_intensity_spectrum',
                                  mode='a', format='table',
                                  data_columns=list(out_dict.keys()))


def find50(x, y, min_intensity=0, miny=0.05):
    """
    This calculates the value of intensity (x) for which the 50% of the peak
    differential intensity spectrum dR/dI (y) is reached. It is a proxy for the
    camera threshold, when the peak is the one due to cosmic rays (in some
    cases there are spurious peaks that make this calculation complicated)

    x: array of intensity values
    y: corresponding dR/dI values (events/s/p.e.)
    min_intensity: float - see below
    min_y: only peaks with at least this value of y will be considered

    return: x50, drdimax, x100

    x50: intensity for which 50% of the peak dR/dI value is reached
    drdimax: peak value of dR/dI
    x100: intensity at which the peak dR/dI is found

    All the values above refer to the dR/dI peak which is deemed to be the
    genuine cosmics peak... If the first calculated x50 is below min_intensity,
    then the value is recomputed using as reference the next-highest peak in
    the dR/dI spectrum, until the x50 value is above min_intensity.
    Values at too-low-intensity probably do not correspond to the true cosmics
    dR/dI peak, and often indicate some issue in the data (e.g. a satellite
    moving across the field of view). In some cases there is no intensity (
    lower than that at which the peak occurs) for which dR/dI is below the
    peak value. This happens for example when a fake peak blends with the
    genuine cosmics peak.

    """

    if y.sum() == 0:
        return -1, -1, -1  # No intensity spectrum

    indices_maxima = argrelextrema(y, np.greater)

    if len(y[indices_maxima]) == 0:
        return -2, -2, -2  # No maximum found

    peak_values = y[indices_maxima]
    peak_positions = x[indices_maxima]

    # Exclude too-low peaks (may be just fluctuations if stats is low, e.g. at
    # high intensities). If there are no peaks above miny, just return the
    # height and position of the maximum one:
    if peak_values.max() < miny:
        return -3, peak_values.max(), peak_positions[np.argmax(peak_values)]

    # Keep only those higher than miny:
    above_miny = peak_values >= miny
    peak_values = peak_values[above_miny]
    peak_positions = peak_positions[above_miny]

    # Peak values ordered from biggest to smallest:
    ordered_peaks = np.sort(peak_values)[::-1]
    ordered_peaks_pos = peak_positions[np.argsort(peak_values)][::-1]

    # We first assume that the highest peak is the genuine cosmics peak.
    # If x50 is too low to be a reasonable threshold for cosmics, perhaps
    # the highest peak is an artifact (sometimes we see spurious peaks not due
    # to showers, at low intensities - perhaps caused by meteors or satellites)
    # In that case we try the next peaks in order of height, until we get
    # one for which x50 is below min_intensity. Note: this does not work
    # against fake peaks at high intensities, that we sometimes have e.g. due
    # to the MAGIC LIDAR crossing the LST FOV. In those cases we may get a
    # completely wrong value for x50 (i.e. unrelated to the cosmics intensity
    # spectrum).

    ipeak = -1
    x50 = np.nan

    while not (x50 >= min_intensity):  # Note this is true for x50==np.nan
        ipeak += 1
        if ipeak >= len(peak_values):
            break

        peak_bin = np.where(x == ordered_peaks_pos[ipeak])[0][0]
        # We move from the peak towards left (lower intensity), until we find
        # the 50% of the peak rate
        x_left_side = x[:peak_bin]
        y_left_side = y[:peak_bin]
        # Get the last intensity before the peak for which the rate is 50% of
        # peak:
        ix50 = np.where(y_left_side < 0.5 * ordered_peaks[ipeak])[0]
        if len(ix50) > 0:
            x50 = x_left_side[ix50[-1]]
        else:
            x50 = np.nan

    if not (x50 >= min_intensity):  # = did not find an x50 in a reasonable
        # range
        return -4, ordered_peaks[0], ordered_peaks_pos[0]
        # in this case return dR/dI and intensity for the highest peak

    return x50, ordered_peaks[ipeak], ordered_peaks_pos[ipeak]


def pol1(x, a, b):
    return a + b * x


def pol2(x, a, b, c):
    return a + b * x + c * x * x


def expfunc(x, a, b):
    return a * np.exp(b * x)


def time_to_yyyymmdd(t):
    hour = np.array([datetime.fromtimestamp(x).hour for x in t])
    t2 = np.array([x if h > 12 else x - 43200 for x, h in zip(t, hour)])
    dt = [datetime.fromtimestamp(x) for x in t2]

    return np.array([(x.year * 10000 + x.month * 100 + x.day) for x in dt])


if __name__ == '__main__':
    main()
