import streamlit as st
import sys
import math
import numpy as np





def thermal_satisfaction(uses, temps):

    satisfaction = []

    for use, t in zip(uses, temps):
        t = float(t)
        if use.startswith('RESID') or use.startswith('SOCIAL'):
            y = -0.05468 * t**2 + 2.596 * t - 26.64
        elif use.startswith('COMMERC'):
            y = -0.1073 * t**2 + 4.943 * t - 52.89
        else:
            y = float('nan')  

        if y < 2.5:
            satisfaction.append(0)
        elif y > 3.5:
            satisfaction.append(2)
        else:
            satisfaction.append(1)

    return satisfaction


def thermal_sensation(uses, temps):

    sensation = []

    for use, t in zip(uses, temps):
        t = float(t)

        if use.startswith('RESID') or use.startswith('SOCIAL'):
            y = 0.2324 * t - 5.695
        elif use.startswith('COMMERC'):
            y = 0.3907 * t - 9.124
        else:
            y = float('nan')  

        if y > 1.5:
            sensation.append(2)
        elif y > 0.5:
            sensation.append(1)
        elif y > -0.5:
            sensation.append(0)
        elif y > -1.5:
            sensation.append(-1)
        else:
            sensation.append(-2)

    return sensation


def views_horiz_satisf(views_results):
    satisfaction_values = []
    for sublist in views_results:
        mean_val = np.mean(sublist)

        if mean_val < 3.5:
            sat_val = 0
        elif mean_val > 5.5:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values
        

def views_sky_satisf(views_results):
    satisfaction_values = []
    for sublist in views_results:
        mean_val = np.mean(sublist)

        if mean_val < 3.4:
            sat_val = 0
        elif mean_val > 5.4:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values

def views_green_satisf(views_results):
    satisfaction_values = []
    for sublist in views_results:
        mean_val = np.mean(sublist)

        if mean_val < 1.10:
            sat_val = 0
        elif mean_val > 3.76:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values


def balconies_satisf(_results):
    satisfaction_values = []
    for sublist in _results:
        mean_val = np.mean(sublist)

        if mean_val < 4:
            sat_val = 0
        elif mean_val > 11:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values



def size_satisf(occupancy_rates, room_ids):
    satisfaction_values = []

    for val, room_name in zip(occupancy_rates, room_ids):
        if room_name.startswith("COMMERC"):
            if val > 0.10:
                sat_val = 0
            elif val < 0.08:
                sat_val = 2
            else:
                sat_val = 1
        else:  
            if val == 0:
                sat_val = 1
            elif val > 0.026:
                sat_val = 0
            elif val < 0.021:
                sat_val = 2
            else:
                sat_val = 1

        satisfaction_values.append(sat_val)

    return satisfaction_values


def sgreen_satisf(_results):
    satisfaction_values = []
    for val in _results:

        if val == None:
            sat_val = 1

        elif val < 1.50:
            sat_val = 0
        elif val > 2.00:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append(sat_val)
    return satisfaction_values

def samount_satisf(_results):
    satisfaction_values = []
    for val in _results:

        if val == 0:
            sat_val = 1

        elif val < 5.00:
            sat_val = 0
        elif val > 6.00:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append(sat_val)
    return satisfaction_values


def sdist_satisf(_results):
    satisfaction_values = []
    for val in _results:

        if val == 0:
            sat_val = 1

        elif val < 0.40:
            sat_val = 0
        elif val > 0.50:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append(sat_val)
    return satisfaction_values


def dl_percep(_results):
    satisfaction_values = []
    for sublist in _results:
        mean_val = np.mean(sublist)

        if mean_val < 50:
            sat_val = 0
        elif mean_val >70:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values


def udi_satisf(_results):
    satisfaction_values = []
    for sublist in _results:
        if not sublist:
            satisfaction_values.append([])
            continue

        count_above_50 = sum(val > 50.0 for val in sublist)
        percentage_above_50 = (count_above_50 / len(sublist)) * 100

        sat_val = 1 if percentage_above_50 >= 60 else 0
        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values


def sound_percep(_results):
    satisfaction_values = []
    for sublist in _results:
        mean_val = np.mean(sublist)

        if mean_val >= 60:
            sat_val = 0
        elif mean_val <= 35:
            sat_val = 2
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values


def sound_satisf(_results):
    satisfaction_values = []
    for sublist in _results:
        mean_val = np.mean(sublist)

        if mean_val > 45:
            sat_val = 0
        else:
            sat_val = 1

        satisfaction_values.append([sat_val] * len(sublist))
    return satisfaction_values



def predict_air_quality(co2_lists, rh_lists):
    all_satisfaction = []

    for co2_row, rh_row in zip(co2_lists, rh_lists):
        row_satisfaction = []

        for co2, rh in zip(co2_row, rh_row):
            
            if co2 > 1000:
                co2_sat = 0
            elif co2 < 800:
                co2_sat = 2
            else:
                co2_sat = 1

           
            if rh > 70 or rh < 20:
                rh_sat = 0
            elif (20 <= rh < 30) or (60 < rh <= 70):
                rh_sat = 1
            else:
                rh_sat = 2

            min_sat = min(co2_sat, rh_sat)
            row_satisfaction.append(min_sat)

        all_satisfaction.append(row_satisfaction)

    return all_satisfaction

