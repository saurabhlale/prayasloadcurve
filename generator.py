from help import populate_dictionary,is_in_range,is_within_on_intervals,get_temp_array,is_within_sleep_intervals

from help import get_speed_power,get_irradiation_array,is_heater_on,get_scaled_load_curve
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import json
from scipy.fft import fft, rfft, rfftfreq,irfft
import joblib
import pickle
import sklearn
import os
import random
from datetime import datetime

class stathelper:
    app_counter={}
    app_power_counter={}
    app_list=['AC', 'Fan', 'Generic', 'EV', 'Fridge', 'Heater', 'Light']
    collect_data = False
    resolution = None
    
    def insert_load_curve(self,app,power,resolution):

        if ((app in self.app_list) and self.collect_data):
            kwh=np.sum(power*resolution)/60.
            self.resolution = resolution
            if (app in self.app_counter):
                self.app_counter[app] = self.app_counter[app] + kwh
                self.app_power_counter[app] = self.app_power_counter[app] + power
            else:
                self.app_counter[app] = kwh
                self.app_power_counter[app] = power

    def plot_energy_pi(self):
        

        _, _, autotexts = plt.pie(list(self.app_counter.values()), labels=list(self.app_counter.keys()), autopct='%1.1f%%', shadow=False, startangle=140)

        # Add size values as annotations
        for i, size in enumerate(list(self.app_counter.values())):
            text = f'{size:.2f} kWh \n{autotexts[i].get_text()}'
            autotexts[i].set_text(text)
            autotexts[i].set_fontsize(8)

        en = sum(self.app_counter.values())
        # Add a title
        text = f'Energy consumption by appliance {en:.2f} kWh'
        plt.title(text)

        # Show the pie chart
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # plt.show()
        plt.savefig('energy_pi_chart.png')
        plt.clf()

        timesreq = np.arange(0,24*60,self.resolution, dtype=np.double)
        hours = timesreq/60.
        for app in self.app_power_counter:
            plt.fill_between(hours, self.app_power_counter[app], alpha=0.5, label=app)

        plt.xlabel('Hours')
        plt.ylabel('Power(kW)')
        plt.title('Aggregated area chart')
        plt.legend()
        plt.savefig('power_aggregate_area_chart.png')
        plt.clf()
   
def modelAC(T,t,cap,cond,Q,TA):
    dTdt = (cond *(TA - T) - Q)/cap
    return dTdt

def modelACWall(Z,t,capR,capW,cond,Q,TA,H):
    TW = Z[0]
    TR = Z[1]
    #dTWdt = ((cond *(TA - Z[1])) - H*(Z[0] - Z[1]))/capW
    dTRdt = ((cond *(TA - Z[1])) + (H*(Z[0] - Z[1])) - Q)/capR
    dTWdt = (H*(Z[1] - Z[0]))/capW
    dZdt = [dTWdt,dTRdt]
    return dZdt
    
def generate_ac_load_curve(inputs):
    if (inputs['I_include_wall_storage']):
        return generate_ac_load_curve_with_wall(inputs)
    else:
        return generate_ac_load_curve_without_wall(inputs)
 
def generate_ac_load_curve_with_wall(inputs):
    
    I_eer = inputs['I_eer']
    I_capacity = inputs['I_capacity']
    I_is_temp_profile = inputs['I_is_temp_profile']
    T_ambient = inputs['T_ambient']
    I_ref_Temperature = inputs['I_ref_Temperature']
    I_room_vol = inputs['I_room_vol']
    I_exposed_area = inputs['I_exposed_area']
    I_resolution = inputs['I_resolution']
    I_temp_file_path = inputs['I_temp_file_path']
    
    A_thickness = inputs['A_thickness']
    A_thermal_conductivity = inputs['A_thermal_conductivity']
    A_On_offset = inputs['A_On_offset']
    A_Off_offset = inputs['A_Off_offset']
    A_Denisty = inputs['A_Denisty']
    D_Specific_heat = inputs['D_Specific_heat']
    A_Eff = inputs['A_Eff']
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']
    
    A_Denisty_Wall = inputs['A_Denisty_Wall']
    D_Specific_heat_Wall = inputs['D_Specific_heat_Wall']
    A_Convective_Coefficient = inputs['A_Convective_Coefficient']
    
    D_cop = I_eer
    D_heat_removed = I_capacity*A_Eff*3.5 #kW
    D_Conductance = ((A_thermal_conductivity*I_exposed_area)/(A_thickness*1000)) # (kA/L), kW
    D_Convectance = ((A_Convective_Coefficient*I_exposed_area)/1000.) # (hAL), kW/K
    D_Temperature_on = I_ref_Temperature + A_On_offset
    D_Temperature_off = I_ref_Temperature + A_Off_offset
    
    
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
    temperature =  get_temp_array(times)
    temperatures_room = temperature.copy()
    temperatures_wall = temperature.copy()
    power = np.full_like(times,0, dtype=np.double)
    is_AC_on = False
    is_Cooling_on = False
    T_Curr = temperature[0]
    T_CurrW = temperature[0]

    for i in range(len(times)-1):
        
        #if(i>3):
            #break
        
        #is_Cooling_onePrev = 
        
        is_AC_on = is_within_on_intervals(times[i])
        
        if is_AC_on & (is_Cooling_on == False) & (T_Curr >= D_Temperature_on):
            is_Cooling_on = True
        if (is_Cooling_on == True) & (T_Curr <= D_Temperature_off):
            is_Cooling_on = False


        Qin = D_Conductance*(temperature[i] - T_Curr)

        Qout = 0


        if is_Cooling_on & is_AC_on:
            Qout = D_heat_removed


        delta_t = times[i+1] - times[i]
        
        capR = I_room_vol*A_Denisty*D_Specific_heat
        capW = I_exposed_area*A_thickness*A_Denisty_Wall*D_Specific_heat_Wall
        cond = D_Conductance
        H = D_Convectance
        Q = Qout
        TR = [T_Curr]
        TW = [T_CurrW]
        TA = temperature[i]
        TZ = [T_CurrW,T_Curr]
        t = np.linspace(0,delta_t*60,300)
        result = odeint(modelACWall,TZ,t,args=(capR,capW,cond,Q,TA,H,))
        
        Tnew = result[-1,1]
        Tnew2 =  result[-1,0]
        H_to_be_removed_total = I_room_vol*A_Denisty*D_Specific_heat*(T_Curr - D_Temperature_off) +  (I_exposed_area*A_thickness*A_Denisty_Wall*D_Specific_heat_Wall*(T_CurrW - D_Temperature_off))

        T_Curr = temperatures_room[i+1] = Tnew
        T_CurrW = temperatures_wall[i+1] = Tnew2

        
        if is_AC_on:
            if is_Cooling_on & (Tnew > D_Temperature_off):
                power[i] = D_heat_removed/D_cop
                #temperatures_room[i+1] = Tnew
            elif is_Cooling_on & (H_to_be_removed_total > 0):
                fct = result[result > D_Temperature_off].shape[0]/result.shape[0]
                tmp = fct*D_heat_removed/D_cop
                if fct > 1.0:
                    tmp = np.min([D_heat_removed/D_cop, fct*D_heat_removed/D_cop])
                power[i] = np.max([0.2*D_heat_removed/D_cop,tmp])
                T_Curr = temperatures_room[i+1] = D_Temperature_off
                #temperatures_room[i+1] = D_Temperature_off
            else:
                power[i] = 0.2*D_heat_removed/D_cop
        
        
            
        #power[i+1] = power[i]
    
    
    return times,power,temperatures_room,temperature,temperatures_wall 
    
    
 
def generate_ac_load_curve_without_wall(inputs):
    
    I_eer = inputs['I_eer']
    I_capacity = inputs['I_capacity']
    I_is_temp_profile = inputs['I_is_temp_profile']
    T_ambient = inputs['T_ambient']
    I_ref_Temperature = inputs['I_ref_Temperature']
    I_room_vol = inputs['I_room_vol']
    I_exposed_area = inputs['I_exposed_area']
    I_resolution = inputs['I_resolution']
    I_temp_file_path = inputs['I_temp_file_path']
    
    A_thickness = inputs['A_thickness']
    A_thermal_conductivity = inputs['A_thermal_conductivity']
    A_On_offset = inputs['A_On_offset']
    A_Off_offset = inputs['A_Off_offset']
    A_Denisty = inputs['A_Denisty']
    D_Specific_heat = inputs['D_Specific_heat']
    A_Eff = inputs['A_Eff']
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']
    
    D_cop = I_eer
    D_heat_removed = I_capacity*A_Eff*3.5 #kW
    D_Conductance = ((A_thermal_conductivity*I_exposed_area)/(A_thickness*1000)) # (kA/L), kW
    D_Temperature_on = I_ref_Temperature + A_On_offset
    D_Temperature_off = I_ref_Temperature + A_Off_offset

    times = np.arange(0,24*60,inputs['I_resolution'], dtype=np.double)
    #print(times)
    temperature =  get_temp_array(times)
    temperatures_room = temperature.copy()
    power = np.full_like(times,0, dtype=np.double)
    is_AC_on = False
    is_Cooling_on = False
    T_Curr = temperature[0]

    for i in range(len(times)-1):
        
        #if(i>3):
            #break
        
        #is_Cooling_onePrev = 
        
        is_AC_on = is_within_on_intervals(times[i])
        
        if is_AC_on & (is_Cooling_on == False) & (T_Curr >= D_Temperature_on):
            is_Cooling_on = True
        if (is_Cooling_on == True) & (T_Curr <= D_Temperature_off):
            is_Cooling_on = False


        Qin = D_Conductance*(temperature[i] - T_Curr)

        Qout = 0


        if is_Cooling_on & is_AC_on:
            Qout = D_heat_removed


        delta_t = times[i+1] - times[i]
        
        cap = I_room_vol*A_Denisty*D_Specific_heat
        cond = D_Conductance
        Q = Qout
        T0 = [T_Curr]
        TA = temperature[i]
        t = np.linspace(0,delta_t*60,300)
        result = odeint(modelAC,T0,t,args=(cap,cond,Q,TA,))
        
        Tnew = result[-1,0]
        H_to_be_removed_total = I_room_vol*A_Denisty*D_Specific_heat*(T_Curr - D_Temperature_off)

        T_Curr = temperatures_room[i+1] = Tnew

        
        if is_AC_on:
            if is_Cooling_on & (Tnew > D_Temperature_off):
                power[i] = D_heat_removed/D_cop
                #temperatures_room[i+1] = Tnew
            elif is_Cooling_on & (H_to_be_removed_total > 0):
                fct = result[result > D_Temperature_off].shape[0]/result.shape[0]
                power[i] = np.max([0.2*D_heat_removed/D_cop,fct*D_heat_removed/D_cop])
                T_Curr = temperatures_room[i+1] = D_Temperature_off
                #temperatures_room[i+1] = D_Temperature_off
            else:
                power[i] = 0.2*D_heat_removed/D_cop
        
        
            
        #power[i+1] = power[i]
    
    
    return times,power,temperatures_room,temperature,np.empty([0, 0])
    
def generate_fan_load_curve(inputs):
    
    I_watt = inputs['I_watt']
    I_is_temp_profile = inputs['I_is_temp_profile']
    T_ambient = inputs['T_ambient']
    I_resolution = inputs['I_resolution']
    I_default_start_speed = inputs['I_default_start_speed']
    A_speeds = inputs['A_speeds']
    A_speed_temps = inputs['A_speed_temps']
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']
    A_speed_prob = inputs['A_speed_prob']
    A_interval_of_no_change = inputs['A_interval_of_no_change']

    is_fan_on = False
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
    temperature =  get_temp_array(times)
    temperature = temperature - 3 #assume room temperature is less than ambient
    power = np.full_like(times,0, dtype=np.double)
    
    speed = 0
    pwr = 0
    
    last_change = 0.
    if I_default_start_speed != 0:
        pwr = A_speeds[I_default_start_speed-1]*I_watt
    
    for i in range(len(times)-1):
        is_On_interval = is_within_on_intervals(times[i])
        is_sleep_interval = is_within_sleep_intervals(times[i])
        
        
        
        if ((is_On_interval == True) & (is_sleep_interval == False)):
            if last_change > A_interval_of_no_change:
                pwr = get_speed_power(temperature[i])
                last_change = 0
            else:
                last_change = last_change + np.abs(times[i+1] - times[i])
        elif ((is_On_interval == False) & (is_sleep_interval == False)):
            pwr =0.
            last_change = 0
        
        power[i] = power[i+1] = pwr
        
    
    return times,power,temperature
    
def generate_light_load_curve(inputs):
    
    I_watt = inputs['I_watt']
    I_radiation_profile = inputs['I_radiation_profile']
    I_light_threshold = inputs['I_light_threshold']
    I_resolution = inputs['I_resolution']
    I_preffered_time = inputs['I_preffered_time']
    
    A_Room_Radiaction_Efficiency = inputs['A_Room_Radiaction_Efficiency']
    A_flux_tol_lux = inputs['A_flux_tol_lux']
    A_SD_DEV_TIME_ON = inputs['A_SD_DEV_TIME_ON']
    A_SD_DEV_TIME_OFF = inputs['A_SD_DEV_TIME_OFF']

    is_light_on = False
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
    irradiation =  get_irradiation_array(times)
    luxsolarroom =  irradiation*A_flux_tol_lux*A_Room_Radiaction_Efficiency
    power = np.full_like(times,0, dtype=np.double)
    
    on_time = np.random.normal(I_preffered_time[0]*60.0,A_SD_DEV_TIME_ON,size=1)[0]
    off_time = np.random.normal(I_preffered_time[1]*60.0,A_SD_DEV_TIME_OFF,size=1)[0]
    
    #on_time = I_preffered_time[0]*60.0 -90
    #off_time = I_preffered_time[1]*60.0
    
    for i in range(len(times)-1):
        
        inrange = is_in_range(on_time,off_time, times[i])
        
        is_light_on_prev = is_light_on
        
        if ((luxsolarroom[i] <= I_light_threshold) & inrange) & (is_light_on_prev == False):
            is_light_on = True
        
        if (inrange == False) & (is_light_on_prev == True):
            is_light_on = False
            
        is_light_on_prev = is_light_on
        
        if is_light_on:
            p = I_watt
        else:
            p = 0
            
        
        power[i] = power[i+1] = p
            
    
    return times,power,luxsolarroom
    
def modelHT(T,t,cap,cond,Q,TA):
    dTdt = (cond *(TA - T) + Q)/cap
    return dTdt
    
    
def generate_heater_load_curve(inputs):
    
    I_capacity = inputs['I_capacity']
    I_is_temp_profile = inputs['I_is_temp_profile']
    T_ambient = inputs['T_ambient']
    I_ref_Temperature = inputs['I_ref_Temperature']
    I_on_intervals = inputs['I_on_intervals']
    I_resolution = inputs['I_resolution']
    I_Diameter = inputs['I_Diameter']
    I_Length = inputs['I_Length']
    I_Family_size = inputs['I_Family_size']

    A_thickness = inputs['A_thickness']
    A_thermal_conductivity = inputs['A_thermal_conductivity']
    A_On_offset = inputs['A_On_offset']
    A_Off_offset = inputs['A_Off_offset']
    A_Denisty = inputs['A_Denisty']
    D_Specific_heat = inputs['D_Specific_heat']
    A_Eff = inputs['A_Eff']
    A_Consumption_per_person = inputs['A_Consumption_per_person']
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']
    
    D_exposed_area = np.pi*I_Diameter*I_Length
    D_Volume = (np.pi*(I_Diameter*I_Diameter/4)*I_Length)/2.
    D_heat_added = I_capacity*A_Eff #kW
    D_Conductance = ((A_thermal_conductivity*D_exposed_area)/(A_thickness*1000)) # (kA/L), kW/K
    D_Temperature_on = I_ref_Temperature + A_On_offset
    D_Temperature_off = I_ref_Temperature + A_Off_offset
    D_Mixing_Events = np.sort(np.random.randint(low=I_on_intervals[0]*60.0, high=I_on_intervals[1]*60.0, size=I_Family_size).astype(np.double))
    
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
    temperature =  get_temp_array(times)
    temperatures_room = temperature.copy()
    power = np.full_like(times,0, dtype=np.double)
    is_Heater_on = False
    is_Heating_on = False
    T_Curr = temperature[0]

    is_mixing = False
    mixing_index = 0
    
    for i in range(len(times)-1):
        
        #if(i>3):
            #break
        is_Heater_on = is_heater_on(times[i])
        #is_Cooling_onePrev = 
        if is_Heater_on & (is_Heating_on == False) & (T_Curr <= D_Temperature_on):
            is_Heating_on = True
        if (is_Heating_on == True) & (T_Curr >= D_Temperature_off):
            is_Heating_on = False


        

        Qout = 0
        #is_Heating_on = False
        #is_Heater_on = False

        if is_Heating_on & is_Heater_on:
            Qout = D_heat_added
        #Qout = 0

        delta_t = times[i+1] - times[i]
        
        cap = D_Volume*A_Denisty*D_Specific_heat
        cond = D_Conductance
        Q = Qout
        T0 = [T_Curr]
        TA = temperature[i]
        t = np.linspace(0,delta_t*60,300)
        result = odeint(modelHT,T0,t,args=(cap,cond,Q,TA,))
        
        Tnew = result[-1,0]
        
        
        #print('-----------')
        H_to_be_added_total = D_Volume*A_Denisty*D_Specific_heat*(D_Temperature_off - T_Curr)

        T_Curr = temperatures_room[i+1] = Tnew
        
        #print(is_Heating_on)
        #print(Qout)
        #print (T_Curr)
        #print (Tnew)

        
        if is_Heater_on:
            if is_Heating_on & (Tnew < D_Temperature_off):
                power[i] = D_heat_added
                #temperatures_room[i+1] = Tnew
            elif is_Heating_on & (H_to_be_added_total > 0):
                fct = result[result < D_Temperature_off].shape[0]/result.shape[0]
                power[i] = fct*D_heat_added
                T_Curr = temperatures_room[i+1] = D_Temperature_off
                #temperatures_room[i+1] = D_Temperature_off
            else:
                power[i] = 0.
        
        if (mixing_index < D_Mixing_Events.shape[0]) and (times[i] >= D_Mixing_Events[mixing_index]):
            T_Curr = ((T_Curr*(D_Volume - A_Consumption_per_person)) + (temperature[i]*A_Consumption_per_person))/D_Volume
            mixing_index = mixing_index + 1
            
        #print (T_Curr)
        #print (Tnew)
        #print('-----------')
        
        #power[i+1] = power[i]
    
    
    return times,power,temperatures_room,temperature 
    
def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def get_load_curv_knn_ref(inputs):
   
    size = inputs['I_Size']
    typei = inputs['I_type']
    rating = inputs['I_rating']
    month = inputs['I_month']
    K = inputs['A_Knn_N']
    times = np.arange(0,24*60,15., dtype=np.double)
    
    map_fet = {
    'Direct Cool':1,
    'Frost free':2,
    '1 star':3,
    '2 star':4,
    '3 star':5,
    '4 star':6,
    '5 star':7,
    'Don’t know':8,
    'Not rated':9,
    
    }
    
    data_file_path = os.path.join('data', 'ref_size_norm.pkl')
    
    sz = load_object(data_file_path)
    
    data_file_path = os.path.join('data', 'nn_model_refrigerator.joblib')
    
    nbrs = joblib.load(data_file_path)
    # Load DataFrame from compressed CSV file
    
    data_file_path = os.path.join('data', 'features.csv.gz')
    features = pd.read_csv(data_file_path, compression='gzip')
    
    data_file_path = os.path.join('data', 'dfload1.csv.gz')
    dfload1 = pd.read_csv(data_file_path, compression='gzip')

    search_array = np.full((1, features.shape[1]-1),0.)
    search_array[0,0] = (size - sz[0])/ (sz[1] - sz[0])
    search_array[0,map_fet[typei]] = 1.
    search_array[0,map_fet[rating]] = 1.
    #print(features.columns)
    
    searchpd = pd.DataFrame(data=search_array, columns=features.columns[1:])
    
    distances, indices = nbrs.kneighbors(searchpd,K, return_distance=True)
    
    #print(distances)
    #print(indices)
    
    if distances[0,0] == 0.:
        devc = features.iloc[indices[0,0],0]
        temp = dfload1[(dfload1['dpid'] == devc) & (dfload1['month']==month)].sort_values(by=['date', 'block']).reset_index()
        start = np.random.randint(low=0, high=temp.shape[0] - 100, size=1)[0]
        return times,temp['load'].to_numpy()[start:start+ 96]
    else:
        wts =  1./distances.flatten()
        indices = indices.flatten()
        
        fftsum = np.empty((0,0))
        for (ind,wt) in zip(indices,wts):
            devc = features.iloc[ind,0]
            temp = dfload1[(dfload1['dpid'] == devc) & (dfload1['month']==month)].sort_values(by=['date', 'block']).reset_index()
            start = np.random.randint(low=0, high=temp.shape[0] - 100, size=1)[0]
            ss = temp['load'].to_numpy()[start:start+ 96]
            
            yf = rfft(ss)
            
            if(fftsum.shape[0] == 0):
                fftsum = yf*wt
            else:
                fftsum = fftsum + (yf*wt)
        
        #print(wts)
        #print(np.sum(wts))
        yffin = fftsum/np.sum(wts)
        return times,irfft(yffin)
                
def generate_ev_load_curve(inputs):
    
    cv_start_time = 0
    
    def get_charging_current(soc,time):
        if I_charging_method == 1:
            return D_Charging_current_CC
        else:
            if (soc < A_CC_charging_threshold):
                return D_Charging_current_CC
            else:
                Ttemp = abs(time - cv_start_time)
                #print(Ttemp)
                Cval = np.exp(Ttemp/time_constant) 

                Cval =D_Charging_current_CC/Cval
                
                return Cval
            
    I_battery_energy_capacity = inputs['I_battery_energy_capacity']
    I_charging_method = inputs['I_charging_method']
    I_charging_prob = inputs['I_charging_prob']
    I_start_interval = inputs['I_start_interval']
    I_resolution = inputs['I_resolution']
    I_charger_rating = inputs['I_charger_rating']
    
    A_battery_voltage = inputs['A_battery_voltage']
    A_start_soc = inputs['A_start_soc']
    A_charging_voltage = inputs['A_charging_voltage']
    A_CC_charging_threshold = inputs['A_CC_charging_threshold']
    
    D_Charging_current_CC = (I_charger_rating*1000)/A_charging_voltage
    D_charge_capacity = ((1000*I_battery_energy_capacity)/A_battery_voltage)
    D_charge_current_CC = D_charge_capacity*D_Charging_current_CC
    D_Charging_start_time = np.random.randint(low=I_start_interval[0]*60, high=I_start_interval[1]*60, size=1).astype(np.double)[0]


    is_charger_on = random.random() <= I_charging_prob
    is_charging_on = False
    is_charging_complete = False
    is_cc_threshold = False
    
    
    soc = A_start_soc
    
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
    power = np.full_like(times,0, dtype=np.double)
    socs = np.full_like(times,A_start_soc, dtype=np.double)
    time_constant = 1;
    
    for i in range(len(times)-1):
        
        if ((times[i] >= D_Charging_start_time) & is_charger_on & (is_charging_complete == False)):
            is_charging_on = True
            
        if((soc >= 1.) & (is_charging_complete == False)):
            is_charging_on = False
            is_charging_complete = True
            #print ((D_Charging_start_time - times[i])/60.)
            
        if((soc >= A_CC_charging_threshold) & (is_cc_threshold == False)):
            cv_start_time = times[i]
            is_cc_threshold = True
            time_constant = abs(cv_start_time - D_Charging_start_time)/2
            #print(time_constant)
            
        pwr = 0.
        if is_charging_on:
            charging_curr = get_charging_current(soc,times[i])
            
            soc = soc + ((charging_curr*I_resolution/60.)/D_charge_capacity)
            pwr = charging_curr*A_charging_voltage
            #current[i] = charging_curr
        
        if is_charger_on:
            socs[i] = soc
        
        power[i] = pwr
        
    
    return times,power,socs
                         
            
def generate_generic_load_curve(inputs):
    
    I_watt = inputs['I_watt']
    I_method = inputs['I_method']
    I_on_intervals = inputs['I_on_intervals']
    I_start_interval = inputs['I_start_interval']
    I_resolution = inputs['I_resolution']
    I_Duration = inputs['I_Duration']
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']

    is_device_on = False
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    #print(times)
   
    power = np.full_like(times,0, dtype=np.double)
    
    D_Charging_start_time = np.random.randint(low=I_start_interval[0]*60, high=I_start_interval[1]*60, size=1).astype(np.double)[0]
    duration_st = np.random.normal(I_Duration,A_SD_DEV_TIME,size=1)[0]
    off_time =  D_Charging_start_time + duration_st
    
    #print(D_Charging_start_time)
    #print(off_time)

    pwr = 0
    
    
    for i in range(len(times)-1):
        
        if(I_method == 1):
            is_device_on = is_within_on_intervals(times[i])
        else:
            
            on_time = D_Charging_start_time
            
            is_device_on = is_in_range(D_Charging_start_time,off_time, times[i])
        
        if (is_device_on):
            pwr = I_watt
        else:
            pwr =0.
        
        power[i] =  pwr
        
    #print('in generic method')
    #print(times.shape)
    #print(power.shape)
    return times,power
    
def get_irradiation_array_solar_array(timearray,inputs):

    A_Solar_Constant = inputs['A_Solar_Constant']
    A_Air_transmissibility = inputs['A_Air_transmissibility']
    I_Lattitute = inputs['I_Lattitute']
    I_Tilt_Angle = inputs['I_Tilt_Angle']
    I_Day = inputs['I_Day']
    I_radiation_profile = inputs['I_radiation_profile']
    I_radiation_file_path = inputs['I_radiation_file_path']
    
    #timearray = times
    if I_radiation_profile:
        T_tempdata = pd.read_csv(I_radiation_file_path)
        tempf = pd.DataFrame(timearray, columns=['Minutes'])
        tempf['Hour'] = np.floor_divide(timearray,60,dtype=np.double)
        tempf = tempf.join(T_tempdata[['Hour', 'GHI']],on='Hour', how='left', lsuffix='_caller', rsuffix='_other')
        return tempf['GHI'].to_numpy()
    else:
        tempf = pd.DataFrame(timearray, columns=['Minutes'])
        tempf['Hour'] = timearray/60

        date_object = datetime.strptime(I_Day, '%Y-%m-%d')
        Nday = date_object.timetuple().tm_yday

        Delta = 23.45*np.sin(np.radians((360*(284 + Nday))/365.))

        Effective_Lat = I_Lattitute - I_Tilt_Angle

        #H_angle = (15 * (tempf['Hour'].to_numpy() - 12)) + I_Longitude

        H_angle = (15 * (tempf['Hour'].to_numpy() - 12))

        #H_angle = (15 * (12 - tempf['Hour'].to_numpy()))

        #[sinδ sin(θ −α)+cosδ cos(θ −α)cosφ]dφ .

        temp = (np.sin(np.radians(Delta))*np.sin(np.radians(Effective_Lat))) + (np.cos(np.radians(Delta))*np.cos(np.radians(Effective_Lat))*np.cos(np.radians(H_angle)))


        eps = 1 + (0.033*np.cos((2*np.pi*Nday)/365))
        IH = (12/np.pi)*A_Solar_Constant*eps*A_Air_transmissibility*temp

        IH = A_Solar_Constant*A_Air_transmissibility*temp

        mask = IH < 0.

        IH[mask] = 0
        return IH
    
def generate_solar_panel_load_curve(inputs):
    
    I_resolution = inputs['I_resolution']
    I_rated_power = inputs['I_rated_power']
    A_Startup_insolation = inputs['A_Startup_insolation']
    I_Area = inputs['I_Area']
    I_Eff = inputs['I_Eff']
    I_No_panel = inputs['I_No_panel']
    
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    irradiation =  get_irradiation_array_solar_array(times, inputs)
    power = np.full_like(times,0, dtype=np.double)
    #on_time = I_preffered_time[0] -90
    #off_time = I_preffered_time[1]
    U_limit = I_rated_power
    #print(U_limit)
    #print(A_Startup_insolation)
    for i in range(len(times)-1):
        
        pwr =0.
        if (irradiation[i] > A_Startup_insolation):
            #print('here')
            ptemp = (irradiation[i] - A_Startup_insolation)*I_Area*I_Eff
            pwr = np.min([ptemp,U_limit])*I_No_panel
        
        
        power[i] = pwr
        
    kwh=np.sum(power*I_resolution)/(1000*60)

    
    print(f"Energy produced by solar cell: {kwh:.3f} kWh")
    return times,power,irradiation
    
def plot_AC_images(times,power,temperatures_room,temperature,temperatures_wall):
    
    hours = times/60
    plt.step(hours, power)
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve 24 hours')
    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()
    
    if (temperatures_wall.shape[0] == 0):
        plt.plot(hours, temperatures_room, label='room temperature')
        plt.plot(hours, temperature, label='ambient temperature')
        plt.legend()
        plt.xlabel('Hours')
        plt.ylabel('Temperature')
        plt.title('Temperatures')
        #plt.show()
        plt.savefig('temperatures.png')
        plt.clf()
    else:
        plt.plot(hours, temperatures_room, label='room temperature')
        plt.plot(hours, temperatures_wall, label='wall temperature')
        plt.plot(hours, temperature, label='ambient temperature')
        plt.legend()
        plt.xlabel('Hours')
        plt.ylabel('Temperature')
        plt.title('Temperatures')
        #plt.show()
        plt.savefig('temperatures.png')
        plt.clf()

    plt.step(hours, power)
    plt.xlim([6, 12])
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve partial')
    #plt.show()
    plt.savefig('load_curve_partial.png')
    plt.clf()



def plot_Fan_images(times,power,temperature):

    hours = times/60
    fig, ax1 = plt.subplots()
    plt.title("Fan load curve")
    ax2 = ax1.twinx()
    ax1.plot(hours, power,'-g')
    ax2.plot(hours, temperature,'-b')

    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Power', color='g')
    ax2.set_ylabel('Room Temperature', color='b')

    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()
    

    
    
def plot_Light_images(times,power,temperature):

    hours = times/60
    fig, ax1 = plt.subplots()
    plt.title("Light load curve")
    ax2 = ax1.twinx()
    ax1.plot(hours, power,'-g')
    ax2.plot(hours, luxsolarroom,'-b')

    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Power', color='g')
    ax2.set_ylabel('Room Irridation', color='b')

    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()
    
def plot_Heater_images(times,power,temperatures_room,temperature):
    
    hours = times/60
    plt.step(hours, power)
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve 24 hours')
    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()
    
    plt.plot(hours, temperatures_room, label='water temperature')
    plt.plot(hours, temperature, label='ambient temperature')
    plt.legend()
    plt.xlabel('Hours')
    plt.ylabel('Temperature')
    plt.title('Temperatures')
    #plt.show()
    plt.savefig('temperatures.png')
    plt.clf()

    plt.step(hours, power)
    plt.xlim([6, 12])
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve partial')
    #plt.show()
    plt.savefig('load_curve_partial.png')
    plt.clf()

def plot_Fridge_images(times,power):

    hours = times/60
    plt.step(hours, power)
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve 24 hours')
    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()


def plot_EV_images(times,power,socs):
    hours = times/60
    fig, ax1 = plt.subplots()
    plt.title("Load curve")
    ax2 = ax1.twinx()
    ax1.plot(hours, power,'-g')
    ax2.plot(hours, socs*100.,'-b')

    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Power', color='g')
    ax2.set_ylabel('Charge %', color='b')

    plt.savefig('load_curve_full.png')
    plt.clf()
    
def plot_Solar_Panel_images(times,power,irr):

    hours = times/60
    fig, ax1 = plt.subplots()
    plt.title("Light load curve")
    ax2 = ax1.twinx()
    ax1.plot(hours, power,'-g')
    ax2.plot(hours, irr,'-b')

    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Power', color='g')
    ax2.set_ylabel('Solar Irridation', color='b')
    plt.savefig('load_curve_full.png')
    plt.clf()
    
def load_curve_from_file(filein,resolution):
    
    #print(filein)
    inputs2 = populate_dictionary(filein)
    
    timesreq = np.arange(0,24*60,resolution, dtype=np.double)
    
    
    times = None
    power = None
    
    #print('New app')
    #print(inputs2["EQ"])
    
    if inputs2["EQ"] == "AC":
        times,power,temperatures_room,temperature,temperatures_wall = generate_ac_load_curve(inputs2)
    elif inputs2["EQ"] == "Fan":
        times,power,temperature = generate_fan_load_curve(inputs2)
        power = power/1000.
    elif inputs2["EQ"] == "Light":
        times,power,luxsolarroom = generate_light_load_curve(inputs2)
        power = power/1000.
    elif inputs2["EQ"] == "Heater":
        times,power,temperatures_room,temperature = generate_heater_load_curve(inputs2) 
    elif inputs2["EQ"] == "Fridge":
        times,power = get_load_curv_knn_ref(inputs2)
    elif inputs2["EQ"] == "EV":
        times,power,socs = generate_ev_load_curve(inputs2)
        power = power/1000.
    elif inputs2["EQ"] == "Generic":
        #print('In generic')
        times,power = generate_generic_load_curve(inputs2)
        power = power/1000.
    elif inputs2["EQ"] == "Agg":
        times,power = agg_household(inputs2,'appliances') 


    
    #print(times.shape)
    #print(power.shape)
    powereq = get_scaled_load_curve(times,power,timesreq)
    
    app_energy_counter.insert_load_curve(inputs2["EQ"],powereq,resolution)

    return powereq
        
    
    
def agg_household(inputs, component, silent=True):
    appliances = inputs[component]
    I_resolution = inputs['I_Resolution']
    
    times = np.arange(0,24*60,I_resolution, dtype=np.double)
    poweragg = np.full_like(times,0, dtype=np.double)

    app_energy_counter.collect_data = True

    for app in appliances:
        #print(app)
        for cnt in range(app[1]):
            powertemp = load_curve_from_file(app[0],I_resolution)
            poweragg = poweragg + powertemp
            
    kwh=np.sum(poweragg*I_resolution)/60

    app_energy_counter.plot_energy_pi()

    if(silent == False):
        print(f"Energy Consumption : {kwh:.3f} kWh")

    return times, poweragg

def plot_Agg_images(times,power):

    hours = times/60
    plt.step(hours, power)
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve 24 hours')
    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()



filein = input("Enter file name: ")
inputs = populate_dictionary(filein)

app_energy_counter = stathelper()

if inputs["EQ"] == "AC":
    times,power,temperatures_room,temperature,temperatures_wall = generate_ac_load_curve(inputs)
    plot_AC_images(times,power,temperatures_room,temperature,temperatures_wall)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Fan":
    times,power,temperature = generate_fan_load_curve(inputs)
    plot_Fan_images(times,power,temperature)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Light":
    times,power,luxsolarroom = generate_light_load_curve(inputs) 
    plot_Light_images(times,power,luxsolarroom)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Heater":
    times,power,temperatures_room,temperature = generate_heater_load_curve(inputs) 
    plot_Heater_images(times,power,temperatures_room,temperature)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Fridge":
    times,power = get_load_curv_knn_ref(inputs) 
    plot_Fridge_images(times,power)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Generic":
    times,power = generate_generic_load_curve(inputs) 
    plot_Fridge_images(times,power)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "EV":
    times,power,socs = generate_ev_load_curve(inputs)
    plot_EV_images(times,power,socs)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Agg":
    times,power = agg_household(inputs, 'appliances', False) 
    plot_Fridge_images(times,power)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "Settlement":
    times,power = agg_household(inputs,'houses', False) 
    plot_Fridge_images(times,power)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
elif inputs["EQ"] == "SolarPanel":
    times,power,irr = generate_solar_panel_load_curve(inputs) 
    plot_Solar_Panel_images(times,power,irr)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
    
    
