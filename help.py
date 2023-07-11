import pandas as pd
from scipy.integrate import odeint
import numpy as np
import json

inputs = {}

def populate_dictionary(path):
    global inputs
    with open(path, 'r') as file:
        obj = json.load(file)
        inputs = obj
    return obj



def is_in_range(start,end,val):
    
    if (start <= end):
        return ((val>start) & (val <= end))
    else:
        return ((val>start) or (val <= end))
    
def is_within_on_intervals(time):
    is_within_interval = False
    global inputs
    for intv in inputs['I_on_intervals']:
        on_time = np.random.normal(intv[0]*60.0,inputs['A_SD_DEV_TIME'],size=1)[0]
        off_time = np.random.normal(intv[1]*60.0,inputs['A_SD_DEV_TIME'],size=1)[0]
        
        inrange = is_in_range(on_time,off_time, time)
        
        if inrange:
            is_within_interval = True
            break
        
    return is_within_interval
    
def is_within_sleep_intervals(time):
    global inputs
    
    A_SD_DEV_TIME = inputs['A_SD_DEV_TIME']
    I_sleep_time = inputs['I_sleep_time']
    
    on_time = np.random.normal(I_sleep_time[0]*60.0,A_SD_DEV_TIME,size=1)[0]
    off_time = np.random.normal(I_sleep_time[1]*60.0,A_SD_DEV_TIME,size=1)[0]
    
    return is_in_range(on_time,off_time, time)
    
    
def get_speed_power(temp):
    speed =0
    power = 0
    global inputs
    A_speed_temps = inputs['A_speed_temps']
    A_speed_prob = inputs['A_speed_prob']
    A_speeds = inputs['A_speeds']
    I_watt = inputs['I_watt']
    
    if (is_in_range(A_speed_temps[0],A_speed_temps[1],temp)):
        speed = 1
    elif (is_in_range(A_speed_temps[1],A_speed_temps[2],temp)):
        speed = 2
    elif (is_in_range(A_speed_temps[2],A_speed_temps[3],temp)):
        speed = 3
    elif (is_in_range(A_speed_temps[3],A_speed_temps[4],temp)):
        speed = 4
    elif temp > A_speed_temps[4]:
        speed = 5
        
    if(speed != 0):
        prb = np.full(5,.00,dtype=np.double)
        prb[speed-1] = 1.
        #print(prb)
        power_factor_index = np.random.choice(np.arange(0, 5), p=A_speed_prob[speed-1])
        power = A_speeds[power_factor_index]*I_watt
    
    return power
    
def get_temp_array(timearray):
    global inputs
    if inputs['I_is_temp_profile']:
        #hours = np.floor_divide(times,60,dtype=np.double)
        T_tempdata = pd.read_csv(inputs['I_temp_file_path'])
        tempf = pd.DataFrame(timearray, columns=['Minutes'])
        tempf['Hour'] = np.floor_divide(timearray,60,dtype=np.double)
        tempf = tempf.join(T_tempdata[['Hour', 'Temperature']],on='Hour', how='left', lsuffix='_caller', rsuffix='_other')
        return tempf['Temperature'].to_numpy()
    else:
        return np.full_like(timearray,inputs['T_ambient'],dtype=np.double)
        
def get_irradiation_array(timearray):
    if inputs['I_radiation_profile']:
        #hours = np.floor_divide(times,60,dtype=np.double)
        T_tempdata = pd.read_csv(inputs['I_radiation_file_path'])
        tempf = pd.DataFrame(timearray, columns=['Minutes'])
        tempf['Hour'] = np.floor_divide(timearray,60,dtype=np.double)
        tempf = tempf.join(T_tempdata[['Hour', 'DHI']],on='Hour', how='left', lsuffix='_caller', rsuffix='_other')
        return tempf['DHI'].to_numpy()
        
def is_heater_on(time):
    on_time = inputs['I_on_intervals'][0]*60.0
    off_time = inputs['I_on_intervals'][1]*60.0
    
    return is_in_range(on_time,off_time, time)