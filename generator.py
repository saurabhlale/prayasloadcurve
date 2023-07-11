from help import populate_dictionary,is_in_range,is_within_on_intervals,get_temp_array,is_within_sleep_intervals,get_speed_power,get_irradiation_array,is_heater_on
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import json

def modelAC(T,t,cap,cond,Q,TA):
    dTdt = (cond *(TA - T) - Q)/cap
    return dTdt

def generate_ac_load_curve(inputs):
    
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
    
    D_cop = I_eer/3.41214
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
    
    
    return times,power,temperatures_room,temperature 
    
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
    
    
def generate_heater_load_curve():
    
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
    
    

    
def plot_AC_images(times,power,temperatures_room,temperature):
    
    hours = times/60
    plt.step(hours, power)
    plt.xlabel('Hours')
    plt.ylabel('Load')
    plt.title('Load curve 24 hours')
    #plt.show()
    plt.savefig('load_curve_full.png')
    plt.clf()
    
    plt.plot(hours, temperatures_room, label='room temperature')
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
    ax2.set_ylabel('Room Temperature', color='g')

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


filein = input("Enter file name: ")
inputs = populate_dictionary(filein)
if inputs["EQ"] == "AC":
    times,power,temperatures_room,temperature = generate_ac_load_curve(inputs)
    plot_AC_images(times,power,temperatures_room,temperature)
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
    times,power,temperatures_room,temperature = generate_heater_load_curve() 
    plot_Heater_images(times,power,temperatures_room,temperature)
    np.savetxt('power.csv', np.concatenate((times.reshape((-1, 1)), power.reshape((-1, 1))), axis=1), delimiter=',')
    
    
