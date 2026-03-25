# 5 orbits in each shell
from sgp4.api import Satrec, WGS72
from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84
from datetime import datetime
import datetime
from datetime import timedelta
import pandas as pd
import copy
import skyfield.api as sf
import numpy as np
from progress.bar import Bar
import math

def simulateConstellation(satellites, groundstation, minimumElevation, startTime, stopTime, ts = None, safetyMargin = 0):
    """Simulate visibility from <groundstation> for satellites in constellation <satellites> at a minimum elevation angle <minimumElevation> from <startTime> to <stopTime>

Optional arguments:
    ts : skyfield timescale
    safetyMargin: simulate <safetyMargin> days before and after time span. This is necessary because otherwise there might be set/rise missing from dataframe

Output:
    pandas.DataFrame, indexed with satellite names. Contains the columns:
        Rise: time the satellite rises above minimum elevation angle, as seen from groundstation
        Set: time the satellite sets below the minimum elevation angle, as seen from groundstation
        Visibility: The time between Rise and Set
        Offline: The time between Set and the next Rise of that satellite
    """
    # init time scale
    if ts is None:
        ts = sf.load.timescale()

    # load constellation
    #print('Loaded', len(satellites), 'satellites for starlink constellation with epoch', satellites[0].epoch.utc_jpl().removeprefix('A.D. '))

    # time span
    tspan = (startTime, stopTime)
#    print('Simulating from {} until {}'.format(*[s.utc_jpl().removeprefix('A.D. ') for s in tspan]))

    real_tspan = (ts.tt_jd(tspan[0].tt-safetyMargin), ts.tt_jd(tspan[1].tt+safetyMargin)) # safety margin
    #print('real_tspan'+str(real_tspan))

    # sat visibility finder
    def satVisibility(sat):
        t, events = sat.find_events(groundstation, real_tspan[0], real_tspan[1], altitude_degrees = minimumElevation)
        #print('real_tspan[0] = '+str(real_tspan[0])+'real_tspan[1] = '+str(real_tspan[1]))
        #print('t = '+str(t))
        #print('events = '+str(events))
        sat_rise = list()
        sat_set = list()
        lastevent = -1
        for ti, event, cnt in zip(t, events, range(len(events))):
            if lastevent == -1:
                if event != 0:
                    continue
                else:
                    lastevent = 2

            if event == 0: # rise
                if lastevent != 2:
                    if sat_rise[-1] == pd.Timestamp(ti.utc_iso()):
                        # work around bug in starfield
                        pass
                    else:
                        raise RuntimeError("satellite did not set")
                else:
                    lastevent = 0
                    sat_rise.append(pd.Timestamp(ti.utc_iso()))

            elif event == 1: # culminate
                if lastevent != 0 and lastevent != 1:
                    if abs((sat - groundstation).at(ti).altaz()[0].degrees - minimumElevation) <= 1:
                        # work around bug in starfield
                        sat_rise.append(pd.Timestamp(ti.utc_iso()))
                    elif events[cnt+1] == 0 and ti == t[cnt+1]:
                        # work around bug in starfield
                        sat_rise.append(pd.Timestamp(ti.utc_iso()))
                    else:
                        raise RuntimeError("satellite did not rise")

                lastevent = 1


            elif event == 2: # set
                if lastevent != 1:
                    raise RuntimeError("satellite did not culminate")
                lastevent = 2

                sat_set.append(pd.Timestamp(ti.utc_iso()))

            else:
                raise RuntimeError("unknown event")

        if event != 2:
            sat_rise.pop()

        df = pd.DataFrame({'Rise': sat_rise, 'Set': sat_set, 'Satellite': sat.name})

        # calculate offline times
        t = df['Rise'].copy()
        t.index = t.index-1
        t = t.drop(index=-1)
        df.insert(len(df.columns), 'Offline', t-df['Set'])

        # bracketing
        idx = np.logical_and(df['Set'] >= pd.Timestamp(tspan[0].utc_iso()), df['Rise'] <= pd.Timestamp(tspan[1].utc_iso()))

        # degenerate visits
        idx = np.logical_and(idx, df['Set'] != df['Rise'])

        # drop
        df = df[idx].reset_index().drop(columns='index')

        assert(np.all(df['Rise'] < df['Set']))
        return df

    # find satellite rise and set
    df = None
    with Bar(' ...', max = len(satellites), suffix = '%(index)d/%(max)d  ETA: %(eta)g s') as bar:
        for sat in satellites:
            if df is None:
                df = satVisibility(sat)
            else:
                df = df.append(satVisibility(sat))

            bar.next()

    # post process
    df.sort_values(by=['Rise','Set'], inplace=True)
    df.insert(2, 'Visibility', df['Set']-df['Rise'])
    df = df.set_index('Satellite')

    return df


def walkerConstellation(height, inclination, numSat, numPlanes, phasing, ts = None, name = "Sat", rE = 6371e3, gm = 3.986004418e14):
    """Walker Delta Pattern Constellation i: t/p/f

    i: inclination [deg]
    t: total number of satellites
    p: number of orbital planes (equally spaced)
    f: relative spacing between satellites in adjacent planes (0 .. p-1)

    height: oribtal height above Earth [m]
    inclincation: i [deg]
    numSat: t
    numPlanes: p
    phasing: f

    ts : skyfield timescale
    rE: Earth radius [m]
    gm: Earth gravitational constant [m^3 s^−2]
    """

    if ts is None:
        ts = sf.load.timescale()

    # check number of satellites
    S = numSat / numPlanes
    assert S == int(S), "numSat / numPlanes is not integer"
    S = int(S)

    # convert parameters
    ro = rE + height # orbital radius
    incRad = 80 * np.pi / 180
    motion = np.sqrt(gm / (ro)**3) * 60 # speed in radians / minute
    period_time = (2*np.pi)/(motion)    #Period time in minutes
    #print(' period_time_minute = ' + str(period_time))
    period_time_seconds = period_time * 60
    print(period_time_seconds)
    #print(' period_time_second = ' + str(period_time_seconds))

    # get epoch (now)
    epoch = datetime.datetime.fromisoformat("1949-12-31 00:00")
    diff = datetime.datetime.now() - epoch
    days = diff.days + diff.seconds/(24*3600)

    # build constellation
    satellites = list()
    cnt = 0


    if height == 500e3:
        raan1 = [0, 72, 144, 216, 288]
        print(raan1)
    elif height == 2000e3:
        raan1 = [36, 108, 180, 252, 324]


    for i in range(numPlanes):
        # formulas raan and ma taken from doi:10.3390/rs12111845

        # right ascension of the ascending nodes (RAAN)
        #raan = i / numPlanes * 2 * np.pi

        raan = raan1[i]*((2*np.pi)/360)
        print(raan)

        for j in range(S):
            cnt = cnt + 1
            # mean anomaly
            ma = 2 * (j / S + phasing * i / numSat) * np.pi

            satrec = Satrec()
            satrec.sgp4init(
                    WGS72,  # gravity model
                    'i',    # keep as is
                    cnt,      # increment per satellite
                    days,   # epoch (keep)
                    0,      # drag coefficient (idealized: 0)
                    0,      # not used, 0 is idealized
                    0,      # not used, 0 is idealized
                    0,      # eccentricity
                    0,      # argument of perigee (radians)
                    incRad,
                    ma,      # mean anomaly
                    motion, # mean motion
                    raan   # right ascension of ascending node
                ) # https://rhodesmill.org/skyfield/earth-satellites.html

            sat = sf.EarthSatellite.from_satrec(satrec, ts)
            sat.name = "{} {}".format(name, cnt)
            satellites.append(sat)

    return satellites


# Specify the satellites

ts = sf.load.timescale()
simulate_days = 1 # time span for simulation (epoch, epoch+simulate_days) / in days
#groundstation = sf.wgs84.latlon(+90, 0) # north pole
groundstation = sf.wgs84.latlon(+53.10373, +8.85132) # Bremen
#print(groundstation)
min_elev = 10
Num_shell = 2
Num_sat = 5

satellites = walkerConstellation(500e3, 80, Num_sat, 5, 1, ts = None, name = "Sat", rE = 6371e3, gm = 3.986004418e14)
df = simulateConstellation(satellites, groundstation, min_elev, satellites[0].epoch, ts.tt_jd(satellites[0].epoch.tt + simulate_days), ts)
start_time_main = satellites[0].epoch.utc_iso()
start_time = start_time_main[11:-1]

#print(start_time)
finish_time_main = ts.tt_jd(satellites[0].epoch.tt + simulate_days).utc_iso()
finish_time = finish_time_main[11:-1]
#print(finish_time)

#print(df)
satellites1 = walkerConstellation(2000e3, 80, Num_sat, 5, 1, ts = None, name = "Sat", rE = 6371e3, gm = 3.986004418e14)
df1 = simulateConstellation(satellites1, groundstation, min_elev, satellites[0].epoch, ts.tt_jd(satellites[0].epoch.tt + simulate_days), ts)
#print(df1)


period_time_second = 7622.141262852221


Sats_name = ['Sat 1','Sat 2','Sat 3','Sat 4','Sat 5']

# df for satellites in altitude 500km and df1 for the ones in altitude 2000km
DFs = [df ,df1] # To make a matrix consisting of satellites from shell 1 and shell 2
sat_arr = []
sat_arr_time = []
for num_shell_index in range(Num_shell):
 DF = DFs[num_shell_index]
 for i in range(len(Sats_name)):
   rise_df1 = DF.loc[Sats_name[i]]  # Classify each satellite information
   rise_df1 = rise_df1['Rise']  # Only put the 'Rise' information of satellites
   arr1 = []
   arr2 = []
   for ii in range(len(rise_df1)):
        arr1.append(rise_df1[ii].value) # One array with the size of (Num_sat*) with the rise time of satellites
        arr2.append(rise_df1[ii]) # One array with the size of (Num_sat*) with the rise time of satellites
   sat_arr.append(arr1)
   sat_arr_time.append(arr2)

#print('sat_arr '+str(sat_arr))
#print('sat_arr_time '+str(sat_arr_time))
#####
sat_arr_hour = copy.deepcopy(sat_arr)
for i in range(len(sat_arr)):
    for j in range(len(sat_arr[i])):
       sat_arr_hour[i][j] = sat_arr_time[i][j].strftime("%X")  #To derive the hour, minute and seconds

#print('sat_arr_hour = '+str(sat_arr_hour))  # Derive the time of satellites in terms of hour:minute:second

#####
FMT = '%H:%M:%S'
sat_arr_sec_async = []
for i in range(len(sat_arr)):
    sat_arr_sec_async.append([0]*((len(sat_arr[i])-1))) #Make the matrix for making the differencing between each time and last time for async


#####
for i_sec in range(len(sat_arr)):
    for j_sec in range(1, (len(sat_arr[i_sec]))):
        tdelta = datetime.datetime.strptime(sat_arr_hour[i_sec][j_sec], FMT) - datetime.datetime.strptime(sat_arr_hour[i_sec][j_sec-1], FMT)
        if tdelta.days < 0:
            tdelta = timedelta(
                days=0,
                seconds=tdelta.seconds,
                microseconds=tdelta.microseconds
            )
        #tdelta = (tdelta.total_seconds() % 3600 ) // 60
        sat_arr_sec_async[i_sec][j_sec-1] = tdelta.total_seconds()  #Derive the differencing between each time and last time for async in terms of second

#print('sat_arr_sec_async = ' + str(sat_arr_sec_async))

#####################################################################################
#print(sat_arr)
## Calculate the number of updates of all satellites
len_time = 0
for i in range(len(sat_arr)):
     len_time += len(sat_arr[i])


## Derive the index of satellites based on the rise time of satellites
## Derive the rise time of satellites
satellite_indexes_rise = []
satellite_time_rise = []
satellite_time_rise_datetime = []
for time in range(len_time):
    #print(' sat_arr = '+str(sat_arr))

     A = len(sat_arr)
     for index_satellite in range(len(sat_arr)):
         if len(sat_arr[index_satellite]) == 0: # Not to change the order of satellites
            sat_arr[index_satellite] = [100000000000000000000000]
            sat_arr_time[index_satellite] = [100000000000000000000000]


     min_value_vector = []
     min_value_vector_UTC = []
     sum_check = []
     for yy in range(len(sat_arr)):
         #print('sat_arr[yy] = '+str(sat_arr[yy]))
         if len(sat_arr[yy]) == 1:
             #print('sat_arr[yy] = '+str(sat_arr[yy]))
             if sat_arr[yy] == [100000000000000000000000]:
                sum_check.append(sat_arr[yy])


     for index_satellite in range(len(sat_arr)):
        min_value = min(sat_arr[index_satellite])    #Find the minimum time of rise time for each satellite
        min_value_UTC = min(sat_arr_time[index_satellite])
        min_value_vector.append(min_value)  #min_value_vector consists of the minimum rise time of each satellite
        min_value_vector_UTC.append(min_value_UTC)  #min_value_vector consists of the minimum rise time of each satellite


     sort_min_value_vector = np.argsort(min_value_vector)   # Sort the index of satellites based on the rise time of satellites
     satellite_indexes_rise.append(sort_min_value_vector[0])    # Add the index of satellite
     x = min_value_vector_UTC[sort_min_value_vector[0]].strftime("%X")  #To derive the hour, minute and seconds
     #x1 = min_value_vector_UTC[sort_min_value_vector[0]].strftime("%M")  #To derive the hour, minute and seconds
     #print(x1)
     satellite_time_rise.append(x)
     sat_arr[sort_min_value_vector[0]] = np.delete(sat_arr[sort_min_value_vector[0]], 0)
     sat_arr_time[sort_min_value_vector[0]] = np.delete(sat_arr_time[sort_min_value_vector[0]], 0)



#print('satellite_time_rise = ' + str(satellite_time_rise))
#print(int(satellite_time_rise[1]).strftime("%f"))

satellite_time_rise = [start_time] + satellite_time_rise

FMT = '%H:%M:%S'



difference = []
for i in range(1,len(satellite_time_rise)):
        #print(i)
        tdelta = datetime.datetime.strptime(satellite_time_rise[i], FMT) - datetime.datetime.strptime(satellite_time_rise[i-1], FMT)
        if tdelta.days < 0:
            tdelta = timedelta(
                days=0,
                seconds=tdelta.seconds,
                microseconds=tdelta.microseconds
            )
        #tdelta = (tdelta.total_seconds() % 3600 ) // 60
        tdelta = tdelta.total_seconds()
        difference.append(tdelta) # Difference between times of visting satellites in seconds
#for i in len(satellite_time_rise):

period_time_second = 7622.141262852221
#print('satellite_indexes_rise = ' + str(satellite_indexes_rise))
#print('satellite_time_rise'+str(satellite_time_rise) )
#print('difference = ' + str(difference))


# Test
#sat_arr_sec_async = [[5671.0, 5673.0, 5671.0], [5674.0, 5671.0, 5672.0], [5672.0, 5671.0], [5672.0, 5671.0, 5672.0], [5671.0, 5672.0, 5671.0], [7626.0, 7625.0], [7626.0], [7626.0], [7626.0, 7625.0], [7625.0, 7626.0]]
#satellite_indexes_rise = [1, 0, 5, 4, 9, 3, 8, 2, 1, 7, 0, 6, 4, 5, 3, 2, 9, 1, 8, 0, 7, 4, 3, 6, 2, 5, 1, 9, 0, 4, 8, 3]
#difference = [124.0, 1135.0, 12.0, 1122.0, 403.0, 731.0, 794.0, 340.0, 1137.0, 48.0, 1084.0, 441.0, 693.0, 833.0, 302.0, 1134.0, 88.0, 1048.0, 478.0, 656.0, 869.0, 264.0, 1134.0, 127.0, 1007.0, 518.0, 619.0, 906.0, 227.0, 1133.0, 165.0, 970.0]



# North pole with 5 orbits in each shell
#sat_arr_sec_async = [[5673.0, 5672.0, 5671.0, 5672.0, 5671.0, 5671.0, 5671.0, 5673.0, 5671.0, 5672.0, 5671.0, 5671.0, 5671.0, 5673.0], [5674.0, 5671.0, 5672.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0], [5672.0, 5672.0, 5671.0, 5671.0, 5672.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0], [5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0], [5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5671.0, 5672.0, 5672.0, 5671.0, 5671.0, 5672.0], [7626.0, 7626.0, 7625.0, 7625.0, 7625.0, 7625.0, 7625.0, 7626.0, 7625.0, 7625.0], [7626.0, 7625.0, 7625.0, 7625.0, 7625.0, 7626.0, 7625.0, 7625.0, 7625.0, 7625.0], [7626.0, 7625.0, 7625.0, 7625.0, 7625.0, 7626.0, 7625.0, 7625.0, 7625.0, 7625.0], [7626.0, 7625.0, 7625.0, 7625.0, 7625.0, 7625.0, 7626.0, 7625.0, 7625.0, 7625.0], [7626.0, 7625.0, 7625.0, 7625.0, 7625.0, 7626.0, 7625.0, 7625.0, 7625.0, 7625.0]]   #North Pole
#satellite_indexes_rise = [1, 0, 5, 4, 9, 3, 8, 2, 1, 7, 0, 6, 4, 5, 3, 2, 9, 1, 8, 0, 7, 4, 3, 6, 2, 5, 1, 9, 0, 4, 8, 3, 7, 2, 6, 1, 0, 5, 4, 9, 3, 8, 2, 1, 7, 0, 6, 4, 5, 3, 2, 9, 1, 8, 0, 7, 4, 3, 6, 2, 5, 1, 9, 0, 4, 8, 3, 7, 2, 6, 1, 0, 5, 4, 9, 3, 2, 8, 1, 7, 0, 6, 4, 3, 5, 2, 9, 1, 8, 0, 4, 7, 3, 6, 2, 5, 1, 0, 9, 4, 8, 3, 7, 2, 1, 6, 0, 5, 4, 9, 3, 2, 8, 1, 7, 0, 6, 4, 3, 5, 2, 9, 1, 8, 0, 4, 7, 3, 6, 2, 1] #North Pole
#difference = [124.0, 1134.0, 12.0, 1123.0, 402.0, 732.0, 793.0, 341.0, 1137.0, 47.0, 1086.0, 439.0, 694.0, 832.0, 303.0, 1134.0, 88.0, 1048.0, 477.0, 657.0, 868.0, 265.0, 1134.0, 126.0, 1009.0, 517.0, 619.0, 905.0, 228.0, 1133.0, 164.0, 971.0, 554.0, 580.0, 945.0, 192.0, 1133.0, 201.0, 932.0, 592.0, 542.0, 983.0, 151.0, 1137.0, 237.0, 896.0, 629.0, 504.0, 1022.0, 113.0, 1134.0, 277.0, 860.0, 665.0, 467.0, 1058.0, 76.0, 1134.0, 315.0, 820.0, 706.0, 430.0, 1094.0, 38.0, 1134.0, 353.0, 782.0, 743.0, 391.0, 1134.0, 3.0, 1133.0, 390.0, 743.0, 782.0, 352.0, 1135.0, 37.0, 1099.0, 427.0, 706.0, 819.0, 314.0, 1135.0, 76.0, 1058.0, 467.0, 670.0, 855.0, 278.0, 1133.0, 114.0, 1020.0, 505.0, 630.0, 896.0, 240.0, 1133.0, 151.0, 983.0, 542.0, 592.0, 933.0, 201.0, 1137.0, 187.0, 945.0, 581.0, 553.0, 971.0, 163.0, 1135.0, 227.0, 909.0, 616.0, 516.0, 1009.0, 125.0, 1135.0, 266.0, 868.0, 656.0, 481.0, 1044.0, 89.0, 1133.0, 303.0, 831.0, 694.0, 441.0, 1136.0] #North Pole



# Bremen with 5 orbits in each shell
sat_arr_sec_async = [[5940.0, 29206.0, 5800.0, 44121.0], [5831.0, 29316.0, 5752.0], [5757.0, 29387.0, 5714.0], [44252.0, 5684.0, 29451.0], [5652.0, 44329.0, 5597.0], [7473.0, 7887.0, 8219.0, 8056.0, 7781.0, 7743.0, 7828.0, 29040.0], [7203.0, 7708.0, 8092.0, 8210.0, 7863.0, 7745.0, 7766.0], [7513.0, 7922.0, 8232.0, 8021.0, 7771.0, 7744.0], [7741.0, 7811.0, 29127.0, 7268.0, 7743.0, 8122.0, 8188.0], [8227.0, 7888.0, 7748.0, 7760.0, 36588.0, 7552.0]] #Bremen
satellite_indexes_rise = [5, 0, 9, 3, 8, 0, 5, 9, 8, 4, 6, 5, 1, 9, 4, 8, 6, 1, 5, 9, 6, 5, 2, 9, 7, 0, 6, 2, 5, 0, 7, 6, 5, 3, 8, 7, 1, 6, 3, 5, 8, 1, 7, 6, 4, 8, 7, 2, 6, 4, 9, 8, 2, 7, 9, 8, 7, 3, 5, 0] #Bremen
difference = [625.0, 9.0, 1869.0, 1545.0, 906.0, 1620.0, 1524.0, 2632.0, 1965.0, 1562.0, 374.0, 1354.0, 521.0, 2112.0, 1291.0, 597.0, 1328.0, 503.0, 1867.0, 2162.0, 3176.0, 2718.0, 133.0, 1733.0, 1541.0, 113.0, 1854.0, 516.0, 1891.0, 1539.0, 1600.0, 2664.0, 1940.0, 516.0, 1333.0, 1469.0, 551.0, 2054.0, 277.0, 1628.0, 1289.0, 504.0, 1929.0, 2118.0, 2786.0, 406.0, 2711.0, 182.0, 1681.0, 617.0, 879.0, 2052.0, 485.0, 1875.0, 3140.0, 2688.0, 1916.0, 565.0, 1217.0, 1049.0] #Bremen

#print('sat_arr_sec_async = ' +str(sat_arr_sec_async))
#print('satellite_indexes_rise = ' + str(satellite_indexes_rise))
#print('difference = ' + str(difference))
def Period_fedasync():
    return period_time_second

def satellite_rise_difference_for_fedasync():
    return sat_arr_sec_async

def visibility_satellite_GS_matrix():
    return satellite_indexes_rise

def satellite_GS_matrix_time_difference():
    return difference
