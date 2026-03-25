import numpy as np
import datetime
from sgp4.api import Satrec, WGS72
from skyfield.api import load, wgs84
import skyfield.api as sf
from GEO import LEO_GEO
import matplotlib.pyplot as plt
import copy
from matplotlib.ticker import AutoMinorLocator


if __name__ == '__main__':

    h_LEO = 2000e3
    inclination = 60
    inc_GEO = 0
    numSat = 40
    numPlanes = 5
    phasing = 1
    r_E = 6371e3
    gm = 3.986004418e14
    h_GEO = 20000e3



    LG = LEO_GEO(r_E, gm, h_GEO)
    GEO = LG.GEO(inc_GEO)
    LEOs = LG.walkerConstellation(h_LEO, inclination, numSat, numPlanes, phasing, name = "Sat")


    ##################### Seperate the satellites in each plane ##########################
    LEO_sats_in_plane = []
    for i in range(numPlanes):
        a = int(numSat/numPlanes)
        LEO_sats_in_plane.append(LEOs[i*a: a + i * a])


    ############################start and end time of satellites simulation#################################

    ts = sf.load.timescale()
    time1 = ts.utc(2021,9,23,00,00,00)      # Start point, UTC time
    time2 = ts.utc(2021,9,23,24,00,00)      # End point, UTC time
    seconds_difference_time = LG.difference_time_in_seconds(time1, time2) # How many seconds between start and end

    ####################################*********** GS as the server **************##########################################

    groundstation = wgs84.latlon(+53.00, 8.80) #GS-BR
    #groundstation = wgs84.latlon(+90.0, 0.0) #GS-NP
    DF = LG.simulateConstellation(LEOs, groundstation, 10, time1, time2, ts = None, safetyMargin = 0)

    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []

    Sats_name = []
    for i in range(1,numSat+1):
        Sats_name.append('Sat '+str(i))

    for i in range(len(LEOs)):
       rise_df = []
       rise_df = DF.loc[Sats_name[i]]  # Classify each satellite information

       rise_df1 = rise_df['Rise']  # Only put the 'Rise' information of satellites
       end_time_df1 = rise_df['end_time']  # Only put the 'end_time' information of satellites

       arr1_start = []
       arr1_end = []
       distancee = []

       #print(f"rise_df1 = {rise_df1}")
       for ii in range(len(rise_df1)):

            t_r = rise_df1[ii]  # The moment of rising
            t_e = end_time_df1[ii]


            time_rise_now = ts.utc(t_r.year, t_r.month, t_r.day, t_r.hour, t_r.minute, t_r.second)

            position_LEO = LEOs[i].at(time_rise_now)
            position_GS = groundstation.at(time_rise_now)
            position_LEO = LEOs[i].at(time_rise_now)
            distancee.append(LG.distance(position_GS.position.km, position_LEO.position.km))
            time_end_now = ts.utc(t_e.year, t_e.month, t_e.day, t_e.hour, t_e.minute, t_e.second)
            #time_offline_now = ts.utc(t_off.year, t_off.month, t_off.day, t_off.hour, t_off.minute, t_off.second)
            arr1_start.append(LG.difference_time_in_seconds(time1, time_rise_now)) # One array with the size of (Num_sat*) with the rise time of satellites
            arr1_end.append(LG.difference_time_in_seconds(time1, time_end_now)) # One array with the size of (Num_sat*) with the rise time of satellites


       visiting_start_GS_LEO.append(arr1_start)
       visiting_end_GS_LEO.append(arr1_end)
       distance_GS_LEOs.append(distancee)
       #print(f"distance_GS_LEOs = {distance_GS_LEOs}")





#set_idx_sat = [8,9,10,11,12,13,14,15]
#set_idx_sat = [0, 4, 9, 13, 18, 21, 25, 27 , 29, 35, 38,39]
set_idx_sat = [1, 5, 9, 13, 17, 21, 25, 29 , 31, 33, 37,39]
#plt.yticks(ylim, [ 1 , 5,  10, 14, 19, 22,26,28, 30, 36, 39,40])
xmin = []
xmax = []
y = []
jmin = []
jmax = []
output = []
a = 0
for i in range(len(set_idx_sat)):
    xmin.append(visiting_start_GS_LEO[set_idx_sat[i]])
    xmax.append(visiting_end_GS_LEO[set_idx_sat[i]])
    print(f"xmin, xmax = {xmin,xmax}")
    y.append([a for i in range(len(visiting_start_GS_LEO[set_idx_sat[i]]))])

    jmin.append(visiting_end_GS_LEO[set_idx_sat[i]])
    jmax.append(visiting_start_GS_LEO[set_idx_sat[i]])
    output.append([a for i in range(len(visiting_start_GS_LEO[set_idx_sat[i]]))])
    a += 1
#print(y)
#print(f"xmin = {xmin}")
#print(f"xmax = {xmax}")
xmin1 = copy.deepcopy(xmin)
xmax1 = copy.deepcopy(xmax)


for idxx in range(len(output)):
    output[idxx].append(output[idxx][-1])

for i in range(len(jmin)):
    jmin[i].append(0)
    jmin[i].sort()
    jmax[i].append(24*3600)
    jmax[i].sort()

#print(f"xmin = {xmin}")
#print(f"xmax = {xmax}")
#print(f"jmin = {jmin}")
#print(f"jmax = {jmax}")
#print(f"xmin1 = {xmin1}")
#print(f"xmax1 = {xmax1}")
#colors=['cornflowerblue', 'cornflowerblue', 'cyan', 'cyan', 'olive', 'olive', 'burlywood', 'burlywood' , 'burlywood', 'fuchsia', 'fuchsia', 'fuchsia' ]

colors  = []
colors1  = []
for i in range(len(set_idx_sat)):
    colors.append('lightseagreen')
    colors1.append('pink')

for i in range(len(y)):
    if set_idx_sat[i]==1:
        plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i], label="Visible to PS",linewidth=10.0)
        plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],label="Non-visible to PS",linewidth=10.0)
    elif 2<=set_idx_sat[i]<=7 :
      plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i], linewidth=10.0)
      plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],linewidth=10.0)
    elif 8<=set_idx_sat[i]<=15:
      plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i],linewidth=10.0)
      plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],linewidth=10.0)
    elif 16<=set_idx_sat[i]<=23:
      plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i],linewidth=10.0)
      plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],linewidth=10.0)
    elif 24<=set_idx_sat[i]<=31:
      plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i],linewidth=10.0)
      plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],linewidth=10.0)
    elif 32<=set_idx_sat[i]<=40:
      plt.hlines(y[i], xmin1[i], xmax1[i], colors=colors[i],linewidth=10.0)
      plt.hlines(output[i], jmin[i], jmax[i], colors=colors1[i],linewidth=10.0)

plt.xlim([0, 43200])
plt.ylim([-0.5, 13])
ylim = np.linspace(0, 11, 12)
#print(ylim)
#plt.yticks(ylim, [ "S" + str(1) , "S" + str(2),  "S" + str(3), "S" + str(4), "S" + str(5), "S" + str(6),"S" + str(7), "S" + str(8), "S" + str(9), "S" + str(10),"S" + str(11), "S" + str(12)])
#plt.yticks(ylim, [ "k = " + str(1) , "k = " + str(5),  "k = " + str(10), "k = " + str(14), "k = " + str(19), "k = " + str(22),"k = " + str(26),"k = " + str(28), "k = " + str(30), "k = " + str(36), "k = " + str(39),"k = " + str(40)])
#plt.yticks(ylim, [ 1 , 5,  10, 14, 19, 22,26,28, 30, 36, 39,40])
plt.yticks(ylim, [r'$k_{1,2}$', r"$k_{1,6}$", r"$k_{2,2}$",r"$k_{2,6}$",r"$k_{3,2}$"
                     , r'$k_{3,6}$',r"$k_{4,2}$",r"$k_{4,6}$",r"$k_{4,8}$"
                     ,r"$k_{5,2}$",r"$k_{5,6}$",r"$k_{5,8}$"])

xlim = np.linspace(0, 12*60*60, 7)
plt.xticks(xlim, ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00'])
ax = plt.gca()
#plt.grid(b=True, which='major', color='k', linestyle='--')
#plt.grid(b=True, which='minor', color='darkgray', linestyle='--')
ax.xaxis.grid(b=True, which='major', color='darkgray', linestyle='--')
ax.xaxis.grid(b=True, which='minor', color='darkgray', linestyle='--')
#ax.yaxis.grid(b=True, which='major', color='darkgray', linestyle='--')

ax.xaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_tick_params(which='minor', bottom=False)
#ax.legend([line1, line2], ['label1', 'label2'])
plt.xlabel('Time [h]')
plt.ylabel('Satellite ID')
#bbox_to_anchor=(0.50, 1.00),
plt.legend(loc='upper right',
          ncol=3, fancybox=False, shadow=False)
plt.show()













'''
visiting_start = [[] for i in range(len(set_idx_sat))]
start = [[] for i in range(len(set_idx_sat))]


for idx_sat in range(len(set_idx_sat)):
  combine = []
  combine.append(visiting_start_GS_LEO[set_idx_sat[idx_sat]])
  combine.append(visiting_end_GS_LEO[set_idx_sat[idx_sat]])
  visiting_start[idx_sat]= sum(combine, [])
  visiting_start[idx_sat].sort()
  print(visiting_start[idx_sat])
  if visiting_start[idx_sat][0] != 0:
     print(f"idx_sat = {idx_sat}")
     visiting_start[idx_sat].append(0)
     visiting_start[idx_sat].sort()
     if visiting_start[idx_sat][-1] != seconds_difference_time:
         visiting_start[idx_sat].append(seconds_difference_time)

     for i in range(len(visiting_start[idx_sat])):
         #print(i)

         if i % 2 == 0:
           start[idx_sat].append(1)
         elif i % 2 == 1:
           start[idx_sat].append(0)
  else:
      if visiting_start[idx_sat][-1] != seconds_difference_time:
               visiting_start[idx_sat].append(seconds_difference_time)
      for i in range(len(visiting_start[idx_sat])):
         if i%2 == 0:
           start[idx_sat].append(1)
         elif i//2 == 1:
           start[idx_sat].append(0)

print(f"visiting_start = {visiting_start}")
print(f"start = {start}")

for i in range(len(start)):
  plt.step(visiting_start[i], start[i] , label="Satellite in orbit "+str(i))
'''
