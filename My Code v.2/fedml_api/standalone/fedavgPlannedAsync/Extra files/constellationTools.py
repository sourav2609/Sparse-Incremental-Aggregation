import skyfield.api as sf
from sgp4.api import Satrec, WGS72
import numpy as np
import pandas as pd
from progress.bar import Bar
from datetime import datetime

def simulateConstellation(satellites, groundstation, minimumElevation, startTime, stopTime, ts = None, safetyMargin = 1):
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
    print('Simulating from {} until {}'.format(*[s.utc_jpl().removeprefix('A.D. ') for s in tspan]))

    real_tspan = (ts.tt_jd(tspan[0].tt-safetyMargin), ts.tt_jd(tspan[1].tt+safetyMargin)) # safety margin
    print('real_tspan'+str(real_tspan))

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

    # get epoch (now)
    epoch = datetime.fromisoformat("1949-12-31 00:00")
    diff = datetime.now() - epoch
    days = diff.days + diff.seconds/(24*3600)

    # build constellation
    satellites = list()
    cnt = 0
    for i in range(numPlanes):
        # formulas raan and ma taken from doi:10.3390/rs12111845

        # right ascension of the ascending nodes (RAAN)
        raan = i / numPlanes * 2 * np.pi

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
