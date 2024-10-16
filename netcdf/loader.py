import itertools
import numpy
import netCDF4
import pandas


def make_wind( dataset, check_shape ):
    # Three variables are crs, lon, lat
    # The rest are times
    dim_tim = len(dataset.variables) - 3
    assert (check_shape == None) or (check_shape[0] == dim_tim)
    dim_lat = dataset["lat"].shape[0]
    assert (check_shape == None) or (check_shape[1] == dim_lat)
    dim_lon = dataset["lon"].shape[0]
    assert (check_shape == None) or (check_shape[2] == dim_lon)

    ll = []
    for i in range(1,dim_tim+1):
        # At each timestep, there is an array with shape (dim_lat,dim_lon)
        key = f"Band{i}"
        # Must go through numpy array, or the list looks strange
        ll.append( numpy.array(dataset[key]).astype(list) )
    arr = numpy.array(ll)    
    # arr must be time x lat x lon
    assert (check_shape == None) or (check_shape == arr.shape)
    return arr


def make_new_wind( dataset, check_shape ):
    # The file is lon x lat x sfc(time)
    # Before returning,
    # transpose to time x lat x lon to be the same as waves
    dim_tim = dataset["sfc"].shape[0]
    assert (check_shape == None) or (check_shape[0] == dim_tim)
    dim_lat = dataset["lat"].shape[0]
    assert (check_shape == None) or (check_shape[1] == dim_lat)
    dim_lon = dataset["lon"].shape[0]
    assert (check_shape == None) or (check_shape[2] == dim_lon)
    return numpy.array( dataset["Band1"] )


def make_waves( dataset, check_shape ):
    # "VHM0", "VTM01_WW", "VMDR_WW"
    # with three dimensions: time x latitude x longitude
    vhm = numpy.array( dataset["VHM0"] )
    assert (check_shape == None) or (check_shape == vhm.shape)
    vtm = numpy.array( dataset["VTM01_WW"] )
    assert (check_shape == None) or (check_shape == vtm.shape)
    vmdr = numpy.array( dataset["VMDR_WW"] )
    assert (check_shape == None) or (check_shape == vmdr.shape)
    return vhm,vtm,vmdr


def make_space( dataset, check_shape ):
    key = None
    if "latitude" in dataset.dimensions.keys(): key = "latitude"
    elif "lat" in dataset.dimensions.keys(): key = "lat"
    if key == None: lat = None
    else: lat = numpy.array( dataset[key] )

    key = None
    if "longitude" in dataset.dimensions.keys(): key = "longitude"
    elif "lon" in dataset.dimensions.keys(): key = "lon"
    if key == None: lon = None
    else: lon = numpy.array( dataset[key] )

    return lat,lon


def make_time( dataset, check_shape ):
    key = None
    if "time" in dataset.dimensions.keys(): key = "time"
    if key == None: tim = None
    else: tim = numpy.array( dataset[key] )
    return tim


def old_files():
    fname = "data/med-hcmr-wav-rean-h_multi-vars_23.00E-26.96E_34.02N-39.98N_2020-08-10-2020-08-18.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF4" )
    vhm,vtm,vmdr = make_waves( f, (216,144,96) )
    tim = make_time( f, (216) )
    f.close()

    fname = "data/wind_dir.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
    wind_dir = make_wind( f, vhm.shape )
    lat,lon = make_space( f, (144,96) )
    f.close()

    fname = "data/wind_speed.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
    wind_speed = make_wind( f, wind_dir.shape )
    f.close()

    return tim,lat,lon,vhm,vtm,vmdr,wind_dir,wind_speed

def new_files():
    fname = "data/med-hcmr-wav-rean-h_multi-vars_23.00E-26.96E_34.02N-39.98N_2019-01-21-2019-01-26.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF4" )
    vhm,vtm,vmdr = make_waves( f, (144,144,96) )
    #lat,lon = make_space( f, (144,96) )
    tim = make_time( f, (144) )
    f.close()

    fname = "data/wind_dir.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
    wind_dir = make_new_wind( f, (144,144,96) )
    lat,lon = make_space( f, (144,96) )
    f.close()

    fname = "data/wind_speed.nc"
    f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
    wind_speed = make_new_wind( f, (144,144,96) )
    f.close()

    return tim,lat,lon,vhm,vtm,vmdr,wind_dir,wind_speed


tim,lat,lon,vhm,vtm,vmdr,wind_dir,wind_speed = new_files()

# Now make an array time x lat x lon x var
# where var is 0: vhm, 1: vtm, 2: vmdr, 3: wind_dir, 4: wind_speed
cube = numpy.stack( [vhm,vtm,vmdr,wind_dir,wind_speed], axis=-1 )

# Now make the dimensions
dims = numpy.vstack( list(itertools.product(tim,lat,lon)) )

# Put everything together
# New array is time x lat x long x var

#dataframe = numpy.concatenate(
#    (dims.reshape(216,144,96,3), cube),
#    axis=3 )

dataframe = numpy.concatenate(
    (dims.reshape(144,144,96,3), cube),
    axis=3 )

#df = pandas.DataFrame(
#    dataframe.reshape( (216*144*96,8) ),
#    columns=["TIME","LAT","LON","VHM0","VTM01_WW","VMDR_WW","wind_dir","wind_speed"] )
df = pandas.DataFrame(
    dataframe.reshape( (144*144*96,8) ),
    columns=["TIME","LAT","LON","VHM0","VTM01_WW","VMDR_WW","wind_dir","wind_speed"] )

df.VHM0 = df.VHM0.replace(to_replace=-32767.0, value=-999.0)
df.VTM01_WW = df.VTM01_WW.replace(to_replace=-32767.0, value=-999.0)
df.VMDR_WW = df.VMDR_WW.replace(to_replace=-32767.0, value=-999.0)

df = df.astype('float64').round( {"LAT": 3, "LON":3,
                                  "VHM0": 2, "VTM01_WW": 3, "VMDR_WW": 0,
                                  "wind_dir": 0, "wind_speed": 1} )
df.TIME = df.TIME.astype(int)
df.wind_dir = df.wind_dir.astype(int)

# Alternative, but creates dep on jinja2
#df.style.format( {"LAT": "{:.3f}", "LON": "{:.3f}",
#                  "VHM0": "{:.2f}", "VTM01_WW": "{:.3f}",
#                  "wind_dir": "{:d}", "wind_speed": "{:.1f}"} )

df.to_csv("data.csv")
