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
    if "latitude" in f.dimensions.keys(): key = "latitude"
    elif "lat" in f.dimensions.keys(): key = "lat"
    if key == None: lat = None
    else: lat = numpy.array( dataset[key] )

    key = None
    if "longitude" in f.dimensions.keys(): key = "longitude"
    elif "lon" in f.dimensions.keys(): key = "lon"
    if key == None: lon = None
    else: lon = numpy.array( dataset[key] )

    return lat,lon


def make_time( dataset, check_shape ):
    key = None
    if "time" in f.dimensions.keys(): key = "time"
    if key == None: tim = None
    else: tim = numpy.array( dataset[key] )
    return tim



fname = "data/med-hcmr-wav-rean-h_multi-vars_23.00E-26.96E_34.02N-39.98N_2020-08-10-2020-08-18.nc"
f = netCDF4.Dataset( fname, "r", format="NETCDF4" )
vhm,vtm,vmdr = make_waves( f, (216,144,96) )
lat,lon = make_space( f, (144,96) )
tim = make_time( f, (216) )
f.close()

fname = "data/wind_dir.nc"
f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
wind_dir = make_wind( f, vhm.shape )
f.close()

fname = "data/wind_speed.nc"
f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
wind_speed = make_wind( f, wind_dir.shape )
f.close()

# Now make an array time x lat x lon x var
# where var is 0: vhm, 1: vtm, 2: vmdr, 3: wind_dir, 4: wind_speed
cube = numpy.stack( [vhm,vtm,vmdr,wind_dir,wind_speed], axis=-1 )

# Now make the dimensions
dims = numpy.vstack( list(itertools.product(tim,lat,lon)) )

# Put everything together
# New array is time x lat x long x var
dataframe = numpy.concatenate(
    (dims.reshape(216,144,96,3), cube),
    axis=3 )
df = pandas.DataFrame(
    dataframe.reshape( (216*144*96,8) ),
    columns=["TIME","LAT","LON","VHM0","VTM01_WW","VMDR_WW","wind_dir","wind_speed"] )
df.to_csv("data.csv")
