import netCDF4
import numpy


def make_wind( fname, check_shape ):
    f = netCDF4.Dataset( fname, "r", format="NETCDF3" )
    # Three variables are crs, lon, lat
    # The rest are times
    dim_tim = len(f.variables) - 3
    assert (check_shape == None) or (check_shape[0] == dim_tim)
    dim_lat = f["lat"].shape[0]
    assert (check_shape == None) or (check_shape[1] == dim_lat)
    dim_lon = f["lon"].shape[0]
    assert (check_shape == None) or (check_shape[2] == dim_lon)

    ll = []
    for i in range(1,dim_tim+1):
        # At each timestep, there is an array with shape (dim_lat,dim_lon)
        key = f"Band{i}"
        # Must go through numpy array, or the list looks strange
        ll.append( numpy.array(f[key]).astype(list) )
    arr = numpy.array(ll)    
    # arr must be time x lat x lon
    assert (check_shape == None) or (check_shape == arr.shape)
    f.close()
    return arr



def make_waves( fname, check_shape ):
    f = netCDF4.Dataset( fname, "r", format="NETCDF4" )
    # "VHM0", "VTM01_WW", "VMDR_WW"
    # with three dimensions: time x latitude x longitude
    vhm = numpy.array( f["VHM0"] )
    assert (check_shape == None) or (check_shape == vhm.shape)
    vtm = numpy.array( f["VTM01_WW"] )
    assert (check_shape == None) or (check_shape == vtm.shape)
    vmdr = numpy.array( f["VMDR_WW"] )
    assert (check_shape == None) or (check_shape == vmdr.shape)
    return vhm,vtm,vmdr

fname = "data/med-hcmr-wav-rean-h_multi-vars_23.00E-26.96E_34.02N-39.98N_2020-08-10-2020-08-18.nc"
vhm,vtm,vmdr = make_waves( fname, (216,144,96) )

fname = "data/wind_dir.nc"
wind_dir = make_wind( fname, vhm.shape )

fname = "data/wind_speed.nc"
wind_speed = make_wind( fname, wind_dir.shape )

# Now make an array time x lat x lon x var
# where var is 0: vhm, 1: vtm, 2: vmdr, 3: wind_dir, 4: wind_speed
cube = numpy.stack( [vhm,vtm,vmdr,wind_dir,wind_speed], axis=-1 )
