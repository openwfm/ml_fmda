# Functions used for conventient plotting and mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Credit to Brian Blaylock for EasyMap within Herbie project

from herbie import paint
from herbie.toolbox import EasyMap, pc, ccrs
import matplotlib.pyplot as plt


# Dictionary to store mapping schemes, eg cmaps and titles
## Using NWS color maps from the herbie package
## Moisture vars, like equil and FMC, use NWS humidity scheme
map_dict = {
    'fm': {'cmap': paint.NWSRelativeHumidity.cmap, 
           'legend_title': 'Fuel Moisture Content (%)'},
    'Ed': {'cmap': paint.NWSRelativeHumidity.cmap, 
           'legend_title': 'Drying Equilibrium Moisture Content (%)'},
    'Ew': {'cmap': paint.NWSRelativeHumidity.cmap, 
           'legend_title': 'Wetting Equilibrium Moisture Content (%)'},
    'rh': {'cmap': paint.NWSRelativeHumidity.cmap, 
           'legend_title': 'Relative Humidity (%)',
           'xarray_name': "r2"
          },
    'temp': {'cmap': paint.NWSTemperature.cmap, 
             'legend_title': 'Air Temperature (K)',
             'xarray_name': "t2m"
            },
    'rain': {'cmap': paint.NWSPrecipitation.cmap, 
             'legend_title': 'Hourly Precipitation (mm/hr)',
             'xarray_name': "tp"
            },
    'elev': {'cmap': paint.LandGreen.cmap, 
             'legend_title': 'Elevation (m)'},
    'wind': {'cmap': paint.NWSWindSpeed.cmap, 
             'legend_title': 'Wind Speed (m/s)',
             'xarray_name': "si10"
            }
}

def map_var(ds, var_str, time_step=0, scale='110m', figsize=[15, 9], title=None):
    """
    Wrapper to generate EasyMap given xarray and variable string. Uses map_dict conventions. Should be robust to renaming certain vars
    """

    # Extract variable and time step
    # If input var_str not in data variables, try to get name from metadata dict
    if var_str not in ds:
        if var_str not in map_dict.keys():
            raise ValueError(f"var_str not recognized: {var_str}")
        else:
            x = ds[map_dict[var_str]["xarray_name"]]
    else:
        x = ds[var_str]
    x = x.isel(time=time_step)
    print(f"Mapping Variable {var_str} at time {ds.time[time_step].to_numpy()}")
    # Get mapping convention from map_dict
    if var_str in map_dict.keys():
        cmap = map_dict[var_str]["cmap"]
        legend_title = map_dict[var_str]["legend_title"]
    else:
        print(f"No mapping convention detected for input var_str: {var_str}.")
        cmap="viridis"
        legend_title=None

    ax = EasyMap("110m", figsize=figsize, crs=ds.herbie.crs).STATES().ax
    p = ax.pcolormesh(
        ds.longitude,
        ds.latitude,
        x,
        transform=pc,
        cmap=cmap,
    )
    
    cbar = plt.colorbar(
        p,
        ax=ax,
        orientation="horizontal",
        pad=0.01,
        shrink=0.8,
    )
    cbar.set_label(fontsize=14, label=legend_title)
    plt.title(title, size=18)




