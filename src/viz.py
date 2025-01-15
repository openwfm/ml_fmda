# Functions used for conventient plotting and mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Credit to Brian Blaylock for EasyMap within Herbie project

from herbie import paint
from herbie.toolbox import EasyMap, pc, ccrs
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt





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

def map_var(ds, var_str, time_step=0, scale='110m', figsize=[15, 9], legend_title=None, title=None, save_path=None, vmin=None, vmax=None):
    """
    Wrapper to generate EasyMap given xarray and variable string. Uses map_dict conventions. Should be robust to renaming certain vars.
    """
    if var_str not in ds:
        if var_str not in map_dict.keys():
            raise ValueError(f"var_str not recognized: {var_str}")
        else:
            x = ds[map_dict[var_str]["xarray_name"]]
    else:
        x = ds[var_str]
    x = x.isel(time=time_step)

    if var_str in map_dict.keys():
        cmap = map_dict[var_str]["cmap"]
        if legend_title is None:
            legend_title = map_dict[var_str]["legend_title"]
    else:
        cmap = "viridis"
        legend_title = legend_title

    ax = EasyMap("110m", figsize=figsize, crs=ds.herbie.crs).STATES().OCEAN().COASTLINES().LAKES().ax
    
    # Add vmin and vmax to fix the colorbar range
    p = ax.pcolormesh(
        ds.longitude,
        ds.latitude,
        x,
        transform=pc,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    

def create_gif(ds, var_str, tsteps, gif_path='output.gif', duration=0.5):
    temp_dir = "./temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Calculate global vmin and vmax across all frames
    vmin = ds[var_str].min().item()
    vmax = ds[var_str].max().item()
    
    frames = []
    for tstep in tsteps:
        t = ds.valid_time[tstep]
        formatted_time = f"{t.dt.year.item():04d}-{t.dt.month.item():02d}-{t.dt.day.item():02d} {t.dt.hour.item():02d}:{t.dt.minute.item():02d}:{t.dt.second.item():02d}"

        frame_path = os.path.join(temp_dir, f"frame_{tstep:03d}.png")
        map_var(ds, var_str, time_step=tstep, legend_title="Fuel Moisture Content (%)",
                title=f"FMC Forecast at {formatted_time}", save_path=frame_path, vmin=vmin, vmax=vmax)
        plt.close()
        plt.clf()
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")


