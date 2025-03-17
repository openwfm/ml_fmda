# Functions used for conventient plotting and mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Credit to Brian Blaylock for EasyMap within Herbie project

import numpy as np
from herbie import paint
# from herbie.toolbox import EasyMap, pc, ccrs
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Dictionary used to store timeseries plot schemes, ts for single location

plot_styles = {
    'fm': {'color': '#468a29', 'linestyle': '-', 'label': 'Observed FMC'},
    'Ed': {'color': '#EF847C', 'linestyle': '--', 'alpha':.8, 'label': 'drying EQ'},
    'Ew': {'color': '#7CCCEF', 'linestyle': '--', 'alpha':.8, 'label': 'wetting EQ'},
    'rain': {'color': 'b', 'linestyle': '-', 'alpha':.9, 'label': 'Rain'}
}

def plot_one(d, st, start_time="2024-01-01", end_time = "2024-01-07", title2 = "", save_path = None, show=True):
    """
    Plot univariate timeseries for formatted dictionary, one station key from output of build_ml_data
    """
    import pandas as pd

    if type(start_time) is str:
        start_time = pd.Timestamp(start_time, tz="UTC")
        end_time = pd.Timestamp(end_time, tz="UTC")

    title = f"Observed FMC at RAWS {st}"
    if title2:
        title = title + " - " + title2
    
    timestamps = d[st]["times"]
    inds = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    fm = d[st]["data"]["fm"].to_numpy()[inds]
    Ed = d[st]["data"]["Ed"].to_numpy()[inds]
    Ew = d[st]["data"]["Ew"].to_numpy()[inds]
    rain = d[st]["data"]["rain"].to_numpy()[inds]
    x = d[st]["times"][inds]
    plt.plot(x, fm, **plot_styles['fm'])
    plt.plot(x, Ed, **plot_styles['Ed'])
    plt.plot(x, Ew, **plot_styles['Ew'])
    plt.plot(x, rain, **plot_styles['rain'])
    plt.xlabel("Hour")
    plt.ylabel("FMC (%)")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save plot if path provided
    if save_path is not None:
        plt.savefig(save_path)

    # Show plot unless False
    if not show:
        plt.close()


        
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

    # Rename coords if so
    if not "latitude" in ds:
        ds = ds.rename({"lat": "latitude"})
        ds = ds.rename({"lon": "longitude"})
    
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

    

def create_gif(ds, var_str, tsteps, gif_path='output.gif', duration=0.5, legend_title = None, title=None):
    temp_dir = "./temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    if var_str not in ds:
        if var_str not in map_dict.keys():
            raise ValueError(f"var_str not recognized: {var_str}")
        else:
            x = ds[map_dict[var_str]["xarray_name"]]
    else:
        x = ds[var_str]    
    # Calculate global vmin and vmax across all frames
    vmin = x.min().item()
    vmax = x.max().item()
    
    frames = []
    for tstep in tsteps:
        t = ds.valid_time[tstep]
        formatted_time = f"{t.dt.year.item():04d}-{t.dt.month.item():02d}-{t.dt.day.item():02d} {t.dt.hour.item():02d}:{t.dt.minute.item():02d}:{t.dt.second.item():02d}"

        frame_path = os.path.join(temp_dir, f"frame_{tstep:03d}.png")
        map_var(ds, var_str, time_step=tstep, legend_title=legend_title,
                title=f"Forecast at {formatted_time}", save_path=frame_path, vmin=vmin, vmax=vmax)
        plt.close()
        plt.clf()
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")





def make_st_map_interactive(df, color=None):
    """
    Make interactive map with plotted bounding box. If color None, default scatter color. If color not none, it specifies a numeric column used for plotting color of scatter
    """
    marker_dict={
        'size': 10,
        'opacity': 0.7
    }
    
    if color is not None and color in df.columns:
        marker_dict["color"] = df[color]
        marker_dict["colorscale"] = "viridis"
        marker_dict["colorbar"] = dict(title=color)        
        marker_dict["colorbar"] = {
            "title": color,  
            "orientation": "h",  # Makes the colorbar horizontal
            "x": 0.5,  # Centers it horizontally
            "y": -0.15,  # Moves it below the map
            "xanchor": "center",  # Ensures centering
            "yanchor": "bottom"
        }    
        
    fig = go.Figure(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(**marker_dict),
        text=df['stid'],
        showlegend=False  # Turn off legend
    ))

    # Add Points
    center_lon=df['lon'].median()
    center_lat=df['lat'].median()
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center=dict(lat=center_lat, lon=center_lon)
    )
    # Add Lines for Bounding Box
    
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[df['lon'].min(), df['lon'].min(), df['lon'].max(), df['lon'].max(), df['lon'].min()],
        lat=[df['lat'].min(), df['lat'].max(), df['lat'].max(), df['lat'].min(), df['lat'].min()],
        marker=dict(size=5, color="black"),
        line=dict(width=1.5, color="black"),
        showlegend=False
    ))
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_zoom =5,
        mapbox_center={"lat": np.median(df.lat), "lon": np.median(df.lon)},  # Center the map on desired location
        width=800, height=600
    )
    return fig
