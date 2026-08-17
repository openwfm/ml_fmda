# Functions used for conventient plotting and mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Credit to Brian Blaylock for EasyMap within Herbie project

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Dictionary used to store timeseries plot schemes, ts for single location

plot_styles = {
    'fm': {'color': '#468a29', 'linestyle': '-', 'label': 'Observed FMC'},
    'fm_preds': {'color': '#468a29', 'linestyle': '-', 'label': 'Observed FMC'},
    'Ed': {'color': '#EF847C', 'linestyle': '--', 'alpha':.8, 'label': 'Drying EQ'},
    'Ew': {'color': '#7CCCEF', 'linestyle': '--', 'alpha':.8, 'label': 'Wetting EQ'},
    'rain': {'color': 'b', 'linestyle': '-', 'alpha':.9, 'label': 'Rain'},
    'model': {'color': 'k', 'linestyle': '-', 'label': 'Predicted FMC'}
}

import numpy as np
import matplotlib as mpl

dfm_bounds = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

dfm_colors = np.array([
    (156, 22, 27), (188, 28, 32), (217, 45, 43),
    (234, 84, 43), (245, 137, 56), (249, 201, 80),
    (215, 225, 95), (203, 217, 88), (114, 190, 75),
    (74, 167, 113), (60, 150, 120)
]) / 255.0

dfm_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "dfm_red_to_green", dfm_colors, N=len(dfm_colors)
)

dfm_norm = mpl.colors.BoundaryNorm(
    boundaries=dfm_bounds + [1e10],
    ncolors=len(dfm_bounds)
)

def plot_one(d, st, features=True, m=None, start_time="2024-01-01", end_time = "2024-01-07", title2 = "", save_path = None, show=True):
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
    x = d[st]["times"][inds]
    plt.plot(x, fm, **plot_styles['fm'])
    if features:
        Ed = d[st]["data"]["Ed"].to_numpy()[inds]
        Ew = d[st]["data"]["Ew"].to_numpy()[inds]
        rain = d[st]["data"]["rain"].to_numpy()[inds]
        plt.plot(x, Ed, **plot_styles['Ed'])
        plt.plot(x, Ew, **plot_styles['Ew'])
        plt.plot(x, rain, **plot_styles['rain'])
        if m is not None:
            plt.plot(x, m, **plot_styles['model'])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Hour")
    plt.ylabel("FMC (%)")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()

    # Save plot if path provided
    if save_path is not None:
        plt.savefig(save_path)

    # Show plot unless False
    if not show:
        plt.close()


def map_var(ds, var_str, time_step=0, scale="110m", figsize=(15, 9),
            legend_title=None, title=None, save_path=None,
            vmin=None, vmax=None, cmap="viridis", land_mask=None):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Rename coords if needed
    if "latitude" not in ds:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    x = ds.isel(time=time_step)
    t = x.time.values
    time_str = np.datetime_as_string(t, unit="m")

    # NOTE documentation for HRRR LSM is wrong, 
    # 1 is land 0 is sea. Not sure if herbie or NOAA issue
    if land_mask is not None:
        x = x[var_str].where(x[land_mask] == 1)
    else:
        x = x[var_str]
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.LambertConformal())

    ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.OCEAN.with_scale(scale))

    p = ax.pcolormesh(
        ds.longitude,
        ds.latitude,
        x,
        transform=ccrs.PlateCarree(),
        shading="auto",
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
    cbar.set_label(legend_title, fontsize=14)

    if title is None:
        ax.set_title(f"{var_str} ({time_str} UTC)", fontsize=18)
    else:
        ax.set_title(f"{title} ({time_str} UTC)", fontsize=18)       

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig, ax

    

def create_gif(ds, var_str, tsteps, gif_path='output.gif', duration=0.5, legend_title = None, title=None, cmap="viridis", vmin=None, vmax=None):
    import imageio.v2 as imageio
    
    temp_dir = "./temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    x = ds[var_str]    
    # Calculate global vmin and vmax across all frames
    if vmin is None:
        vmin = x.min().item()
    if vmax is None:
        vmax = x.max().item()
    
    frames = []
    for tstep in tsteps:
        t = ds.valid_time[tstep]
        formatted_time = f"{t.dt.year.item():04d}-{t.dt.month.item():02d}-{t.dt.day.item():02d} {t.dt.hour.item():02d}:{t.dt.minute.item():02d}:{t.dt.second.item():02d}"

        frame_path = os.path.join(temp_dir, f"frame_{tstep:03d}.png")
        map_var(ds, var_str, time_step=tstep, legend_title=legend_title,
                title=f"Forecast at {formatted_time}", save_path=frame_path, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.close()
        plt.clf()
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")





def make_st_map_interactive(df, color=None, binary=False):
    """
    Make interactive map with plotted bounding box. If color None, default scatter color. If color not none, it specifies a numeric column used for plotting color of scatter
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    marker_dict={
        'size': 6,
        'opacity': 0.7
    }
    
    if color is not None and color in df.columns:
        if not binary:
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
        else:
            binary_colors = {1: "blue", 0: "red"}
            df["color_mapped"] = df[color].map(binary_colors)
            
            marker_dict["color"] = df["color_mapped"]
            marker_dict["colorbar"] = None  # Optional: hide colorbar for discrete classes
            marker_dict["colorscale"] = None  # Not needed for discrete colors
            marker_dict["showscale"] = False  # Prevents colorbar from appearing        
        
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
        width=1000, height=600
    )
    return fig
