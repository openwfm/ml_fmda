import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



df = pd.read_csv("../outputs/forecast_outputs/rnn_preds.csv")
d = df[(df.stid == "TT562") & (df.rep == 1)]
d = d.iloc[0:48]

# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
d.time = pd.to_datetime(d.date_time)

plt.plot(d.time, d.preds, color="k", linestyle="--", label="Predicted", alpha=.8)
plt.plot(d.time, d.fm, color="#468a29", label="Observed", alpha=.8)
plt.legend()
plt.grid()

inds = [0, 12, 24, 36, 47]
ts = [d.time.iloc[i] for i in inds]
plt.xticks(rotation=45, ha="right")
plt.xticks(ts)
plt.xticks(d.time, minor=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
plt.ylabel("Fuel Moisture (%)")

ind=1
x0 = d.time.iloc[ind]
y_pred = d.preds.iloc[ind]
y_obs = d.fm.iloc[ind]
residual = y_obs - y_pred

plt.errorbar(x=x0, y=y_pred + residual / 2, yerr=abs(residual) / 2,
             fmt='none', ecolor='red', elinewidth=2, capsize=5, capthick=2)
plt.text(x0 + pd.Timedelta(hours=1), y_pred + residual / 2, f"Error, t={ind + 1}", color='red', va='center', ha='left')

plt.tight_layout()
# plt.savefig("../outputs/pred_ex_t1.png")

# Gif
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation, PillowWriter

d.time = pd.to_datetime(d.date_time)

fig, ax = plt.subplots()
ax.plot(d.time, d.preds, color="k", linestyle="--", label="Predicted", alpha=.8)
ax.plot(d.time, d.fm, color="#468a29", label="Observed", alpha=.8)
ax.legend()
ax.grid()
inds = [0, 12, 24, 36, 47]
ts = [d.time.iloc[i] for i in inds]
ax.set_xticks(ts)
ax.tick_params(axis='x', rotation=45)
ax.set_xticks(d.time, minor=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
ax.set_ylabel("Fuel Moisture (%)")
plt.tight_layout()

err_line = ax.errorbar([], [], yerr=[], fmt='none', ecolor='red', elinewidth=2, capsize=5, capthick=2)
text = ax.text(0, 0, '', color='red', va='center', ha='left')

def update(ind):
    global err_line
    if err_line:
        err_line.remove()
    x0 = d.time.iloc[ind]
    y_pred = d.preds.iloc[ind]
    y_obs = d.fm.iloc[ind]
    residual = y_obs - y_pred
    
        
    new_err = ax.errorbar(x=x0, y=y_pred + residual / 2, yerr=abs(residual) / 2,
                          fmt='none', ecolor='red', elinewidth=2, capsize=5, capthick=2)
    err_line = new_err
    text.set_position((x0 + pd.Timedelta(hours=1), y_pred + residual / 2))
    text.set_text(f"Error, t={ind + 1}")
    return new_err, text

anim = FuncAnimation(fig, update, frames=range(48), interval=500, blit=False, repeat=True)

anim.save("../outputs/pred_ex.gif", writer=PillowWriter(fps=5))
plt.close()











