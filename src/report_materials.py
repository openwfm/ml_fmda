# Script to generate tables and figures associated with a paper
# Outputs combined and analyzed in src/forecast_eval.py, 
# This script formats tables and plots to be copied into a latex document


import numpy as np
import sys
import os
import os.path as osp
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import re
import matplotlib.pyplot as plt

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, str2time, time_range


# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Used to turn std column into a +/- for style
def format_table(table1):
    table1["Bias"] = table1["bias_mean"].astype('float64').round(2).astype(str) + r" $\pm$ " + table1["bias_std"].round(2).astype(str)
    table1["RMSE"] = table1["rmse_mean"].astype('float64').round(2).astype(str) + r" $\pm$ " + table1["rmse_std"].round(2).astype(str)
    table_formatted = table1[["Model", "Bias", "RMSE"]]
    return table_formatted 


def table_to_latex(df, caption="CAPTION TEXT.", label="tab1"):
    """
    Modify pandas to_latex to match table formatting in given journal. 
    This implementation is specific to tex template from MDPI Fire.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert to LaTeX.
    caption (str): The caption text for the table.
    label (str): The LaTeX label for referencing the table.

    Returns:
    str: A LaTeX-formatted table string 
    """
    import re

    # Bold column names
    bold_df = df.copy()
    bold_df.columns = [f"\\textbf{{{col}}}" for col in df.columns]
    
    # Generate basic LaTeX table with booktabs for top/mid/bottom rule
    latex_str = bold_df.to_latex(index=False, escape=False, float_format="%.2f")
    
    # Extract column count from the tabular declaration
    col_count = len(df.columns)
    col_spec = 'C' * col_count

    # Replace tabular environment with tabularx and correct column spec
    latex_str = re.sub(
        r"\\begin{tabular}{.*?}",
        rf"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        latex_str
    )
    latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabularx}")

    # Wrap in table environment with caption and label
    latex_str = (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        + latex_str +
        "\n\\end{table}"
    )
    
    return latex_str

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/report_materials.py forecasts/fmc_forecast_test/")
        sys.exit(-1)

    f_dir = sys.argv[1]
    fconf = Dict(read_yml(osp.join(f_dir, "forecast_config.yaml")))
    params_models = Dict(read_yml(osp.join(f_dir, "params_models.yaml")))
    out_dir = osp.join(f_dir, "report_materials")
    os.makedirs(out_dir, exist_ok=True)


    # Read tables from output directory
    print(f"Reading outputs from: {osp.join(f_dir, 'error_analysis')}")
    overall = pd.read_csv(osp.join(f_dir, "error_analysis", "overall.csv"))
    by_dt = pd.read_csv(osp.join(f_dir, "error_analysis", "by_dt.csv"))
    by_hod = pd.read_csv(osp.join(f_dir, "error_analysis", "by_hod.csv"))
    by_st = pd.read_csv(osp.join(f_dir, "error_analysis", "by_stid.csv"))
    sts = pd.read_csv(osp.join(f_dir, "error_analysis", "stid_locs.csv"))
    by_st = by_st.merge(sts, on="stid", how="left")
    rnn = pd.read_csv(osp.join(f_dir, "error_analysis", "rnn_preds.csv"))
    all_vars = pd.read_csv(osp.join(f_dir, "error_analysis", "all_variables_summary.csv"))

    # Create Tables
    print(f"Creating latex format tables, saving to {out_dir}")
    ## Overall error table
    print("    Formatting overall forecast error table")
    table_overall_error = table_to_latex(format_table(overall), caption="Overall Forecast Error", label="tab:overall")

    ## Predictor Variable Summary
    print("    Formatting predictor summary table")
    all_vars.Variable = ["Drying Equilibrium", "Wetting Equilibrium", "Solar Radiation", "Wind Speed", "Elevation", "Longitude", "Latitude", "Rain", "Hour of Day", "Day of Year"]
    all_vars["Units"] = [r"$\%$", r"$\%$", r"$W/m^2$", r"$m/s$", "meters", "degree", "degree",r"mm/h", r"hours", "days"]
    all_vars["Description"] = ["Derived from RH and temperature.", "Derived from RH and temperature.", "Downward shortwave radiative flux.", "Wind speed at 10m.", "Height above sea level.", "X-Coordinate", "Y-Coordinate", "Calculated from rain accumulated over the hour.", "From 0 to 23, UTC time.", "From 0 to 365 or 366 (leap year in 2024)"]
    all_vars = all_vars[["Variable", "Units", "Description", "Mean", "Low", "High"]]
    all_vars = table_to_latex(all_vars, caption="Predictors Summary", label="tab:all_vars")

    ## Model Hyperparam Tables
    ## RNN, not including model architecutre params, those will be presented separately
    print("    Formatting RNN hyperparam table")
    params = params_models.rnn
    keep_params = ['learning_rate', 'timesteps', 'batch_size', 'dropout', 'early_stopping_patience', 'scaler']
    renames = ["Learning Rate", "Timesteps", "Batch Size", "Dropout", "Scaling", "Early Stopping Patience"]
    description = ["Controls how fast weights update during training", 
                   "Number of timesteps for each input", 
                   "Number of sequences to process before updating weights",
                   "Rate that weights are randomly to set during training",
                   "Number of epochs with no improvement to validation error before stopping training",
                   "Scaling inputs to mean zero and std. 1"
                  ]
    rnn_tab = pd.DataFrame({
        'Hyperparameter': list(params.keys()),
        'Value': [str(v) if isinstance(v, list) else v for v in params.values()]
    })
    rnn_tab = rnn_tab.set_index('Hyperparameter').loc[keep_params].reset_index()
    rnn_tab.Hyperparameter = renames
    rnn_tab["Description"]=description
    rnn_tab = table_to_latex(rnn_tab, caption="RNN Hyperparameters", label="tab:rnn_params")


    ## ODE Model
    print("    Formatting ODE hyperparam table")
    params = params_models.ode
    keep_params = ['spinup_hours', 'process_variance', 'data_variance', 'r0', 'rs', 'Tr', 'S', 'T']
    renames = ["Spinup", "Process Variance", "Data Variance", "$r_0$", "$r_s$", "$T_r$", "$S$", "$T$"]
    description = ["Number of hours to run model with data assimilation", "Uncertainty in model dynamics",
                  r"Uncertainty in measurement of input data", "Threshold rainfall intensity ($mm h^{-1}$)",
                  r"Saturation rainfall intensity ($mm h^{-1}$)",
                  "Characteristic decay time for wetting dynamics ($h$)",
                  "Saturation FMC level",
                  "Characteristic decay time for fuel class"]
    ode_tab = pd.DataFrame({
        'Hyperparameter': list(params.keys()),
        'Value': [str(v) if isinstance(v, list) else v for v in params.values()]
    })
    ode_tab = ode_tab.set_index('Hyperparameter').loc[keep_params].reset_index()
    ode_tab.Hyperparameter = renames
    ode_tab["Description"]=description
    ode_tab = table_to_latex(ode_tab, caption="ODE Hyperparameters", label="tab:ode_params")   

    ## XGB Model
    print("    Formatting XGB hyperparam table")
    params = params_models.xgb
    keep_params = ['n_estimators', 'max_depth', 'eta', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree']
    renames = ["N Estimators", "Max Tree Depth", "Learning Rate", "Min Child Weight", r"$\gamma", "Subsample",
              "Colsample by Tree"]
    description = ["Number of trees in ensemble", "Maximum tree depth.", 
                   "Controls how fast weights update during training",
                   "Minimum sum of Hessian of samples in a partition",
                   "Minimum loss reduction required for a tree partition",
                   "Percent of training data randomly selected in each boosting iteration",
                   "Percent of predictors randomly selected in each boosting iteration"
                  ]
    xgb_tab = pd.DataFrame({
        'Hyperparameter': list(params.keys()),
        'Value': [str(v) if isinstance(v, list) else v for v in params.values()]
    })
    xgb_tab = xgb_tab.set_index('Hyperparameter').loc[keep_params].reset_index()
    xgb_tab.Hyperparameter = renames
    xgb_tab["Description"]=description
    xgb_tab = table_to_latex(xgb_tab, caption="XGB Hyperparameters", label="tab:xgb_params")


    ## Climatology Method
    print("    Formatting climatology hyperparam table")
    params = params_models.climatology
    keep_params = ['nyears', 'ndays', 'min_years']
    renames = ["Years", "Days", "Min Years"]
    description = ["Number of years to look for historical FMC at a given location",
                  "Number of days plus or minus the target time to aggregate historical FMC data",
                  "Minimum unique number of years of observed FMC to generate a prediction"
                  ]
    clim_tab = pd.DataFrame({
        'Hyperparameter': list(params.keys()),
        'Value': [str(v) if isinstance(v, list) else v for v in params.values()]
    })
    clim_tab = clim_tab.set_index('Hyperparameter').loc[keep_params].reset_index()
    clim_tab.Hyperparameter = renames
    clim_tab["Description"]=description
    clim_tab = table_to_latex(clim_tab, caption="Climatology Hyperparameters", label="tab:clim_params")
    
    
    # Write tables into a tex document, used to copy into a tex project
    print(f"Writing tex formatted tables to {osp.join(out_dir, 'tables_output.tex')}")
    with open(osp.join(out_dir, 'tables_output.tex'), "w") as f:
        f.write("Error Summary:\n\n")
        f.write(table_overall_error + "\n\n")
        f.write("Tables from the model comparison:\n\n")
        f.write(rnn_tab + "\n\n")  # Adds spacing between tables
        f.write(ode_tab + "\n\n")
        f.write(xgb_tab + "\n\n")
        f.write(clim_tab + "\n\n")
        f.write(all_vars + "\n\n")



    # Plots
    print("    Creating plot of error over 24 hours")
    ## Average error over 24 hours
    x = np.arange(24)
    models = by_hod.Model.unique()

    rename = {
        "rnn": "RNN",
        "ode": "ODE+KF",
        "xgb": "XGBoost",
        "clim": "Climatology"
    }

    plt.figure()
    for model in models:
        y = by_hod[by_hod.Model == model].rmse_mean.to_numpy()
        label = rename.get(model, model)
        plt.plot(x, y, label=label)

    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('RMSE')
    plt.xticks(np.arange(0, 24, 2))
    plt.ylim(0, 7)
    plt.legend(loc='upper left')
    print(f"Writing plot of error averaged over hour of day to: {osp.join(out_dir, 'err24.png')}")
    plt.savefig(osp.join(out_dir, "err24.png"), dpi=300)
    
    ## RNN FM vs Residual Plot
    print("    Creating residual histogram plot")
    plt.figure()
    plt.scatter(rnn.fm, rnn.residual, marker="o", alpha=.7); plt.grid(); plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("FMC (%)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.savefig(osp.join(out_dir, "rnn_residuals.png"), dpi=300)





