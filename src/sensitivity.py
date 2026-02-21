# Script to generate tables for sensitivity analysis from predictors
# Analysis run with different sets of predictors
# We will compare overall metrics for RNN and XGBoost (included so conclusions aren't limited to one model architecture)
# Just focusing on RMSE point estimate from the configurations
# User input to script: space separated file directores, subdirectories of forecasts dir. Expect there to be a struture: forecasts/SUBDIR_INPUT/error_analysis/overall.csv
# Need to provide a full model configuration directory, then at least one sub config
# Two types of comparisons: groups of predictors (i.e. weather, geographic, temporal), and then specific (rain, wind, etc). NOTE: for specific predictors, still combining fundamentally linked ones including (Ed, Ew) and (lat/lon)


# Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import sys
import os
import os.path as osp

from src.utils import read_yml


# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        "\\begin{table}[tbh]\n"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        + latex_str +
        "\n\\end{table}"
    )
    
    return latex_str

# Lookup table for parameter configurations, description and grouping
lookup = {
    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'elev', 'lon', 'lat', 'rain', 'hod', 'doy'])): {
        "type": "full", 
        "name": "Full Model",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'rain', 'hod', 'doy'])): {
        "type": "groups",
        "name": "Weather " + r"$+$" " Temporal",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'elev', 'lon', 'lat', 'rain'])): {    
        "type": "groups",
        "name": "Weather " + r"$+$" " Geographic",        
    },

    tuple(sorted(['elev', 'lon', 'lat', 'hod', 'doy'])): {
        "type": "groups",
        "name": "No Weather",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'rain'])): {
        "type": "groups",
        "name": "Weather Only",
    },

    tuple(sorted(['solar', 'wind', 'elev', 'lon', 'lat', 'rain', 'hod', 'doy'])): {
        "type": "individual",
        "name": "No Equilibria",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'elev', 'lon', 'lat', 'hod', 'doy'])): {
        "type": "individual",
        "name": "No Rain",
    },

    tuple(sorted(['Ed', 'Ew', 'wind', 'elev', 'lon', 'lat', 'rain', 'hod', 'doy'])): {
        "type": "individual",
        "name": "No Solar",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'elev', 'lon', 'lat', 'rain', 'hod', 'doy'])): {
        "type": "individual",
        "name": "No Wind",
    },

    tuple(sorted(['Ed', 'Ew', 'solar', 'wind', 'elev', 'rain', 'hod', 'doy'])): {
        "type": "individual",
        "name": "No Lat/Lon",
    },

}

# Only cols to be compared, we need std from the baseline for the table so just getting from all then filtering later
num_cols = ['rmse_mean',  'rmse_std']
analysis_col = ["rmse_mean"]
std_col = ["rmse_std"]
analysis_col_rename = ["RMSE Mean"]


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Error: Need at least an output file and one directory.")
        print(('Usage: %s <output_txt_file> <dir1> <dir2> <dir3> ...' % sys.argv[0]))
        print("Example: python src/sensitivity.py outputs/sensitivity.txt forecsts/subdir1 forecasts/subdir2")
        sys.exit(1)

    
    output_file = sys.argv[1]
    directories = sys.argv[2:]

    print(f"Forecast Directories: {directories}")
    print(f"Output File: {output_file}")
    
    # Check for needed files
    assert len(directories) > 1, f"Only 1 directory provided, tables won't make sense. Need to give full baseline directory and then a subdirectory that is a subset"
    assert np.all([osp.exists(dir_i) for dir_i in directories]), f"Not all provided directories exist: {[osp.exists(dir_i) for dir_i in directories]}"    
    overall_err_table_path = [osp.join(dir_i, "error_analysis", "overall.csv") for dir_i in directories]
    assert np.all([osp.exists(path_i) for path_i in overall_err_table_path]), f"Not all directories have 'error_analysis/overall.csv': {[osp.exists(path_i) for path_i in overall_err_table_path]}"

    # Read Tables, extract only XGB and RNN, extract features lists to partition
    overall_err_tables = [pd.read_csv(path_i) for path_i in overall_err_table_path]
    overall_err_tables = [df[df["Model"].isin(["rnn", "xgb"])][["Model"] +  num_cols] for df in overall_err_tables]
    feature_lists = [read_yml(osp.join(dir_i, "forecast_config.yaml"), subkey="features_list") for dir_i in directories]
    
    config = [lookup[tuple(sorted(fi))] for fi in feature_lists] # get grouping info for features list
    assert any(c["type"] == "full" for c in config), "'full' configuration not found. Tables won't make sense without full baseline" 
    
    # Divide into groups vs individual comparisons
    # List of configs and error tables for the 2 sets
    idx_full       = [i for i, c in enumerate(config) if c["type"] == "full"]
    idx_group      = [i for i, c in enumerate(config) if c["type"] == "groups"]
    idx_individual = [i for i, c in enumerate(config) if c["type"] == "individual"]
    
    # Full Model
    config_full =  [config[i] for i in idx_full][0]
    err_full    =  [overall_err_tables[i] for i in idx_full][0]
    err_full.insert(0, "Configuration", "Full Model")

    # Predictor Group Removal
    config_group =  [config[i] for i in idx_group]
    err_group    =  [overall_err_tables[i] for i in idx_group]
    for cfg, df in zip(config_group, err_group):
        df.insert(0, "Configuration", cfg["name"])

    
    # Predictor Individual Removal
    config_individual = [config[i] for i in idx_individual]
    err_individual    = [overall_err_tables[i] for i in idx_individual]
    for cfg, df in zip(config_individual, err_individual):
        df.insert(0, "Configuration", cfg["name"])
    
    
    # Generate Tables
    def make_table(full, comparison, model="rnn", num_dec = 2):
        full = full[full.Model == model]
        full_std = full[std_col].round(num_dec)
        full = full[['Configuration']+analysis_col]
        full[analysis_col] = full[analysis_col].round(num_dec)

        comparison = [df[df.Model == model][['Configuration']+analysis_col] for df in comparison]
        comparison = pd.concat(comparison).sort_values(analysis_col)
        comparison[analysis_col] = comparison[analysis_col].round(num_dec)
        comparison["Difference"] = comparison[analysis_col] - full[analysis_col]       # Raw diff
        comparison["Relative Difference"] = (comparison[analysis_col]-full[analysis_col])/full[analysis_col]  # Relative diff, full as baseline
        comparison["Difference"] = comparison["Difference"].round(num_dec)
        comparison["Relative Difference"] = (comparison["Relative Difference"] * 100).round(num_dec).astype(str) + r"$\%$" 

        # Format Full comparison with +/- bounds
        for col in set(full.columns).union(comparison.columns):
            full[col] = full.get(col, "")
            comparison[col] = comparison.get(col, "")
        combined = pd.concat([full, comparison], ignore_index=True)

        # Make columns pretty
        combined[analysis_col] = combined[analysis_col].astype(str)
        combined.loc[combined.Configuration == "Full Model", "rmse_mean"] = str(full[analysis_col].values[0][0]) + r' $\pm$ ' + str(full_std.values[0][0])
        combined = combined.rename(columns={analysis_col[0]: analysis_col_rename[0]})
        return combined


    tableg = None
    tablei = None
    if len(err_group)>=1:
        print(f"Evaluating grouped predictor sensitivity")
        tableg = make_table(err_full, err_group)
        tableg = table_to_latex(tableg, caption="Sensitivity analysis for predictor groups.", label="sens1")


    if len(err_individual)>=1:
        print(f"Evaluating specific predictor sensitivity")
        tablei = make_table(err_full, err_individual)
        tablei = table_to_latex(tablei, caption="Sensitivity analysis for specific predictors", label="sens2")

    print(f"Writing tex formatted tables to {output_file}")
    with open(output_file, "w") as f:
        f.write("Sensitivity Analysis:\n\n")
        if tablei: f.write(tablei + "\n\n")
        if tableg: f.write(tableg + "\n\n")



