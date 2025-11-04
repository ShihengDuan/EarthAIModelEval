import glob
import json
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pcmdi_metrics.graphics import download_archived_results
from pcmdi_metrics.utils import sort_human
from pcmdi_metrics.graphics import normalize_by_median
from pcmdi_metrics.graphics import portrait_plot
import json, copy

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stat', type=str, choices=['rms', 'rmsc', 'stdv_pc_ratio_to_obs'])
    args = vars(parser.parse_args())
    return args

args = get_args()

modes = ['NAM', 'NAO', 'PNA', 'SAM']
# modes = ['NAM', 'NAO', 'PNA']
json_dir = './json_files'

mip = "cmip6"
exp = "amip"
data_version = "v20210119"


stat = args['stat']
#stat = "rmsc"
#stat = "stdv_pc_ratio_to_obs"
if stat == "rms":
    stat_name = "RMSE"
elif stat == "rmsc":
    stat_name = "Centered RMSE"
elif stat == "stdv_pc_ratio_to_obs":
    stat_name = "Amplitude ratio to OBS"

for mode in modes:
    if mode in ['PDO', 'NPGO']:
        ref = "HadISSTv1.1"
    else:
        ref = "NOAA-CIRES_20CR"

    if mode in ['NPO', 'NPGO']:
        eof = "EOF2"
    else:
        eof = "EOF1"

    path = os.path.join("metrics_results/variability_modes/"+mip+"/"+exp+"/"+data_version,
                        mode, ref,
                        "_".join(["var", "mode", mode, eof, "stat", mip, exp, "mo_atm_allModels_allRuns_1900-2005.json"]))
    download_archived_results(path, json_dir)
json_list = sorted(glob.glob(os.path.join(json_dir, 'var_mode_*' + mip + '*' + '.json')))
for json_file in json_list:
    print(json_file.split('/')[-1])

cmip_result_dict = dict()
modes = list()

for json_file in json_list:
    mode = json_file.split('/')[-1].split('_')[2]
    modes.append(mode)
    with open(json_file) as fj:
        dict_temp = json.load(fj)['RESULTS']
        cmip_result_dict[mode] = dict_temp
def dict_to_df(cmip_result_dict):
    models = sorted(list(cmip_result_dict['NAM'].keys()))

    df = pd.DataFrame()
    df['model'] = models
    df['num_runs'] = np.nan

    mode_season_list = list()

    modes = ['SAM', 'NAM', 'NAO', 'PNA', ]
    # modes = ['NAM', 'NAO', 'PNA', ]
    for mode in modes:
        if mode in ['PDO', 'NPGO']:
            seasons = ['monthly']
        else:
            seasons = ['DJF', 'MAM', 'JJA', 'SON']

        for season in seasons:
            df[mode+"_"+season] = np.nan
            mode_season_list.append(mode+"_"+season)
            for index, model in enumerate(models):
                if model in list(cmip_result_dict[mode].keys()):
                    runs = sort_human(list(cmip_result_dict[mode][model].keys()))
                    stat_run_list = list()
                    for run in runs:
                        stat_run = cmip_result_dict[mode][model][run]['defaultReference'][mode][season]['cbf'][stat]
                        stat_run_list.append(stat_run)
                    stat_model = np.average(np.array(stat_run_list))
                    num_runs = len(runs)
                    df.at[index, mode+"_"+season] = stat_model
                    if np.isnan(df.at[index, 'num_runs']):
                        df.at[index, 'num_runs'] = num_runs
                else:
                    stat_model = np.nan
                    num_runs = 0
    return df, mode_season_list
df, mode_season_list = dict_to_df(cmip_result_dict)
df_combined = df
model_labels = [m + ' (' + str(int(r)) + ')' for m, r in zip(df_combined["model"].to_list(), df_combined["num_runs"].to_list())]
landscape = True
#landscape = False



def _merge_one_mov_json(cmip_result_dict, j, reference_key="defaultReference", methods=("cbf",)):
    merged = copy.deepcopy(cmip_result_dict)
    results = j.get("RESULTS", {})

    for model_name, runs_dict in results.items():             # e.g. "ACE2-PCMDI"
        for run_name, ref_block in runs_dict.items():         # e.g. "r1i1p1f1"
            if reference_key not in ref_block:
                continue
            ref_dict = ref_block[reference_key]               # => {MODE: {SEASON: {...}}, "period":...}

            # 模态键：排除元数据
            modes_in_json = [k for k in ref_dict.keys() if k not in ("period","reference_eofs","target_model_eofs","source")]
            for mode in modes_in_json:                        # e.g. "NAM"/"NAO"/"PNA"/"SAM"/"PDO"/"NPGO"
                seasons = [s for s in ref_dict[mode].keys() if s not in ("period","target_model_eofs")]

                merged.setdefault(mode, {})
                merged[mode].setdefault(model_name, {})
                merged[mode][model_name].setdefault(run_name, {})
                merged[mode][model_name][run_name].setdefault(reference_key, {})
                merged[mode][model_name][run_name][reference_key].setdefault(mode, {})

                for season in seasons:                        # e.g. "DJF" / "MAM" / "JJA" / "SON" / "monthly"
                    tgt = merged[mode][model_name][run_name][reference_key][mode].setdefault(season, {})
                    src = ref_dict[mode][season]
                    # 只放入指定的方法（cbf / eof1）
                    for mth in methods:
                        if mth in src:
                            tgt[mth] = src[mth]
                    # 可选保留 period 元数据
                    if "period" in src:
                        tgt["period"] = src["period"]
    return merged

def merge_my_mov_dir_into_cmip(cmip_result_dict,
                               my_dir_pattern,           # e.g. "/pscratch/.../ACE-MOV/*.json"
                               reference_key="defaultReference",
                               methods=("cbf",)):
    merged = copy.deepcopy(cmip_result_dict)
    for path in sorted(glob.glob(my_dir_pattern)):
        try:
            with open(path, "r") as f:
                j = json.load(f)   # Python 的 json 接受 NaN；若报错可先将 "NaN" 替换为 null 再加载
        except Exception as e:
            print(f"[WARN] 跳过 {os.path.basename(path)}: {e}")
            continue
        merged = _merge_one_mov_json(merged, j, reference_key=reference_key, methods=methods)
    return merged

cmip_plus_mine = merge_my_mov_dir_into_cmip(
    cmip_result_dict,
    my_dir_pattern="/pscratch/sd/d/duan0000/PMP/demo_output/ACE-MOV/*.json",
    reference_key="defaultReference",
    methods=("cbf",)  
)
cmip_plus_mine = merge_my_mov_dir_into_cmip(
    cmip_plus_mine,
    my_dir_pattern="/pscratch/sd/d/duan0000/PMP/demo_output/NGCM-MOV/*.json",
    reference_key="defaultReference",
    methods=("cbf",)  
)

df_all, mode_season_list = dict_to_df(cmip_plus_mine)

df_new = df_all.copy()

move_last = ["ACE2", "NeuralGCM"]

df_move = df_new[df_new["model"].isin(move_last)]
df_rest = df_new[~df_new["model"].isin(move_last)]

df_reordered = pd.concat([df_rest, df_move], ignore_index=True)


model_labels = [m + ' (' + str(int(r)) + ')' for m, r in zip(df_reordered["model"].to_list(), df_reordered["num_runs"].to_list())]
data = dict()

if landscape:
    data = df_reordered[mode_season_list].to_numpy().T
else:
    data = df_reordered[mode_season_list].to_numpy()

models = df_reordered.index.values.tolist()

print('data.shape:', data.shape)
print('len(mode_season_list): ', len(mode_season_list))
print('len(models): ', len(models))

if landscape:
    yaxis_labels = mode_season_list
    xaxis_labels = model_labels
else:
    xaxis_labels = mode_season_list
    yaxis_labels = model_labels
if landscape:
    axis = 1
    figsize = (40, 10)
else:
    axis = 0
    figsize = (18, 25)

if stat not in ["stdv_pc_ratio_to_obs"]:
    data_nor = normalize_by_median(data, axis=axis)
    cmap_bounds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    vertical_center = "median"
    cmap = 'RdYlBu_r'
else:
    data_nor = data
    cmap_bounds = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    cmap_bounds = [r/10 for r in range(5, 16, 1)]
    vertical_center = 1
    cmap = 'jet'

fig, ax, cbar = portrait_plot(data_nor,
                              xaxis_labels=xaxis_labels,
                              yaxis_labels=yaxis_labels,
                              cbar_label=stat_name,
                              box_as_square=True,
                              vrange=(-0.5, 0.5),
                              figsize=figsize,
                              cmap=cmap,
                              cmap_bounds=cmap_bounds,
                              cbar_kw={"extend": "both"},
                              missing_color='white',
                              legend_box_xy=(1.11, 1.21),
                              legend_box_size=4,
                              legend_lw=1,
                              legend_fontsize=15,
                              logo_rect = [0.67, 1, 0.15, 0.15],
                             )
ax.set_xticklabels(xaxis_labels, rotation=90, va='center', ha="left")

# Add title
ax.set_title("Variability Modes: "+stat_name, fontsize=30, pad=30)
for i in range(0, len(models)-2):
    plt.setp(ax.get_xticklabels()[i], color='blue')
for i in range(len(models)-2, len(models)):
    plt.setp(ax.get_xticklabels()[i], color='red')
# plt.setp(ax.get_xticklabels()[num_cmip5 + num_cmip6], color='blue')
# plt.setp(ax.get_xticklabels()[num_cmip5 + num_cmip6 + 1], color='red')

# add partition lines between model groups
ax.axvline(x=len(models)-2, color='k', linewidth=3)
ax.axvline(x=len(models), color='k', linewidth=3)
plt.savefig(f'Figs/potrait_MoV_{stat_name}.png', bbox_inches='tight', dpi=200)

# Parallel
data = df_all[mode_season_list].to_numpy()
model_names = model_labels
metric_names = mode_season_list
model_highlights = None
print('data.shape:', data.shape)
print('len(metric_names): ', len(metric_names))
print('len(model_names): ', len(model_names))

from pcmdi_metrics.graphics import parallel_coordinate_plot
model_highlights = ['ACE2 (10)', 'NeuralGCM (10)']

fig, ax = parallel_coordinate_plot(data, metric_names, model_names,
                                   models_to_highlight=model_highlights,
                                   models_to_highlight_colors=['blue', 'red'],
                                   title='Variability Modes: '+stat_name,
                                   figsize=(21, 7),
                                   colormap='tab20',
                                   show_boxplot=False,
                                   show_violin=True,
                                   violin_colors=("lightgrey", "pink"),
                                   xtick_labelsize=10,
                                   logo_rect=[0.8, 0.8, 0.15, 0.15],
                                   comparing_models=model_highlights,
                                   vertical_center=vertical_center,
                                   vertical_center_line=True
                                  )

ax.set_xticklabels(metric_names, rotation=30, va='top', ha="right")
plt.savefig(f'Figs/parallel_MoV_{stat_name}.png', bbox_inches='tight', dpi=200)

