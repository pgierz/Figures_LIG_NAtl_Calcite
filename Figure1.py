#!/usr/bin/env python

from simulation_tools import cosmos_wiso_analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from basemap_wrappers import map_natl2
from plot_tools import plot_var_anom_hatching_from_ncdf_file
#from itertools import product
import matplotlib.colors
import cdo
from custom_io import get_remote_data
from scipy.io import netcdf
#import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from math import sqrt
import publication_settings

CDO = cdo.Cdo(cdfMod="scipy", env={"PATH":"/anaconda3/envs/default/bin"})

# Proxy Data
proxy_data = pd.read_excel("/Users/pgierz/Research/data/LIG_d18Osw_130ka_PG_none_missing.xlsx")

# Simulations
user = "pgierz"
host = "stan1"
prepath = "/ace/user/pgierz/"
sim_dict = {}
sim_dict["PI"] = cosmos_wiso_analysis(user, host, prepath + "/cosmos-aso-wiso",
                                      "EXP003")
sim_dict["LIG-130-S"] = cosmos_wiso_analysis(
    user, host, prepath + "/cosmos-aso-wiso", "Eem130-S2")
sim_dict["LIG-130-H1"] = cosmos_wiso_analysis(
    user, host, prepath + "/cosmos-aso-wiso-hosing", "LIG-130-H1")
sim_dict["LIG-130-H2"] = cosmos_wiso_analysis(
    user, host, prepath + "/cosmos-aso-wiso-hosing", "LIG-130-NAtl-ndO")

model_ssts = {}


use_remote=False

if use_remote:
    model_ssts["PI"] = get_remote_data(
        "pgierz@stan0:/ace/user/pgierz/cosmos-aso-wiso/EXP003/post/mpiom/EXP003_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )
    model_ssts["LIG-130-S"] = get_remote_data(
        "pgierz@stan0:/ace/user/pgierz/cosmos-aso-wiso/Eem130-S2/post/mpiom/Eem130-S2_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )
    model_ssts["LIG-130-H1"] = get_remote_data(
        "pgierz@stan0:/ace/user/pgierz/cosmos-aso-wiso-hosing/LIG-130-H1/post/mpiom/LIG-130-H1_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )
    model_ssts["LIG-130-H2"] = get_remote_data(
        "pgierz@stan0:/ace/user/pgierz/cosmos-aso-wiso-hosing/LIG-130-NAtl-ndO/post/mpiom/LIG-130-NAtl-ndO_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )
else:
    model_ssts["PI"] = netcdf.netcdf_file("/Volumes/Research_HD/cosmos-aso-wiso/EXP003/post/mpiom/EXP003_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc")
    model_ssts["LIG-130-S"] = netcdf.netcdf_file("/Volumes/Research_HD/cosmos-aso-wiso/Eem130-S2/post/mpiom/Eem130-S2_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc")
    model_ssts["LIG-130-H1"] = netcdf.netcdf_file(
        "/Volumes/Research_HD/cosmos-aso-wiso-hosing/LIG-130-H1/post/mpiom/LIG-130-H1_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )
    model_ssts["LIG-130-H2"] = netcdf.netcdf_file(
        "/Volumes/Research_HD/cosmos-aso-wiso-hosing/LIG-130-NAtl-ndO/post/mpiom/LIG-130-NAtl-ndO_mpiom_main_THO_levels_6-100_vertmean_months_678_yearmean.nc"
    )

column_size=2.3
n=column_size*2
f = plt.figure(figsize=(n, n), dpi=300)
gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios=[13, 1], subplot_spec=gs_outer[0], hspace=0.0)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1], wspace=0.2)
#gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[0.5, 0.08, 1])
ax1 = plt.subplot(gs1[0, 0])
ax2 = plt.subplot(gs1[0, 1])
cbar_ax = plt.subplot(gs1[1, :])
ax3 = plt.subplot(gs2[0, 0])
ax4 = plt.subplot(gs2[0, 1])



m1, m2 = [
    map_natl2(thisax=ax, coastlines=False) for ax in [ax1, ax2]
]

SST_CTRL = model_ssts["PI"].filename

cmap = plt.cm.RdBu_r
norm = matplotlib.colors.Normalize(vmin=-3.0, vmax=3.0)


def get_color_from_species(val):
    if proxy_data["Species"][val]=="N. pachy left":
        return "blue"
    elif proxy_data["Species"][val]=="G. bulloides":
        return "red"
    else:
        return "black"


species_color_list = [get_color_from_species(i) for i in range(len(proxy_data))]


for e, m, ax, x, ha in zip(["LIG-130-S", "LIG-130-H1"], [m1, m2],
                     [ax3, ax4], [0.95, 0.05], ["right", "left"]):
    # Ensure we only use the last 50 timesteps of all files
    f1 = CDO.seltimestep("-50/-1", input=model_ssts[e].filename, returnCdf=True, options="-t mpiom1")
    # print(f1.variables)
    f2 = CDO.seltimestep("-50/-1", input=SST_CTRL, returnCdf=True, options="-t mpiom1")
    f3 = CDO.timmean(input="-sub " + f1.filename + " " + f2.filename)
    cf = plot_var_anom_hatching_from_ncdf_file(
        "THO",
        f1,
        f2,
        m,
        levels=[
            -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        ],
        cmap=cmap)
    plt.colorbar(cf, cax=cbar_ax, spacing="proportional", orientation="horizontal",
                 ticks=[-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                 label="SSST Anomaly w.r.t. PI ($^{\circ}$C)", drawedges=True)
    cbar_ax.set_xticklabels(["-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "$\pm$0.1", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"])
    m.scatter(
        proxy_data['Longitude'].values,
        proxy_data["Latitude"].values,
        c=cmap(norm(proxy_data["130 ka SST CTano (C)"].values)),
        s=8,
        cmap=cmap,
        latlon=True,
        edgecolor="black")

    model_vals = [
        CDO.remapnn(
            "lon=" + str(lon) + "/lat=" + str(lat),
            input=f3,
            returnArray="THO").squeeze()
        for lon, lat in zip(proxy_data["Longitude"], proxy_data["Latitude"])
    ]

    # Make 2 sigma errors of model anomalies to compare against the proxy values
    model_errors = [
        CDO.remapnn(
            "lon=" + str(lon) + "/lat=" + str(lat),
            input=CDO.mulc("2", input="-timstd -sub " + f1.filename + " " + f2.filename),
            returnArray="THO").squeeze()
        for lon, lat in zip(proxy_data["Longitude"], proxy_data["Latitude"])
    ]
    model_vals = [t.tolist() for t in model_vals]
    model_errors = [t.tolist() for t in model_errors]
    # print(model_vals, model_errors)
    ax.errorbar(
        proxy_data["130 ka SST CTano (C)"].values,
        model_vals,
        xerr=proxy_data["130 ka SST CTano 2s error"].values,
        yerr=model_errors,
        ecolor=proxy_data["Color"].values,
        mec="black", mfc="white",
        fmt="o")
    ax.plot((-30, 10), (-30, 10), color="gray", marker=None, lw=3)
    ax.set_ylim(-10, 2)
    ax.set_xlim(-10, 2)

    rms = sqrt(mean_squared_error(proxy_data["130 ka SST CTano (C)"], model_vals))
    ax.text(x, 0.1, "RMSE="+str(np.round(rms,2)),
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.5},
            verticalalignment="top",
            horizontalalignment=ha)

#plt.tight_layout()

for ax, l, y in zip([ax1, ax2, ax3, ax4], ["a", "b", "c", "d"], [0.9, 0.9, 0.95, 0.95]):
    ax.text(0.05, y, l,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=7,
            fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.5})

f.text(0.5, 0.1, "Proxy Reconstructed SSST Anomaly ($^{\circ}$C)", ha="center", va="center",
        fontsize=7)
ax3.set_ylabel("Simulated SSST Anomaly ($^{\circ}$C)")

ax1.set_title("LIG-130")
ax2.set_title("LIG-130-H1")
ax3.set_aspect("equal")
ax4.set_aspect("equal")
plt.savefig("Figure1.pdf")
#plt.show()
