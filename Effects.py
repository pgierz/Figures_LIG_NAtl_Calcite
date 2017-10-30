#!/usr/bin/env python

import publication_settings
import matplotlib.pyplot as plt
from scipy.io import netcdf
from plot_tools import plot_var_from_ncdf_file
from basemap_wrappers import map_natl2


column_size=2.3
n=column_size*2

f, axs = plt.subplots(1, 3, figsize=(n, 0.33*n), dpi=300)
m1, m2, m3 = [map_natl2(thisax=ax, coastlines=False) for ax in axs]
# [m.drawcoastlines(linewidth=0.25) for m in [m1, m2, m3]]
f.subplots_adjust(bottom=0.15, top=0.95, wspace=0.15, left=0.05, right=0.95)
axs[0].set_title("Temperature Effect")
axs[1].set_title("Circulation Effect")
axs[2].set_title("Dilution Effect")

temp_file = netcdf.netcdf_file("Temperature_Effect_file.nc")
circ_file = netcdf.netcdf_file("Circulation_Effect_file.nc")
dilu_file = netcdf.netcdf_file("Dilution_Effect_file.nc")

cf_args={"levels": [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0],
         "cmap": plt.cm.PiYG_r}
plot_var_from_ncdf_file("THO", temp_file, m1, **cf_args)
plot_var_from_ncdf_file("THO", circ_file, m2, **cf_args)
cf = plot_var_from_ncdf_file("THO", dilu_file, m3, **cf_args)

cbar_ax = f.add_axes([0.15, 0.25, 0.7, 0.05])
cbar = plt.colorbar(cf, spacing="proportional", orientation="horizontal",
                    cax=cbar_ax,
                   ticks=[-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0],
                    label=r"$\delta^{18}O_{c}$ Anomaly (\textperthousand {} vs. PDB)",
                    drawedges=True)
cbar.set_ticklabels(["-2.0", "-1.5", "-1.0", "-0.5", "$\pm$0.1", "0.5", "1.0", "1.5", "2.0"])
plt.savefig("Three_Effects.pdf")
plt.savefig("Figure_2.pdf")
plt.close()


