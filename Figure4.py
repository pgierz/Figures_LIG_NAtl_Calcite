from scipy.io import netcdf
import cdo
import matplotlib.pyplot as plt
import publication_settings
CDO = cdo.Cdo(cdfMod="scipy")

AMOC = netcdf.netcdf_file("/Volumes/Research_HD/handmade/animation_recovery/AMOC_index.nc")
AMOC_full = AMOC.variables["var101"].data.squeeze()
AMOC_run = CDO.runmean("30", input=AMOC.filename, returnArray="var101").squeeze()

TEMP = netcdf.netcdf_file("/Volumes/Research_HD/handmade/animation_recovery/recovery_temp.nc")
TEMP_full = CDO.fldmean(input="-sellonlatbox,38,-50,60,-10 "+TEMP.filename, returnArray="THO").squeeze()
TEMP_run = CDO.runmean("30", input="-fldmean -sellonlatbox,38,-50,60,-10 "+TEMP.filename, returnArray="THO").squeeze()

SEAWATER = netcdf.netcdf_file("/Volumes/Research_HD/handmade/animation_recovery/recovery_sw.nc")
SEAWATER_full = CDO.fldmean(input="-sellonlatbox,38,-50,60,-10 "+SEAWATER.filename, returnArray="h2o18_t").squeeze()
SEAWATER_run = CDO.runmean("30", input="-fldmean -sellonlatbox,38,-50,60,-10 "+SEAWATER.filename, returnArray="h2o18_t").squeeze()

CALCITE = netcdf.netcdf_file("/Volumes/Research_HD/handmade/animation_recovery/recovery_calcite.nc")
CALCITE_full = CDO.fldmean(input="-sellonlatbox,38,-50,60,-10 "+CALCITE.filename, returnArray="wiso_shakleton").squeeze()
CALCITE_run = CDO.runmean("30", input="-fldmean -sellonlatbox,38,-50,60,-10 "+CALCITE.filename, returnArray="wiso_shakleton").squeeze()




column_size=2.3
n=column_size*1.5
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(n, 3*n), dpi=300)
[ax.spines['top'].set_visible(False) for ax in [ax1, ax2, ax3, ax4]]
[ax.tick_params(axis="x",
    which="both",
    bottom="off",
    top="off",
    labelbottom="off")
    for ax in [ax2, ax3, ax1]]
[ax.spines['bottom'].set_visible(False) for ax in [ax2, ax3, ax1]]
[ax.spines['left'].set_visible(False) for ax in [ax2, ax4]]
[ax.spines['right'].set_visible(False) for ax in [ax1, ax3]]
[ax.yaxis.tick_right() for ax in [ax2, ax4]]
plt.subplots_adjust(hspace=-0.25, left=0.2, right=0.8, top=1.0, bottom=0.05)


ax1.plot(range(len(CALCITE_full)), CALCITE_full, color="blue", alpha=0.33, linewidth=0.33)
ax1.plot([t+15 for t in range(len(CALCITE_run))], CALCITE_run, color="blue", linewidth=2)
ax1.set_ylabel(r"$\delta^{18}O_{c}$ (\textperthousand {} vs. PDB)")
ax1.invert_yaxis()

ax2.plot(range(len(SEAWATER_full)), SEAWATER_full, color="green", alpha=0.33 ,linewidth=0.33)
ax2.plot([t+15 for t in range(len(SEAWATER_run))], SEAWATER_run, color="green", linewidth=2)
ax2.set_ylabel(r"$\delta^{18}O_{sw}$ (\textperthousand {} vs. SMOW)")
ax2.yaxis.set_label_position("right")

ax3.plot(range(len(TEMP_full)), TEMP_full, color="red", alpha=0.33, linewidth=0.33)
ax3.plot([t+15 for t in range(len(TEMP_run))], TEMP_run, color="red", linewidth=2)
ax3.set_ylabel("Temperature ($^{\circ}$C)")

ax4.plot(range(len(AMOC_full)), AMOC_full, color="gray", linewidth=0.33)
ax4.plot([t+15 for t in range(len(AMOC_run))], AMOC_run, color="black", linewidth=2)
ax4.set_xlabel("Simulation Time (years)")
ax4.set_ylabel("AMOC Index (Sv)")
ax4.yaxis.set_label_position("right")

[ax.patch.set_alpha(0) for ax in [ax1, ax2, ax3, ax4]]

plt.savefig("Figure4.pdf")
#plt.show()
