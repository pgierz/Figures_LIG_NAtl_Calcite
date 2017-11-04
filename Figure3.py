from scipy.io import netcdf
import pandas as pd
import cdo
import matplotlib.pyplot as plt
from basemap_wrappers import map_natl2
import matplotlib.colors
import publication_settings

CDO = cdo.Cdo(cdfMod="scipy")

LIG130_Data = pd.read_excel("/Users/pgierz/Research/data/LIG_d18Osw_130ka_PG_none_missing.xlsx")

diff_list = []
for lat, lon, val in zip(LIG130_Data["Latitude"].values, LIG130_Data["Longitude"].values, LIG130_Data["130 ka d18Oc (permil PDB)"].values):
    diff_val = CDO.remapnn("lon="+str(lon)+"/lat="+str(lat), input="Combined_no_dilution.nc", returnArray="THO").squeeze() - val
    diff_list.append(diff_val)


mycmap = plt.cm.PiYG_r
norm = matplotlib.colors.Normalize(vmin=-3.0, vmax=0.0)

n = 2.3
f, ax = plt.subplots(1, 1, figsize=(n, n),dpi=300)
m = map_natl2(thisax=ax, coastlines=True)
c = mycmap(norm(diff_list))
print(c)
cf = m.scatter(LIG130_Data["Longitude"].values, LIG130_Data["Latitude"].values, s=8, c=diff_list, norm=norm, cmap=mycmap, latlon=True, edgecolor="black")
m.colorbar(cf, location="bottom", label="Hypothetical Dilution Required")
plt.savefig("Figure3.pdf")

