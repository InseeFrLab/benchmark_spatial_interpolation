# %%
import os
import glob
import re
import shutil
import requests
from osgeo import gdal
import s3fs
import py7zr

# Function to download a file from a URL
def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path+url.rsplit('/', 1)[-1], 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {url} to {local_path+url.rsplit('/', 1)[-1]}")

# %%
# URL of the BD ALTI page
url = "https://geoservices.ign.fr/rgealti"

# Fetch the HTML content
html = requests.get(url).text

# Regex pattern to find all BDALTI download links
pattern = r"https://data\.geopf\.fr/telechargement/download/RGEALTI/[^>]+5M[^>]+LAMB93[^>]+7z"

# Extract all matching URLs
asc_urls = re.findall(pattern, html)

# Remove duplicates and sort
asc_urls = sorted(set(asc_urls))

# %%
for file_url in asc_urls:
    # Detect departement
    prefix = r'https://data\.geopf\.fr/telechargement/download/RGEALTI/RGEALTI_2-0_5M_ASC_LAMB93-(IGN69|IGN78C)_'

    pattern = prefix + r'([^_]+)'

    match = re.search(pattern, file_url)
    if match:
        departement = match.group(2)
        print(departement)
    else:
        raise ValueError("No departement found")
        
    # Prepare folders
    shutil.rmtree("./rawdata", ignore_errors=True)
    shutil.rmtree("./data", ignore_errors=True)
    shutil.rmtree("./finaldata", ignore_errors=True)
    os.mkdir("./rawdata")
    os.mkdir("./data")
    os.mkdir("./finaldata")

    print("Downloading data")
    archivename = file_url.rsplit('/', 1)[-1]
    download_file(file_url, "./rawdata/")

    # Extract all .asc files
    print("Extracting data")
    print(archivename)
    with py7zr.SevenZipFile("./rawdata/"+archivename, mode='r') as archive:
        list_compressed_files = [f for f in archive.namelist() if re.search('1_DONNEES_LIVRAISON.*\\.asc', f)]
        archive.extract(path='./data/', targets=list_compressed_files)

    print("Converting data")
    gdal.BuildVRT('out.vrt',glob.glob("data/**/*.asc", recursive=True))
    gdal.Translate('RGEALTI.tif', 'out.vrt', format='gtiff')

    print("Writing data to S3")
    os.system(f"mc cp RGEALTI.tif s3/oliviermeslin/RGEALTI/RGEALTI_{departement}.tif")

# %%
# Map the altitude
import numpy as np
import matplotlib.pyplot as plt
import os

# Open the TIFF file
dataset = gdal.Open('RGEALTI.tif', gdal.GA_ReadOnly)
band = dataset.GetRasterBand(1)
# Get NODATA value
nodata = band.GetNoDataValue()

# Read raster data as array
array = band.ReadAsArray()

# Mask nodata values by setting them to np.nan
if nodata is not None:
    array = np.where(array == nodata, np.nan, array)

# Plot using matplotlib
plt.imshow(array, cmap='terrain')
plt.colorbar()
plt.title('Map from GeoTIFF using GDAL')
plt.show()

# %%
