---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Automated multiband forced photometry on large datasets

### Summary:
This code performs photometry in an automated fashion at all locations in an input catalog on 4 bands of IRAC data from IRSA and 2 bands of Galex data from MAST.  The resulting catalog is then cross-matched with a Chandra catalog from HEASARC to generate a multiband catalog to facilitate galaxy evolution studies.

The code will run on 2 different science platforms and makes full use of multiple processors to optimize run time on large datasets.

### Input:
- RA and DEC within COSMOS catalog
- desired catalog radius in arcminutes
- mosaics of that region for IRAC and Galex

### Output:
- merged, multiband, science ready pandas dataframe
- IRAC color color plot for identifying interesting populations

### Authors:
Jessica Krick  
David Shupe  
Marziye JafariYazani  
Brigitta Sipocz  
Vandana Desai  
Steve Groom  

### Acknowledgements:
Nyland et al. 2017 for the workflow of the code  
Lang et al. ??? for the Tractor  
Salvato et al. 2018 for nway  
Laigle et al. 2016 for COSMOS2015  
IRSA, MAST, HEASARC  




+++

### Temporary cell to ensure all dependencies are installed:

```{code-cell} ipython3
!pip install -r requirements.txt
```

```{code-cell} ipython3
# standard lib imports

import math
import time
import warnings
import concurrent.futures
import sys
import os
from contextlib import contextmanager

# Third party imports

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

from tractor import (Tractor, PointSource, PixPos, Flux, PixelizedPSF, NullWCS,
                     NullPhotoCal, ConstantSky, Image)

import pandas as pd
import seaborn as sns
import statsmodels
import mpld3

from firefly_client import FireflyClient
import firefly_client.plot as ffplt

from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.ipac.irsa import Irsa
from astroquery.heasarc import Heasarc
from astroquery.mast import Observations

# Local code imports
sys.path.append('../code/')

from determine_source_type import determine_source_type
from extract_cutout import extract_cutout
from find_nconfsources import find_nconfsources
from display_images import display_images
from plot_SED import plot_SED
from nway_write_header import nway_write_header
#from prepare_prf import prepare_prf

%matplotlib inline
```

```{code-cell} ipython3
sys.version
```

### Pull initial catalog from IRSA
- Automatically set up a catalog with ra, dec, photometric redshifts, fiducial band fluxes, & probability that it is a star  
- Catalog we are using is COSMOS2015 (Laigle et al. 2016)  
- Data exploration

```{code-cell} ipython3
#pull a COSMOS catalog from IRSA using astroquery

#make sure the archive isn't limiting our search
#default values of row_limit are often much lower than what we might want 
Irsa.ROW_LIMIT = 3E6  
Irsa.TIMEOUT = 600


#what is the central RA and DEC of the desired catalog
coords = SkyCoord('150.01d 2.2d', frame='icrs')  #COSMOS center acording to Simbad

#how large is the search radius, in arcmin
radius = 15 * u.arcmin #full COSMOS is 48arcmin  #was testing with smaller like 3 or 15

#use Astroquery to get the catalog
#specify only select columns to limit the size of the catalog
cosmos_table = Irsa.query_region(coords, catalog="cosmos2015", radius=radius, 
                                 selcols='ra,dec,id,Ks_FLUX_APER2,Ks_FLUXERR_APER2, PHOTOZ, SPLASH_1_MAG,SPLASH_1_MAGERR, SPLASH_1_FLUX,SPLASH_1_FLUX_ERR,SPLASH_2_FLUX, SPLASH_2_FLUX_ERR,SPLASH_3_FLUX,SPLASH_3_FLUX_ERR,SPLASH_4_FLUX, SPLASH_4_FLUX_ERR, FLUX_GALEX_NUV,FLUX_GALEX_FUV,FLUX_CHANDRA_05_2,FLUX_CHANDRA_2_10, FLUX_CHANDRA_05_10,ID_CHANDRA09 , type,r_MAG_AUTO,r_MAGERR_AUTO, FLUX_24, FLUXERR_24, MAG_GALEX_NUV, MAGERR_GALEX_NUV,MAG_GALEX_FUV, MAGERR_GALEX_FUV')

#select those rows with either chandra fluxes or Galex NUV fluxes
#this limits the catalog size for testing
#ccosmos_table = cosmos_table[(cosmos_table['flux_chandra_05_10']> 0) | (cosmos_table['flux_galex_fuv'] > 0)]
#ccosmos_table = cosmos_table


```

### Pull image datasets from the cloud

+++

#### Use the fornax cloud access API to obtain the IRAC data from the IRSA S3 bucket. 

Details here may change as the prototype code is being added to the appropriate libraries, as well as the data holding to the appropriate NGAP storage as opposed to IRSA resources.

```{code-cell} ipython3
# Temporary solution
# This relies on the assumption that https://github.com/fornax-navo/fornax-cloud-access-API is being cloned to this environment. 
# If it's not, then run a ``git clone https://github.com/fornax-navo/fornax-cloud-access-API --depth=1`` from a terminal at the highest directory root.

# Until https://github.com/fornax-navo/fornax-cloud-access-API/pull/4 is merged clone the fork instead:
# ``git clone https://github.com/bsipocz/fornax-cloud-access-API --depth=1 -b handler_return``

sys.path.append('../../fornax-cloud-access-API')

import pyvo
import fornax
```

```{code-cell} ipython3
# Getting the COSMOS address from the registry to follow PyVO user case approach. We could hardwire it.
image_services = pyvo.regsearch(servicetype='image')
irsa_cosmos = [s for s in image_services if 'irsa' in s.ivoid and 'cosmos' in s.ivoid][0]

# The search returns 11191 entries, but unfortunately we cannot really filter efficiently in the query
# itself (https://irsa.ipac.caltech.edu/applications/Atlas/AtlasProgramInterface.html#inputparam)
# to get only the Spitzer IRAC results from COSMOS as a mission. We will do the filtering in a next step before download.
cosmos_results = irsa_cosmos.search(coords).to_table()

spitzer = cosmos_results[cosmos_results['dataset'] == 'IRAC']
```

```{code-cell} ipython3
# Temporarily add the cloud_access metadata to the Atlas response. 
# This dataset has limited acces, thus 'region' should be used instead of 'open'.
# S3 access should be available from the daskhub and those who has their IRSA token set up.

fname = spitzer['fname']
spitzer['cloud_access'] = [(f'{{"aws": {{ "bucket": "irsa-mast-tike-spitzer-data",'
                            f'             "region": "us-east-1",'
                            f'             "access": "region",'
                            f'             "path": "data/COSMOS/{fn}" }} }}') for fn in fname]
```

```{code-cell} ipython3
# Adding function to download multiple files using the fornax API. 
# Requires https://github.com/fornax-navo/fornax-cloud-access-API/pull/4
def fornax_download(data_table, data_directory='../data', access_url_column='access_url',
                    fname_filter=None, verbose=False):
    working_dir = os.getcwd()
    
    os.chdir(data_directory)
    for row in data_table:
        if fname_filter is not None and fname_filter not in row['fname']:
            continue
        handler = fornax.get_data_product(row, 'aws', access_url_column=access_url_column, verbose=verbose)
        handler.download()
        
    os.chdir(working_dir)
```

```{code-cell} ipython3
fornax_download(spitzer, access_url_column='sia_url', fname_filter='go2_sci', 
                data_directory='../data/IRAC', verbose=True)
```

#### Use astroquery.mast to obtain Galex from the MAST archive

```{code-cell} ipython3
#the Galex mosaic of COSMOS is broken into 4 seperate images
#need to know which Galex image the targets are nearest to.
#make a new column in dataframe which figures this out

#four centers for 1, 2, 3, 4 are
ra_center=[150.369,150.369,149.869,149.869]
dec_center=[2.45583,1.95583,2.45583,1.95583]

#ra_center = 150.369
#dec_center = 2.45583
galex = SkyCoord(ra = ra_center*u.degree, dec = dec_center*u.degree)
catalog = SkyCoord(ra = cosmos_table['ra'], dec = cosmos_table['dec'])
#idx, d2d, d3d = match_coordinates_sky(galex, catalog)  #only finds the nearest one
#idx, d2d, d3d = galex.match_to_catalog_sky(catalog)  #only finds the nearest one

cosmos_table['COSMOS_01'] = galex[0].separation(catalog)
cosmos_table['COSMOS_02'] = galex[1].separation(catalog)
cosmos_table['COSMOS_03'] = galex[2].separation(catalog)
cosmos_table['COSMOS_04'] = galex[3].separation(catalog)

#convert to pandas
df = cosmos_table.to_pandas()

#which row has the minimum value of distance to the galex images
df['galex_image'] = df[['COSMOS_01','COSMOS_02','COSMOS_03','COSMOS_04']].idxmin(axis = 1)
```

```{code-cell} ipython3
# 76k with 15arcmin diameter IRAC images
df.describe()
```

```{code-cell} ipython3
#pull Galex mosaics from MAST
# Get the observations you want
in_coordinates = '150.01 2.20'
observations = Observations.query_criteria(coordinates=in_coordinates, instrument_name='GALEX')

filtered_observations = observations[(observations['t_exptime'] > 40000.0)]

# Get the products for these observations 
products = Observations.get_product_list(filtered_observations)

# Filter the products so we only download SCIENCE products
filtered_products = Observations.filter_products(products, productType='SCIENCE', productGroupDescription='Minimum Recommended Products')

# Enable cloud access
Observations.enable_cloud_dataset(provider='AWS')

#uncomment to actually download the data
# Download filtered products 
#Observations.download_products(filtered_products, cloud_only=True, download_dir = '../data/Galex/') 
```

```{code-cell} ipython3
#testing to get the GALEX skybg fits files in addition to the mosaics
#don't have this working yet, instead pull these files manually
# get observations
in_coordinates = '150.01 2.20'
observations = Observations.query_criteria(coordinates=in_coordinates, instrument_name='GALEX')

# get products of said observations (i'm just doing the first one) 
products = Observations.get_product_list(observations)

# filtering for skybg 
skybg_products = []
#for irow, row in enumerate(products['dataURI']):
for row in products['dataURI']:
    if 'COSMOS_01-fd-skybg' in row: 
       print(row)
#      skybg_products.append(products[irow])
       skybg_products.append(str(row))
       #Observations.download_file(skybg_products, cloud_only=True, local_path = '../data/Galex/') 
```

```{code-cell} ipython3
#make sure there aren't any troublesome rows in the catalog
#are there missing values in any rows?
df.isna().sum()

#don't mind that there are missing values for some of the fluxes
#The rest of the rows are complete
```

```{code-cell} ipython3
#out of curiosity how many of each type of source are in this catalog
#Type: 0 = galaxy, 1 = star, 2 = X-ray source, -9 is failure to fit
df.type.value_counts()
```

### Setup to run forced photometry
- initialize data frame columsn to hold the results
- supress debugging output of tractor 
- build necessary arrays for multiple bands

```{code-cell} ipython3
####purely for testing
#df = df.head()
```

```{code-cell} ipython3
# initialize columns in data frame for photometry results
df[["ch1flux","ch1flux_unc","ch2flux","ch2flux_unc","ch3flux","ch3flux_unc","ch4flux","ch4flux_unc","ch5flux","ch5flux_unc","ch6flux","ch6flux_unc"]] = 0.0
```

```{code-cell} ipython3
#setup to supress output of tractor
#seems to be the only way to make it be quiet and not output every step of optimization
#https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
```

```{code-cell} ipython3
# parameters needed for the next function

#IRAC
irac_fluxconversion = (1E12) / (4.254517E10) * (0.6) *(0.6) #convert tractor result to microjanskies
mosaic_pix_scale_irac = 0.6
cutout_width_irac = 10 # in arcseconds, taken from Nyland et al. 2017
bkg_method_irac = 'image'  #use the science image itself

#Galex
cutout_width_GALEX = 40
GALEX_nuv_fluxconversion= 3.373E1 #uJy  fudging this to make the numbers bigger for plotting later
GALEX_fuv_fluxconversion=1.076E2  #uJy fudging this to make the numbers bigger for plotting later
mosaic_pix_scale_GALEX = 1.5
bkg_method_GALEX = 'skybg'  #use the GALEX provided skybg fits file


cutout_width_list=[cutout_width_irac,cutout_width_irac,cutout_width_irac,cutout_width_irac,cutout_width_GALEX,cutout_width_GALEX]
flux_conv_list=[irac_fluxconversion,irac_fluxconversion,irac_fluxconversion,irac_fluxconversion,GALEX_nuv_fluxconversion,GALEX_fuv_fluxconversion]
mosaic_pix_scale_list=[mosaic_pix_scale_irac,mosaic_pix_scale_irac,mosaic_pix_scale_irac,mosaic_pix_scale_irac,mosaic_pix_scale_GALEX,mosaic_pix_scale_GALEX]
background_method_list = [bkg_method_irac,bkg_method_irac,bkg_method_irac,bkg_method_irac,bkg_method_GALEX,bkg_method_GALEX]

#GALEX MASTER PSFs
prf_nuv = fits.open("../data/Galex/PSFnuv_faint.fits")[0].data
prf_fuv= fits.open("../data/Galex/PSFfuv.fits")[0].data
prf_nuv=prf_nuv[0:119, 0:119]
prf_fuv=prf_fuv[0:119, 0:119]
#these are much larger than the cutouts we are using, so only keep the central region which is the size of our cutouts
ngalex_pix = cutout_width_GALEX / mosaic_pix_scale_GALEX
prf_cen = int(60)
prf_nuv = prf_nuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)), (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]
prf_fuv = prf_fuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)), (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]


#set up prfs for each channel
prfs = [fits.open('../data/IRAC/PRF_IRAC_ch1.fits')[0].data,
        fits.open('../data/IRAC/PRF_IRAC_ch2.fits')[0].data,
        fits.open('../data/IRAC/PRF_IRAC_ch3.fits')[0].data,
        fits.open('../data/IRAC/PRF_IRAC_ch4.fits')[0].data, prf_nuv, prf_fuv]
    
#set up mosaics for each channel
#for now we are manually creating these mosaics using coords above and
#https://irsa.ipac.caltech.edu/data/COSMOS/index_cutouts.html
#https://irsa.ipac.caltech.edu/data/COSMOS/
#infiles = ['../data/IRAC/COSMOS_IRAC_ch1_mosaic_15arcmin.fits',
#           '../data/IRAC/COSMOS_IRAC_ch2_mosaic_15arcmin.fits',
#           '../data/IRAC/COSMOS_IRAC_ch3_mosaic_15arcmin.fits',
#           '../data/IRAC/COSMOS_IRAC_ch4_mosaic_15arcmin.fits',
#           '../data/Galex/COSMOS_galex_nuv_mosaic_15arcmin.fits',
#           '../data/Galex/COSMOS_galex_fuv_mosaic_15arcmin.fits']

#setup for full field of view
infiles = ['../data/IRAC/COSMOS_IRAC_ch1_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch2_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch3_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch4_mosaic.fits',
           '../data/Galex/COSMOS_01-nd-int.fits','../data/Galex/COSMOS_01-fd-int.fits',
           '../data/Galex/COSMOS_02-nd-int.fits','../data/Galex/COSMOS_02-fd-int.fits',
           '../data/Galex/COSMOS_03-nd-int.fits','../data/Galex/COSMOS_03-fd-int.fits',
           '../data/Galex/COSMOS_04-nd-int.fits','../data/Galex/COSMOS_04-fd-int.fits']

skybgfiles = ['../data/IRAC/COSMOS_IRAC_ch1_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch2_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch3_mosaic.fits',
           '../data/IRAC/COSMOS_IRAC_ch4_mosaic.fits',
           '../data/Galex/COSMOS_01-nd-skybg.fits','../data/Galex/COSMOS_01-fd-skybg.fits',
           '../data/Galex/COSMOS_02-nd-skybg.fits','../data/Galex/COSMOS_02-fd-skybg.fits',
           '../data/Galex/COSMOS_03-nd-skybg.fits','../data/Galex/COSMOS_03-fd-skybg.fits',
           '../data/Galex/COSMOS_04-nd-skybg.fits','../data/Galex/COSMOS_04-fd-skybg.fits']

#3 arcmin radius mosaics
#'../data/IRAC/COSMOS_irac_ch1_mosaic_recenter.fits',
#           '../data/IRAC/COSMOS_irac_ch2_mosaic_recenter.fits',
#           '../data/IRAC/COSMOS_irac_ch3_mosaic_recenter.fits',
#           '../data/IRAC/COSMOS_irac_ch4_mosaic_recenter.fits',
#           '../data/Galex/0001_150.01000000_2.20000000_COSMOS_01-nd-int.fits',
#           '../data/Galex/0001_150.01000000_2.20000000_COSMOS_01-fd-int.fits']


#read in those mosaics 
hdulists = [fits.open(infiles[0])[0], fits.open(infiles[1])[0],fits.open(infiles[2])[0],fits.open(infiles[3])[0],fits.open(infiles[4])[0],fits.open(infiles[5])[0],fits.open(infiles[6])[0],fits.open(infiles[7])[0],fits.open(infiles[8])[0],fits.open(infiles[9])[0],fits.open(infiles[10])[0],fits.open(infiles[11])[0]]
headers = [hdulists[0].header,hdulists[1].header,hdulists[2].header,hdulists[3].header,hdulists[4].header,hdulists[5].header,hdulists[6].header,hdulists[7].header,hdulists[8].header,hdulists[9].header,hdulists[10].header,hdulists[11].header]
bkg_hdus = [fits.open(skybgfiles[0])[0], fits.open(skybgfiles[1])[0],fits.open(skybgfiles[2])[0],fits.open(skybgfiles[3])[0],fits.open(skybgfiles[4])[0],fits.open(skybgfiles[5])[0],fits.open(infiles[6])[0],fits.open(skybgfiles[7])[0],fits.open(skybgfiles[8])[0],fits.open(skybgfiles[9])[0],fits.open(skybgfiles[10])[0],fits.open(skybgfiles[11])[0]]

#grab the WCS of the mosaics
wcs_infos = [wcs.WCS(hdulists[0]),wcs.WCS(hdulists[1]),wcs.WCS(hdulists[2]),wcs.WCS(hdulists[3]),wcs.WCS(hdulists[4]),wcs.WCS(hdulists[5]),wcs.WCS(hdulists[6]),wcs.WCS(hdulists[7]),wcs.WCS(hdulists[8]),wcs.WCS(hdulists[9]),wcs.WCS(hdulists[10]),wcs.WCS(hdulists[11])]
```

### A little Data Exploration

```{code-cell} ipython3
#Use IRSA's firefly to display image and overlay table
#just so we know what the data looks like
fc = FireflyClient.make_client()
```

```{code-cell} ipython3
#give firefly one of the mosaics we are using here
imval = fc.upload_file(infiles[0])
status = fc.show_fits(file_on_server=imval, plot_id="IRAC_ch1", title='IRAC ch1')

#and give firefly a table
#first convert to fits table from pandas
t_df = Table.from_pandas(df)
tablename = '../data/IRAC/COSMOS_table.fits'
t_df.write(tablename, overwrite = "True")
file= fc.upload_file(tablename)
status = fc.show_table(file, tbl_id='df', title='COSMOS catalog')

#this should work, and is simpler, but isn't working.
#file_table = ffplt.upload_table(t_df, title = 'COSMOS catalog')

```

#### Note: 
This view will not display all of the catalog rows overlaid on the image.  To do that, narrow down the catalog size by filtering on the catalog inside of the IRSA Viewer web browser.  Documentation for how to interacto with IRSA Viewer is here: https://irsa.ipac.caltech.edu/onlinehelp/irsaviewer/

+++

### Main Function to do the forced photometry

```{code-cell} ipython3
def calc_instrflux(band, ra, dec, stype, ks_flux_aper2, g_band):
    """
    calculate instrumental fluxes and uncertainties for four IRAC bands 
    
    Parameters:
    -----------
    band: int
        integer in [0, 1, 2, 3,4, 5] for the four IRAC bands and two Galex bands
    ra, dec: float or double
        celestial coordinates for measuring photometry
    stype: int
        0, 1, 2, -9 for star, galaxy, x-ray source
    ks_flux_aper_2: float
        flux in aperture 2
        
    Returns:
    --------
    outband: int
        reflects input band for identification purposes
    flux: float
        measured flux in microJansky, NaN if unmeasurable
    unc: float
        measured uncertainty in microJansky, NaN if not able to estimate
    """
    prf = prfs[band]
    infile = infiles[g_band]
    hdr = headers[g_band]
    cutout_width=cutout_width_list[band]
    mosaic_pix_scale=mosaic_pix_scale_list[band]
    flux_conv=flux_conv_list[band]
    background_method = background_method_list[band]
    
    #tractor doesn't need the entire image, just a small region around the object of interest
    subimage, nodata_param, x1, y1, subimage_wcs = extract_cutout(ra, dec,cutout_width, mosaic_pix_scale, hdulists[g_band], wcs_infos[g_band])
    #for the Galex images, also need to make a background cutout image
    if background_method == 'skybg':
        bgsubimage, bgnodata_param, bgx1, bgy1, bgimage_wcs = extract_cutout(ra, dec,cutout_width, mosaic_pix_scale, hdulists[g_band], wcs_infos[g_band])
    
    
    #catch errors in making the cutouts
    if nodata_param == False:  #meaning we have data in the cutout
        
        #set up the source list by finding neighboring sources
        objsrc, nconfsrcs = find_nconfsources(ra, dec, stype,
                        ks_flux_aper2, x1,y1, cutout_width, subimage_wcs, df)

        #measure sky noise and mean level
        #suppress warnings about nans in the calculation
        if background_method == 'image':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                skymean, skymedian, skynoise = sigma_clipped_stats(subimage, sigma=3.0)
        if background_method == 'skybg':
            skymean, skymedian, skynoise = sigma_clipped_stats(bgsubimage, sigma=3.0)
            
        #make the tractor image
        tim=Image(data=subimage, invvar=np.ones_like(subimage) / skynoise**2, 
              psf=PixelizedPSF(prf) ,
              wcs=NullWCS(),photocal=NullPhotoCal(),sky=ConstantSky(skymean))
               
        # make tractor object combining tractor image and source list
        tractor=Tractor([tim], objsrc) #[src]

        #freeze the parameters we don't want tractor fitting
        tractor.freezeParam('images') #now fits 2 positions and flux
        #tractor.freezeAllRecursive()#only fit for flux
        #tractor.thawPathsTo('brightness')


        #run the tractor optimization (do forced photometry)
        # Take several linearized least squares steps
        fit_fail = False
        try:
            tr = 0
            with suppress_stdout():
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*divide by zero.*')
                    #warnings.simplefilter('ignore')
                    for tr in range(20):
                        dlnp,X,alpha, flux_var=tractor.optimize(variance = True)
                        #print('dlnp',dlnp)
                        if dlnp < 1e-3:
                            break
        # catch exceptions and bad fits
        except:
            fit_fail = True
            
        # record the photometry results
        if fit_fail: 
            #tractor fit failed
            #set flux and uncertainty as nan and move on
            return(band, np.nan, np.nan)
        elif flux_var is None:  
            #fit worked, but flux variance did not get reported
            params_list=objsrc[0].getParamNames()
            bindex = params_list.index('brightness.Flux')
            flux = objsrc[0].getParams()[bindex]
             #convert to microjanskies
            microJy_flux = flux * flux_conv
            return(band, microJy_flux, np.nan)
        else: 
            #fit and variance worked
            params_list=objsrc[0].getParamNames()
            bindex = params_list.index('brightness.Flux')
            flux = objsrc[0].getParams()[bindex]
                
            # determine flux uncertainty
            #which value of flux_var is for the flux variance?
            fv = ((nconfsrcs+1)*3) - 1  #assumes we are fitting positions and flux
            #fv = ((nconfsrcs+1)*1) - 1  #assumes we are fitting only flux

            tractor_std = np.sqrt(flux_var[fv])  
                
            #convert to microjanskies
            microJy_flux = flux * flux_conv
            microJy_unc = tractor_std *flux_conv
            return(band, microJy_flux, microJy_unc)
        
    else:
        return(band, np.nan, np.nan)
```

### Calculate forced photometry

+++

#### Straightforward but slow method
no longer in use

```{raw-cell}
:tags: []

%%time
#do the calculation without multiprocessing for benchmarking

#make a copy for parallel computation
pl_df = df.copy(deep=True)

t0 = time.time()
#for each object
for row in df.itertuples():
    #for each band
    for band in range(6):
        #measure the flux with tractor
        outband, flux, unc = calc_instrflux(band, row.ra, row.dec, row.type, row.ks_flux_aper2)
        #put the results back into the dataframe
        df.loc[row.Index, 'ch{:d}flux'.format(outband+1)] = flux
        df.loc[row.Index, 'ch{:d}flux_unc'.format(outband+1)] = unc
        #print(row.ra, row.dec, row.type, row.ks_flux_aper2, band+1,
        #      outband, flux, unc)
t1 = time.time()


#10,000 sources took 1.5 hours with this code
```

#### Now measure the flux using all of the processors for optimizing speed on large datasets
Parallelization: we can either interate over the rows of the dataframe and run the four bands in parallel; or we could zip together the row index, band, ra, dec, 

```{code-cell} ipython3
paramlist = []
g_band = 4
for row in df.itertuples():
    for band in range(6):
        if band < 4:
            g_band = band
        if band ==4: #galex NUV: need to figure out which galex mosaic to use
            choices = {'COSMOS_01':4, 'COSMOS_02':6,'COSMOS_03':8,'COSMOS_04':10,}
            g_band = choices.get(row.galex_image,'default')
        if band ==5: #galex FUV: need to figure out which galex mosaic to use
            choices = {'COSMOS_01':5, 'COSMOS_02':7,'COSMOS_03':9,'COSMOS_04':11,}
            g_band = choices.get(row.galex_image,'default')
        paramlist.append([row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, g_band])
            
```

```{code-cell} ipython3
#test this out on one object
calc_instrflux(paramlist[0][1], paramlist[0][2], paramlist[0][3], paramlist[0][4], paramlist[0][5], paramlist[0][6])

#same thing, different syntax
#calc_instrflux(*paramlist[0][1:])
```

```{code-cell} ipython3
#wrapper to measure the photometry on a single object, single band
def calculate_flux(args):
    """Calculate flux."""
    f = calc_instrflux
    val = f(*args[1:])
    return(args[0], val)
```

```{code-cell} ipython3
%%time
#Here is where the multiprocessing work gets done
t2 = time.time()
outputs = []
with concurrent.futures.ProcessPoolExecutor(24) as executor:
    for result in executor.map(calculate_flux, paramlist):
        # print(result)
        df.loc[result[0],
                  'ch{:d}flux'.format(result[1][0] + 1)] = result[1][1]
        df.loc[result[0],
                  'ch{:d}flux_unc'.format(result[1][0] + 1)] = result[1][1]
        outputs.append(result)
t3 = time.time()
```

```{code-cell} ipython3
#print('Serial calculation took {:.2f} seconds'.format((t1 - t0)))
print('Parallel calculation took {:.2f} seconds'.format((t3 - t2)))
#print('Speedup is {:.2f}'.format((t1 - t0) / (t3 - t2)))

#speedup was factors of 10 - 12 for 400 - 10000 sources
```

```{code-cell} ipython3
#Count the number of non-zero ch1 fluxes
#print('Serial calculation: number of ch1 fluxes filled in =',
#      np.sum(df.ch1flux > 0))
print('Parallel calculation: number of ch1 fluxes filled in =',
      np.sum(df.ch1flux > 0))
```

```{code-cell} ipython3
#had to call the galex flux columns ch5 and ch6
#fix that by renaming them now
df.rename(columns={'ch5flux':'nuvflux', 'ch5flux_unc':'nuvflux_unc','ch6flux':'fuvflux', 'ch6flux_unc':'fuvflux_unc'}, inplace = True)
#pl_df.rename(columns={'ch5flux':'nuvflux', 'ch5flux_unc':'nuvflux_unc','ch6flux':'fuvflux', 'ch6flux_unc':'fuvflux_unc'}, inplace = True)
```

```{code-cell} ipython3
df
```

### Plotting to confirm photometry results against COSMOS 2015 catalog

```{code-cell} ipython3
%%time
#plot tractor fluxes vs. catalog splash fluxes
#should see a straightline with a slope of 1
#using sns regplot which plots both the data and a linear regression model fit
#this plotting tool is for visualization and not statistics, so I don't have rigorous slopes from it.

#setup to plot
fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2)
fluxmax = 200
ymax = 100
xmax = 100
#ch1 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_1_flux> 0) & (df.splash_1_flux < fluxmax)] #200
#sns.regplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1)

#add a diagonal line with y = x
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax1.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 3.6')
ax1.set_ylim([0, ymax])
ax1.set_xlim([0, xmax])


#ch2 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_2_flux> 0) & (df.splash_2_flux < fluxmax)]
#sns.regplot(data = df_tractor, x = "splash_2_flux", y = "ch2flux", ax = ax2, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_2_flux", y = "ch2flux", ax = ax2)

#add a diagonal line with y = x
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 4.5')
ax2.set_ylim([0, ymax])
ax2.set_xlim([0, xmax])


#ch3 
#first shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_3_flux> 0) & (df.splash_3_flux < fluxmax)]

#sns.regplot(data = df_tractor, x = "splash_3_flux", y = "ch3flux", ax = ax3, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_3_flux", y = "ch3flux", ax = ax3)

#add a diagonal line with y = x
lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
    np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax3.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 5.8')
ax3.set_ylim([0, ymax])
ax3.set_xlim([0, xmax])


#ch4 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_4_flux> 0) & (df.splash_4_flux < fluxmax)]

#sns.regplot(data = df_tractor, x = "splash_4_flux", y = "ch4flux", ax = ax4, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_4_flux", y = "ch4flux", ax = ax4)

#add a diagonal line with y = x
lims = [
    np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
    np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax4.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 8.0')
ax4.set_ylim([0, ymax])
ax4.set_xlim([0, xmax])

#-------
#nuv 
#first shrink the dataframe to only those rows where I have tractor photometry while testing
df_tractor = df[(df.flux_galex_nuv> 0) & (df.flux_galex_nuv < 20 )]

#sns.regplot(data = df_tractor, x = "flux_galex_nuv", y = "nuvflux", ax = ax5, robust = True)
sns.scatterplot(data = df_tractor, x = "flux_galex_nuv", y = "nuvflux", ax = ax5)


#add a diagonal line with y = x
#lims = [
#    np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
#    np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
#]

# now plot both limits against eachother
#ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax5.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'Galex NUV')
ax5.set_yscale('log')
#-------
#fuv 
#first shrink the dataframe to only those rows where I have tractor photometry while testing
df_tractor = df[(df.flux_galex_fuv> 0) & (df.flux_galex_fuv < 20 )]

#sns.regplot(data = df_tractor, x = "flux_galex_fuv", y = "fuvflux", ax = ax6, robust = True)
sns.scatterplot(data = df_tractor, x = "flux_galex_fuv", y = "fuvflux", ax = ax6)


#add a diagonal line with y = x
#lims = [
#    np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
#    np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
#]

# now plot both limits against eachother
#ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax6.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'Galex FUV')
ax6.set_yscale('log')



plt.tight_layout()

fig.subplots_adjust( hspace=0.5)
fig.set_size_inches(8, 12)

#plt.savefig('flux_comparison.png')
```

Tractor is working for IRAC; Comparison of tractor derived fluxes with COSMOS 2015 fluxes for all four Spitzer IRAC channels.  Blue points represent each object from the subset of the COSMOS 2015 catalog.  The blue line is a linear regression robust fit to the data with uncertainties shown as the light blue wedge.  The black line is a y = x line plotted to guide the eye.

```{code-cell} ipython3
#save the dataframe with the forced photometry
#df.to_pickle('../data/COSMOS_15arcmin.pkl')

#or read it back in
#df = pd.read_pickle('../data/COSMOS_15arcmin_FUV.pkl')
```

## Cross match this newly built catalog with an X-ray catalog
We are using nway as the tool to do the cross match Salvato et al. 2017.
nway expects input as two fits table files and outputs a third table file with all the possible matches and their probabilities of being the correct match.  We then sort that catalog and take only the best matches to be the true matches.

```{code-cell} ipython3
#first get an X-ray catalog from Heasarc
heasarc = Heasarc()
table = heasarc.query_mission_list()
mask = (table['Mission'] =="CHANDRA")
chandratable = table[mask]  

#tell me which tables exist there
#chandratable.pprint(max_lines = 200, max_width = 130)

#want ccosmoscat
mission = 'ccosmoscat'
#coords already defined above where I pull the original COSMOS catalog
ccosmoscat_rad = 1 #radius of chandra cosmos catalog
ccosmoscat = heasarc.query_region(coords, mission=mission, radius='1 degree', resultmax = 5000, fields = "ALL")
```

```{code-cell} ipython3
#astropy doesn't recognize capitalized units
#so there will be some warnings here on writing out the file, but we can safely ignore those

#need to make the chandra catalog into a fits table 
#and needs to include area of the survey.
ccosmoscat.meta['NAME'] = 'CHANDRA'
ccosmoscat.meta['SKYAREA'] = float(1.0)  #in square degrees

#also need an 'ID' column
ccosmoscat['ID'] = range(1, len(ccosmoscat) + 1)
ccosmoscat.write('../data/Chandra/COSMOS_chandra.fits', overwrite = "True")

#above isn't working to get the name into the table
#try this
nway_write_header('../data/Chandra/COSMOS_chandra.fits', 'CHANDRA', float(ccosmoscat_rad**2) )

```

```{code-cell} ipython3
#also need to transform the main pandas dataframe into fits table for nway

#make an index column for tracking later
df['ID'] = range(1, len(df) + 1)

#need this to be a fits table and needs to include area of the survey.
df_table = Table.from_pandas(df)
df_table
df_table.meta['NAME'] = 'OPT'
df_table.meta['SKYAREA'] = float((2*rad_in_arcmin/60)**2) # catalog

df_table.write('../data/multiband_phot.fits', overwrite = "True")

#above isn't working to get the name into the table
#try this
nway_write_header('../data/multiband_phot.fits', 'OPT', float((2*rad_in_arcmin/60)**2) )
```

```{code-cell} ipython3
#nway calling sequence
!nway.py '../data/Chandra/COSMOS_chandra.fits' :ERROR_RADIUS '../data/multiband_phot.fits' 0.1 --out=../data/Chandra/chandra_multiband.fits --radius 15 --prior-completeness 0.9
```

```{code-cell} ipython3
#Clean up the cross match results and merge them back into main pandas dataframe

#read in the nway matched catalog
xmatch = Table.read('../data/Chandra/chandra_multiband.fits', hdu = 1)
df_xmatch = xmatch.to_pandas()

#manual suggests that p_i should be greater than 0.1 for a pure catalog.
#ok, so the matched catalog has multiple optical associations for some of the XMM detections.
#simplest thing to do is only keep match_flag = 1
matched = df_xmatch.loc[(df_xmatch['p_i']>=0.1) & df_xmatch['match_flag']==1]

#merge this info back into the df_optical dataframe.
merged = pd.merge(df, matched, 'outer',left_on='ID', right_on = 'OPT_ID')

#will need to delete unnecessary rows that matched has duplicated from pl_df
#for col in merged.columns:
#    print(col)

#remove all the rows which start with "OPT" because they are duplications of the original catalog
merged = merged.loc[:, ~merged.columns.str.startswith('OPT')]

#somehow the matching is giving negative fluxes in the band where there is no detection 
#if there is a detection in the other band
#clean that up to make those negative fluxes = 0

merged.loc[merged['flux_chandra_2_10'] < 0, 'flux_chandra_2_10'] = 0
merged.loc[merged['flux_chandra_05_2'] < 0, 'flux_chandra_05_2'] = 0
```

```{code-cell} ipython3
#How many CHandra sources are there?

#make a new column which is a bool of existing chandra measurements
merged['chandra_detect'] = 0
merged.loc[merged.CHANDRA_FLUX > 0,'chandra_detect']=1

#make one for Galex too
merged['galex_detect'] = 0
merged.loc[merged.flux_galex_nuv > 0,'galex_detect']=1


print('number of Chandra detections =',np.sum(merged.chandra_detect > 0))
print('number of Galex detections =',np.sum(merged.galex_detect > 0))
```

### Plotting to confirm photometry results against COSMOS 2015 catalog

```{code-cell} ipython3
#Plot 
fig, (ax1,ax2) = plt.subplots(1,2)
#first shrink the dataframe to only those rows where I have tractor photometry while testing
merged_small = merged[(merged.chandra_detect >= 0) ] 

sns.scatterplot(data = merged_small, x = "CHANDRA_HB_FLUX", y = "flux_chandra_2_10", ax = ax2)#, robust = True)#scatterplot
#add a diagonal line with y = x
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set(xlabel = 'COSMOS (erg/s/cm2)', ylabel = 'nway matched (erg/s/cm2)', title = 'Chandra HB (2 - 10)')


sns.scatterplot(data = merged_small, x = "CHANDRA_SB_FLUX", y = "flux_chandra_05_2", ax = ax1)#, robust = True)#scatterplot
#add a diagonal line with y = x
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax1.set(xlabel = 'COSMOS (erg/s/cm2)', ylabel = 'nway matched(erg/s/cm2)', title = 'Chandra SB (05 - 2)')


```

## Make some plots which show off the results and facilitate science

```{code-cell} ipython3
#IRAC color color plots akin to Lacy et al. 2004
#overplot galex sources
#overplot xray sources

#first select on 24 micron 
merged_24 = merged[(merged.flux_24 >= 0) ] 

#negative Galex fluxes are causing problems
merged_24.loc[merged_24.fuvflux < 0, 'fuvflux'] = 0
merged_24.loc[merged_24.nuvflux < 0, 'nuvflux'] = 0


fig, ax = plt.subplots()
merged_24['F5.8divF3.6'] = merged_24.ch3flux / merged_24.ch1flux
merged_24['F8.0divF4.5'] = merged_24.ch4flux / merged_24.ch2flux

merged_allirac = merged_24[(merged_24['F8.0divF4.5'] > 0) & (merged_24['F5.8divF3.6'] > 0)]

#plot all the points
sns.scatterplot(data = merged_allirac, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5)

#plot only those points with Galex detections
galex_detect = merged_allirac[merged_allirac.galex_detect > 0]
sns.scatterplot(data = galex_detect, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5)

#plot only those points with chandra detections
chandra_detect = merged_allirac[merged_allirac.chandra_detect > 0]
sns.scatterplot(data = chandra_detect, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax)



ax.set(xscale="log", yscale="log")
ax.set_ylim([0.1, 10])
ax.set_xlim([0.1, 10])

ax.set(xlabel = 'log F5.8/F3.6', ylabel = 'log F8.0/F4.5')
plt.legend([],[], frameon=False)

#apparently there is a known bug in mpld3 that it doesn't work with log scaled plots
#mpld3.display(fig)  
```

```{code-cell} ipython3
#UV IR color color plot akin to Bouquin et al. 2015
fig, ax = plt.subplots()
merged['FUV-NUV'] = merged.mag_galex_fuv - merged.mag_galex_nuv
merged['NUV-3.6'] = merged.mag_galex_nuv - merged.splash_1_mag


#plot all the points
#sns.scatterplot(data = merged, x = 'NUV-3.6', y = 'FUV-NUV',
#                 ax = ax, alpha = 0.5)

#plot only those points with Galex detections
galex_detect = merged[merged.galex_detect > 0]
sns.kdeplot(data = galex_detect, x = 'NUV-3.6', y = 'FUV-NUV',
                 ax = ax, fill = True, levels = 15)#scatterplot , alpha = 0.5

#plot only those points with chandra detections
#now with color coding Chandra sources by hardness ratio a la Moutard et al. 2020
chandra_detect = merged[merged.chandra_detect > 0]
sns.scatterplot(data = chandra_detect, x = 'NUV-3.6', y = 'FUV-NUV',
                 ax = ax, hue= 'CHANDRA_HARDNESS_RATIO',palette="flare")

#whew that legend for the hue is terrible
#try making it into a colorbar outside the plot instead
norm = plt.Normalize(merged['CHANDRA_HARDNESS_RATIO'].min(), merged['CHANDRA_HARDNESS_RATIO'].max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)

#ax.set(xscale="log", yscale="log")
ax.set_ylim([-0.5, 3.5])
ax.set_xlim([-1, 7])

ax.set(xlabel = 'NUV - [3.6]', ylabel = 'FUV - NUV')
#plt.legend([],[], frameon=False)

#fig.savefig("../data/color_color.png")
mpld3.display(fig)  
```

We extend the works of Bouquin et al. 2015 and Moutard et al. 2020 by showing a GALEX - Spitzer color color diagram over plotted with Chandra detections.  Blue galaxies in these colors are generated by O and B stars and so must currently be forming stars. We find a tight blue cloud in this color space identifying those star forming galaxies.  Galaxies off of the blue cloud have had their star formation quenched, quite possibly by the existence of an AGN through removal of the gas reservoir required for star formation.  Chandra detected galaxies host AGN, and while those are more limited in number, can be shown here to be a hosted by all kinds of galaxies, including quiescent galaxies which would be in the upper right of this plot.  This likely implies that AGN are indeed involved in quenching star formation.  Additionally, we show the Chandra hardness ratio (HR) color coded according to the vertical color bar on the right side of the plot.  HR is defined as (H-S)/ (H+S) where H and S are the hard[2-10KeV] and soft[0.5-2KeV] bands of Chandra.  Those AGN with higher hardness ratios have their soft x-ray bands heavily obscured and appear to reside preferentially toward the quiescent galaxies.

```{code-cell} ipython3
#potential plot ideas
#salim et al. 2014 serbia astronomical journal
#(3.6 magniutde) vs. (NUV - 3.6)


#match to cosmos catalog
#get the galex fluxes then make the green valley plots of bouquin et al, 
#then overplot x-ray 

#second option
#match to cosmos for 24 microns and make lacy et al. plot
```

```{code-cell} ipython3
for col in merged.columns:
    print(col)
```

```{code-cell} ipython3

```
