''' Script to generate and download the dataset using the Google Earth Engine.
You should never need to use this script since we provide a copy of the dataset.
It takes over a month to finish processing the entire dataset using this script.
The script is inclueded in the repository for archival purposes.
'''

import ee
import time

ee.Initialize()

# Select tiles
valid_tiles = ee.FeatureCollection("users/isikdogan/valid_tiles_filtered")
valid_tiles = tiles.filter(ee.Filter.gt('occurrence', 1.0))
tile_list = valid_tiles.toList(valid_tiles.size())

# Create the dataset by matching inputs and outputs
date_start = '2015-01-01'
date_end = '2015-12-31'
input_bands = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
                  .filterDate(date_start, date_end).median() \
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']) \
                  .uint16()
labels = ee.ImageCollection('JRC/GSW1_0/YearlyHistory') \
                  .filter(ee.Filter.date(date_start, date_end)) \
                  .select('waterClass').first().uint16()
dataset = input_bands.addBands(labels)

def download_tile(i, tile_list, save_folder):
    current_tile = tile_list.get(i)
    tile_geometry = ee.Feature(current_tile).geometry().getInfo()["coordinates"]
    task = ee.batch.Export.image.toDrive(
                    image=dataset,
                    description=savepath,
                    folder=save_folder,
                    fileNamePrefix=savepath,
                    region=tile_geometry,
                    scale=30)
    task.start()

# Iterate and download
num_tiles = valid_tiles.size().getInfo()
subsample_ratio = 1
for i in range(0, num_tiles, subsample_ratio):
    savepath = "tile_{}".format(i)
    save_folder = 'tiles_data_{}'.format((i//10000) * 10000)
    try:
        download_tile(i, tile_list, save_folder)
    except Exception, e:
        print(e)
        print("Capacity reached, waiting...")
        time.sleep(1200)
        download_tile(i, tile_list, save_folder)
    print("Exporting {} ({} / {})".format(savepath, i, num_tiles))
    time.sleep(10)