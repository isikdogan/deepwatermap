import ee
import time

ee.Initialize()

# Select tiles
valid_tiles = ee.FeatureCollection("users/isikdogan/valid_tiles_filtered")
#valid_tiles = tiles.filter(ee.Filter.gt('occurrence', 1.0))
tile_list = valid_tiles.toList(valid_tiles.size())

# Create the dataset by matching inputs and outputs
date_start = '2015-01-01' #'2015-01-01' #'2015-06-01' 
date_end = '2015-12-31' #'2015-02-01' #'2015-07-01'
# TODO: try 1-month composites for cloudier samples
input_bands = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
                  .filterDate(date_start, date_end).median() \
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']) \
                  .uint16()
labels = ee.ImageCollection('JRC/GSW1_0/YearlyHistory') \
                  .filter(ee.Filter.date('2015-01-01', '2015-12-31')) \
                  .select('waterClass').first().uint16()
dataset = input_bands.addBands(labels)

def download_tile(i, tile_list):
    current_tile = tile_list.get(i)
    tile_geometry = ee.Feature(current_tile).geometry().getInfo()["coordinates"]
    task = ee.batch.Export.image.toDrive(
                    image=dataset,
                    description=savepath,
                    folder='tiles_data_cloudy_1',
                    fileNamePrefix=savepath,
                    region=tile_geometry,
                    scale=30)
    task.start()

# Iterate and download
num_tiles = valid_tiles.size().getInfo()
subsample_ratio = 12
for i in range(108918, num_tiles, subsample_ratio):
    savepath = "tile_{}".format(i)
    try:
        download_tile(i, tile_list)
    except Exception, e:
        print(e)
        print("Capacity reached, waiting...")
        time.sleep(1200)
        download_tile(i, tile_list)
    print("Exporting {} ({} / {})".format(savepath, i, num_tiles))
    time.sleep(10)