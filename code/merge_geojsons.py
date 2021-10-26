import glob
import geopandas as gpd
import shutil

from json import load, JSONEncoder
from optparse import OptionParser
from re import compile
from uploader import Uploader


float_pat = compile(r'^-?\d+\.\d+(e-?\d+)?$')
charfloat_pat = compile(r'^[\[,\,]-?\d+\.\d+(e-?\d+)?$')
precision = 6

env_vars = load_dotenv()

def merge_geojsons(input_folder, outfile):

    infiles = glob.glob(input_folder + '/*.geojson')
    outjson = dict(type='FeatureCollection', features=[])

    for infile in infiles:
        injson = load(open(infile))

        if injson.get('type', None) != 'FeatureCollection':
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        if type(injson.get('features', None)) != list:
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        outjson['features'] += injson['features']

    encoder = JSONEncoder(separators=(',', ':'))
    encoded = encoder.iterencode(outjson)

    format = '%.' + str(precision) + 'f'
    output = open(outfile, 'w')

    for token in encoded:
        if charfloat_pat.match(token):
            # in python 2.7, we see a character followed by a float literal
            output.write(token[0] + format % float(token[1:]))

        elif float_pat.match(token):
            # in python 2.6, we see a simple float literal
            output.write(format % float(token))

        else:
            output.write(token)

def upload_to_il():

    uploader = Uploader(env_vars['IL_USERNAME'], env_vars['IL_PASSWORD'])
    for geojson in glob.glob('combined_geojsons/la/*.geojson'):
        shapefile_name = geojson.replace('.geojson', '')
        gdf = gpd.read_file(geojson)
        if not gdf.empty:
            gdf.to_file(shapefile_name, driver='ESRI Shapefile')
            shutil.make_archive(shapefile_name, 'zip', root_dir=shapefile_name)
            shutil.rmtree(shapefile_name)
            uploader.upload_shapefile(shapefile_name+".zip")

if __name__ == '__main__':
    geojsons_location = 'shapes'
    combined_output_location = 'combined_geojsons/la'
    for json_loc in glob.glob(f'{geojsons_location}/*'):

        day = json_loc.split('/')[-1]
        merge_geojsons(
            json_loc,
            f'{combined_output_location}/ship-{"T".join(day.split("_")[:2])}.geojson'
        )
    # upload_to_il()