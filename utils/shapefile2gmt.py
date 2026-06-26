# Convert a Shapefile to GMT format
# For Linux environment (i.e., where GDAL executables are located in /usr/bin)

from argparse import ArgumentParser
import gdaltools 
#gdaltools.Wrapper.BASEPATH = "C:\programs\gmt6\bin"

class ShapefileConverter(object):
    def __init__(self, inputfile, epsg="EPSG:4326"):
        self.input = inputfile
        self.epsg = epsg
    
    def convertToGMT(self, outputfile, output_epsg=None):
        print(f"Converting from {self.input} to {outputfile}")
        ogr = gdaltools.ogr2ogr()
        ogr.set_encoding("UTF-8")
        ogr.set_input(self.input, srs=self.epsg)
        ogr.set_output(outputfile, file_type='OGR_GMT', srs=output_epsg)
        ogr.execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert geometries from ESRI Shapefile format to GMT format")
    parser.add_argument("input",
                        help="ESRI Shapefile (*.shp)")
    parser.add_argument("output",
                        help="GMT file (*.gmt)")
    args = parser.parse_args()
    sc = ShapefileConverter(args.input)
    sc.convertToGMT(args.output)
