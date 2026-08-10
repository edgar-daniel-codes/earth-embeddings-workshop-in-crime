/*

Basic JavaScript Cheat sheet with useful codes to consult and manipulate 
Google Earth Engine Map and Data Layers 

*/


/*

We start the analysis with the Sentinel Bands 

Sentinel-2 10m Resolution Data Layers — Google Earth Engine
Bands at 10m: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)

*/


/*

 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image

*/

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Map the function over a month of data and take the median.
// Load Sentinel-2 TOA reflectance data (adjusted for processing changes

var dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                  .filterDate('2025-01-01', '2025-01-31')
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);

// RGB Config 
var rgbVis = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};


// NIR config 
var nirVis = {
  min: 0.0,
  max: 0.5,
  bands: ['B8'],
};

// Mexico City Coordinates 
var lon = 19.404438230673044;
var lat =  -99.14337664625175;
var zoom_init = 11;

// Center map view 
Map.setCenter(lat, lon, zoom_init);

// Add RGB layer usign median() for zoom level sampling 
Map.addLayer(dataset.median(), rgbVis, 'RGB');

// Add NIR layer usign median() for zoom level sampling 
Map.addLayer(dataset.median(), nirVis, 'NIR');