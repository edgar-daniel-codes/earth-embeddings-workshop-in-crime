

/*
Index Calculation
We use different measured bands to calculate important index based on physical 
properties of reflectance of some material using the bands of Sentinel-2 
*/


// Getting city boundaries 
var filteredS2 = s2
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .filter(ee.Filter.date('2019-01-01', '2020-01-01'))
  .filter(ee.Filter.bounds(geometry))
  ;
print(filteredS2);

// Sort collection and pick the less cloudy image 
var filteredS2Sorted = filteredS2.sort('CLOUDY_PIXEL_PERCENTAGE');
var image = filteredS2Sorted.first();

// Config mAP dISPLAY
Map.centerObject(geometry, 10);
var rgbVis = {min:0.0, max:3000, bands:['B4','B3','B2']};
Map.addLayer(image, rgbVis, 'Image');

/*
Index Calculation 
*/

// Normalized Difference Vegetation Index (NDVI)
var ndvi = image.normalizedDifference(['B8', 'B4']);
var ndviVis = {
  min: 0.0, 
  max: 0.8, 
  palette:['white', 'green']
};
  
  
Map.addLayer(ndvi, ndviVis, "NDVI")

// Modify Normalized Water Index (MNDWI)
// 'Green' (B3) and 'SWIRI' (B11)
var mndwi = image.normalizedDifference(['B3', 'B11']);
var ndwiVis = {
  min: 0, 
  max: 0.8, 
  palette: ['white', 'blue']
};
Map.addLayer(mndwi, ndwiVis, "MNDWI");

// Time series 

// MapReduce NVDI Calculation 
var ndviCollection = filteredS2.map(function(img) {
  return img.normalizedDifference(['B8', 'B4'])
            .rename('NDVI')
            .copyProperties(img, ['system:time_start']);
});

// Save chart as variable
var chart = ui.Chart.image.series({
  imageCollection: ndviCollection,
  region: geometry,
  reducer: ee.Reducer.mean(),
  scale:10,
  }
  );
  
print(chart);