/*

Working with Google Earth Dasta Sets and Manipulation 
Clipping and saving (Exporting)

*/


// Filter for San Francisco

var filtered = admin2
  .filter(ee.Filter.eq('ADM1_NAME', 'Mexico City'));

Map.addLayer(admin2,{} , 'Admin2');
Map.addLayer(filtered, {color:'red'}, 'Selected');


// Clipping data inside CDMX 

// Getting city boundaries 
var filteredAdmin2 = admin2
  .filter(ee.Filter.eq('ADM1_NAME', 'Distrito Federal'));

// Geometry 
var geometry = filteredAdmin2.geometry();

// Data Layers 
var filteredS2 = s2
  .filter(ee.Filter.date('2019-01-01', '2020-01-01'))
  .filter(ee.Filter.bounds(geometry));
  
var image = filteredS2.median();

var clipped = image.clip(geometry);

var visRGB = {
  min: 0.0,
  max: 3000.0,
  bands: ['B4','B3','B2']
};

Map.centerObject(geometry);
Map.addLayer(clipped, visRGB, "Clipped Composit");


/*

EXPORTING DATA LAYERS

*/

// Selecting bands of Interest 

var exportImage = clipped.select(['B4','B3','B2']);

print(exportImage);


// Export Image

// Auto-Complete 
// Ctrl + space

// Export minimal parameters 
Export.image.toDrive({
  image: exportImage,
  description : "Composite_Export",
  folder: "earthengine",
  fileNamePrefix: "Composite", 
  region: geometry, // Hard bounds for exporting 
  scale:100, // Always in metters
  crs:"EPSG:6369", //  Mexico City CRS 
  maxPixels:1e10 , // Stop Exporting at 
}
  );
