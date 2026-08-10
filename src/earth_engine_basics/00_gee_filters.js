

/*

Working with Google Earth Dasta Sets and Manipulation 
Filters and Calculations 

*/


/*
FILTERS 
*/

// Date Filter 
// Images from 2019
var filtered = s2.filter(
  ee.Filter.date('2019-01-01', '2020-01-01')
  );


// Location Filter 
// All images collected on my AOI
var filtered2 = filtered.filter(
  ee.Filter.bounds(geometry)
  );


// Properties Filter 
// All images having cloud cover < 30%
// Select all images having CLOUDY_PIXEL_PERCENTAGE < 30
var filtered3 = filtered2.filter(
  ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',30)
  );
  

print(filtered3.size());

// Chaining 
var filtered = s2
  .filter(ee.Filter.date('2019-01-01', '2020-01-01'))
  .filter(ee.Filter.bounds(geometry))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',30));
  
print(filtered.size());


// Composite 
var composite = filtered.median();


// Viz Params 
var visParams = {
  min:0, 
  max:3000, 
  bands:['B4','B3','B2']
};

// For adding Imagesd/Layers to the map 
Map.addLayer(filtered, visParams, 'Filtered Collection');

Map.addLayer(composite, visParams, 'Composite');

