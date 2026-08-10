 

 /*

 The present code helos downloading full embeddings for a given yeat and 
 a city level region at maximum 10m metters. 

 It is done in this way becaus eof possible data downloading caps from google Earth Engine. 

 */
 
 
 // 1. Core Configuration
var YEAR = 2025;
var startDate = YEAR + '-01-01';
var endDate = (YEAR + 1) + '-01-01';
var mun="006";

// 2. Load your uploaded INEGI Asset
var inegiMunicipal = ee.FeatureCollection("<your-asset-project>");

// Filter strictly down to La Magdalena Contreras using official INEGI keys
// State/Entidad = '09' (CDMX), Municipality = '008' (Magdalena Contreras)
var lmcPartition = inegiMunicipal
    .filter(ee.Filter.eq('CVE_ENT', '09'))
    .filter(ee.Filter.eq('CVE_MUN', mun));

Map.centerObject(lmcPartition, 13);

// 3. Set Up the Native Embedding Mosaic
var embeddingsCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(startDate, endDate)
    .filterBounds(lmcPartition.geometry());

var nativeTile = embeddingsCol.first();
var nativeProjection = nativeTile.select(0).projection();
var mosaicImage = embeddingsCol.mosaic().setDefaultProjection(nativeProjection);

// 4. Visualize Layers on the Map Canvas using correct AlphaEarth codes
Map.addLayer(mosaicImage.clip(lmcPartition.geometry()), {
  bands: ['A00', 'A01', 'A02'], 
  min: -1, 
  max: 1
}, 'INEGI LMC Embeddings');

// Render the outline of your INEGI municipal boundary asset
Map.addLayer(lmcPartition.draw({color: 'FF0000', strokeWidth: 2}), {}, 'INEGI LMC Boundary');

// 5. Export Pipeline for the full La Magdalena Contreras municipality
lmcPartition.evaluate(function(featureCollection) {
  var features = featureCollection.features;
  
  if (features.length === 0) {
    print('Error: No feature found matching CVE_ENT=09 and CVE_MUN=008. Check your asset attributes!');
    return;
  }
  
  var feature = features[0];
  var props = feature.properties;
  
  var geometry = ee.Feature(feature).geometry();
  
  var clippedImage = mosaicImage.clip(geometry);
  
  Export.image.toDrive({
      image: clippedImage,
      description: 'mun_' + mun + '_' + YEAR,
      folder: 'satellite_embeddings',
      scale: 10,
      region: geometry,
      maxPixels: 1e13,
      crs: 'EPSG:4326', //crs: nativeProjection.crs(),
  });
  
  
  print('Export task created for mun ' + mun + '. Head over to the Tasks tab on the right to run it!');
}); 
