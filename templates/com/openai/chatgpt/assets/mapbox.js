const queryString = window.location.search;
const params = new URLSearchParams(queryString);

mapboxgl.accessToken = 'pk.eyJ1Ijoib2FpLWRhdGEiLCJhIjoiY20yYW84aTZ1MGh6bzJwcHpsY2R6OWxyeiJ9.V2VlnBphK96gruxcrHtt-Q';

const defaultPadding = 25;
const mapConfig = {
  container: 'map',
  style: 'mapbox://styles/mapbox/streets-v11',
  interactive: params.get('interactive') === 'true',
};

// If the location provided, set it as the map center
const latitude = params.get('latitude');
const longitude = params.get('longitude');
if (latitude && longitude) {
  mapConfig.center = [longitude, latitude];
  mapConfig.zoom = parseFloat(params.get('zoom'));
}

const map = new mapboxgl.Map(mapConfig);

function updateCamera(longitude, latitude, zoom) {
  map.flyTo({
    center: [longitude, latitude],
    zoom: zoom,
  });
}

function updateMarkers(markers) {
  document.querySelectorAll('.mapboxgl-marker')
    .forEach(marker => marker.remove());

  markers.forEach(marker => {
    const markerElement = document.createElement('div');
    markerElement.className = 'custom-marker';
    markerElement.dataset.id = marker.id;
    if (marker.rating) {
      markerElement.innerHTML = `${marker.rating.toFixed(1)} <span class="star">★</span>`;
    }

    markerElement.addEventListener('click', () => {
      if (window.AndroidInterface && window.AndroidInterface.notifyMarkerSelect) {
        window.AndroidInterface.notifyMarkerSelect(markerElement.dataset.id);
      }
    });

    new mapboxgl.Marker(markerElement)
      .setLngLat([marker.longitude, marker.latitude])
      .addTo(map);
  });
}

const markers = JSON.parse(decodeURIComponent(params.get('markers')));
updateMarkers(markers);

// If no location provided, fit the map to the markers
if (!latitude || !longitude) {
  const bounds = new mapboxgl.LngLatBounds();
  markers.forEach(marker => bounds.extend([marker.longitude, marker.latitude]));
  map.fitBounds(bounds, {
    animate: false,
    padding: {
      top: defaultPadding,
      bottom: defaultPadding + parseInt(params.get('bottomPadding')),
      left: defaultPadding,
      right: defaultPadding,
    },
  });
}
