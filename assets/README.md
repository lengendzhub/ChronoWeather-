# Map boundary data

`ne_110m_land.geojson` is the Natural Earth 1:110m land layer. It is used only
as a light geographic backdrop for the Phase F city maps; no analytical values
come from this file.

- **Source:** <https://github.com/nvkelso/natural-earth-vector>
- **Layer:** `geojson/ne_110m_land.geojson`
- **Status:** public domain (Natural Earth data are free of copyright
  restrictions)
- **Downloaded:** 2026-11-13
- **Size:** about 138 KB

The project does not require a geo/GIS Python package for the maps. The GeoJSON
is parsed with the standard library and drawn as an equirectangular backdrop in
Matplotlib.
