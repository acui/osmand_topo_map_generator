# Generate reference values for the OsmAnd TopoPrint plugin's math unit tests
# (TopoPrintMathTest.kt in ~/projects/OsmAnd). Run from the repo root:
#
#   python3 tests/reference/generate_reference.py
#
# Uses the same libraries as the CLI pipeline (utm, pygeomag/WMMHR2025), so the
# printed JSON is the ground truth the Kotlin port must reproduce within
# tolerance (UTMPoint in OsmAnd converts via float LatLonPoint, hence ~1 m
# slack; GeomagneticField is WMM vs. WMMHR2025, hence 0.5 deg slack).

import json
import math

import utm
from pygeomag import GeoMag

# (name, lat, lon) — covers normal zones, the Norway and Svalbard specials,
# the southern hemisphere and a zone-boundary neighborhood.
POINTS = [
    ("beijing", 40.066500, 116.183000),
    ("sydney", -33.868800, 151.209300),
    ("bergen_norway", 60.390000, 5.320000),
    ("svalbard", 78.220000, 15.630000),
    ("equator_zone", 0.500000, 120.000000),
    ("new_york", 40.712800, -74.006000),
]

DECLINATION_TIME = 2026.0  # decimal year, matching the plugin's test epoch


def grid_convergence(lat, lon, zone_number):
    central_meridian = (zone_number - 1) * 6.0 - 180.0 + 3.0
    d_lon = math.radians(lon - central_meridian)
    return math.degrees(math.atan(math.tan(d_lon) * math.sin(math.radians(lat))))


def main():
    geomag = GeoMag()
    reference = []
    for name, lat, lon in POINTS:
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        back_lat, back_lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        result = geomag.calculate(glat=lat, glon=lon, alt=0.0, time=DECLINATION_TIME)
        reference.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "easting": easting,
            "northing": northing,
            "zone_number": zone_number,
            "zone_letter": zone_letter,
            "roundtrip_lat": back_lat,
            "roundtrip_lon": back_lon,
            "convergence_deg": grid_convergence(lat, lon, zone_number),
            "declination_deg": result.d,
        })
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
