# Printable Map from OsmAnd

Generate a printable 1:25000 topographic map poster (SVG/A4) from OsmAnd (https://osmand.net) data. The map uses the **true UTM grid** (rotated by the grid convergence, kilometre intersections matching GPS) and carries declination information for compass work.

Rendering is performed exclusively through the **native OsmAnd engine** (`EyePiece`, the same C++/Skia renderer the phone app uses), driven off-screen. The engine exports a `.georef.json` (true lat/lon of the raster corners), and the SVG grid is drawn from that georeference — so grid intersections always land on the coordinates they label, independent of any renderer scale quirks.

## Repository layout

This is the **single repo to clone**. The official OsmAnd projects are included
as submodules at their **official upstream commits**; our print-rendering
changes ride along as a patch series that the build script applies:

```
.                       # clone this one repo
├── topo_map_generator.py   # CLI entry (native core rendering)
├── obf/ schemas/           # OBF coverage filtering and UTM helpers
├── styles/                 # print render styles + hillshade palettes
├── patches/                # our modifications, as git-format-patch series
│   ├── core/                   #   EyePiece georef export, hillshade, grids...
│   ├── tools/                  #   headless build tweaks
│   └── OsmAnd-resources/       #   DEPFILE fix
├── scripts/
│   ├── build-eyepiece.sh   # apply patches + configure + build (idempotent)
│   └── install.sh          # editable pip install + eyepiece onto PATH
└── osmand-native/          # official sources, as submodules
    ├── CMakeLists.txt          # superbuild configuration (headless)
    ├── core/                   # osmandapp/OsmAnd-core   @ f3b581fe (official)
    ├── tools/                  # osmandapp/OsmAnd-tools  @ 2429969df (official)
    ├── OsmAnd-resources/       # osmandapp/OsmAnd-resources @ e99f7f4e (official)
    └── core-legacy/            # osmandapp/OsmAnd-core-legacy
```

The `patches/` series covers:

- raster georeference export (`<image>.georef.json`, true corner lat/lon)
- hillshade: `-slopeColors` blend fix, OsmAnd standard palettes, no 2× vertical exaggeration
- `-primaryGrid=utm|wgs84` overlay (self-check of grid placement)
- 3D relief off unless `-elevation` (displacement warps the raster)
- cold-start frame capture fix (idle hold instead of grabbing stub tiles)

After changing anything under `osmand-native/`, re-export the series with
`git format-patch <official-base>..HEAD -o patches/<repo>` in the submodule and
commit the updated `patches/` here.

## Installation

Tested on Python 3.11+ (Debian/Ubuntu).

```shell
git clone --recurse-submodules git@github.com:acui/osmand_topo_map_generator.git topo-map
cd topo-map

# 1. Python package (editable install; console entry point `topo-map-generator`)
pip install -e .

# 2. EyePiece rasterizer (one-time build; needs build-essential cmake ninja
#    libgdal-dev libgl1-mesa-dev libegl1-mesa-dev qtbase5-dev)
scripts/build-eyepiece.sh

# 3. Put the binary on PATH (optional; the generator also finds it in the build tree)
scripts/install.sh
```

Data note: OBF files / GeoTIFF heightmaps live outside the repo (default
`--obf_dir ~/osmand-data`, EyePiece expects `<obf_dir>/geotiff` for `--hillshade`).

### Getting the .obf data from Waydroid

The OsmAnd data files live inside the Waydroid Android storage:

```shell
# find the data directory (one of the two usually exists)
sudo ls /var/lib/waydroid/data/media/0/Android/data/net.osmand.plus/files/
ls ~/.local/share/waydroid/data 2>/dev/null

# copy the .obf files out (map + srtm contour files)
mkdir -p ~/osmand-data
sudo cp /var/lib/waydroid/data/media/0/Android/data/net.osmand.plus/files/*.obf ~/osmand-data/
sudo cp /var/lib/waydroid/data/media/0/Android/data/net.osmand.plus/files/srtm/*.obf ~/osmand-data/srtm/ 2>/dev/null
sudo chown -R $USER:$USER ~/osmand-data
```

If `/var/lib/waydroid/data` does not exist, your Waydroid keeps userdata in an ext4 image — check `grep mount.entry /var/lib/waydroid/lxc/waydroid/config` and mount it read-only:

```shell
sudo mkdir -p /mnt/wd && sudo mount -o ro,loop <image path> /mnt/wd
sudo find /mnt/wd -name "*.obf"
```

## Usage

```shell
topo-map-generator 七王坟-凉水背 50T 420500 4437185 1:25000 topomaps/七王坟-凉水背 \
    --hillshade True --hillshade_opacity 0.39 \
    --dpi 1200 --orientation landscape \
    --render_style ~/osmand-data/rendering/Hiking.render.xml
```

Result: `<name>.svg` (A4 poster, rotated UTM grid, km labels) plus `<name>_core.png`
(the north-up georeferenced basemap) and `<name>_core.png.georef.json` (corner
coordinates) in the output directory.

### Resolution and supersampling

Physical sizes of roads/labels/icons are pinned by `--style_dpi` (default 300).
`--dpi` is pure output resolution: `density = dpi/style_dpi` drives EyePiece's
`-displayDensityFactor` (all style sizes scale up) and `-referenceTileSize`
(vector tiles stay crisp), so a 1200 dpi print looks exactly like the 300 dpi
proof, only sharper. The raster automatically re-renders until it covers the
rotated grid frame.

### Hillshade

`--hillshade True` enables the relief layer (OsmAnd standard palettes,
`color-palette/hillshade_main_default.txt` converted in `styles/`).
`--hillshade_opacity 0.39` is the OsmAnd default visibility. 3D relief
(displacement) stays off — it warps the raster geometry.

Notes:
- Contour lines: for China, OsmAnd's Beijing-area contours ship inside
  `srtm/China_hebei_asia.srtm.obf` (the standalone `srtmf` file crashes the
  engine when mixed with map files — currently filtered by the bbox pre-check).
- The engine occasionally crashes during shutdown, after the image has been
  written; the pipeline tolerates this.

### Useful options

- `--obf_dir ~/osmand-data` — directory with the `.obf` files (map files at the top level, contour files in `srtm/` or `srtmf/` sub-directories)
- `--map_language {local,user,bilingual}` — label language; `--user_lang zh` selects the user language (default `zh`)
- `--contour_lines {disabled,16,15,14,13,12,11}` — minimum zoom to draw contour lines
- `--contour_width {thin,medium,thick}`, `--contour_color_scheme brown` — contour appearance
- `--render_style` — custom rendering style
- `--paper_size A4 --orientation portrait|landscape`
- `--eyepiece PATH` — alternative EyePiece binary

## Known Issues

- Can not generate a map when the equator is in the region.
- Metric units only.

## License

The source in this repository falls under an MIT license (see LICENSE).
OsmAnd core/tools (the `osmand-native` submodule) keep their upstream licenses.
