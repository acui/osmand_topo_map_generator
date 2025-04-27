import cv2 as cv
import numpy as np
from pathlib import Path
from fractions import Fraction
import pymap3d.enu as enu
import math
from typing import Literal
import datetime
import pygeomag
import svg
import base64
from tempfile import TemporaryDirectory
from tqdm import tqdm
from lxml import etree
from schemas.layout import Layout
from schemas.utm import UTMZoneLetter, UTMCoordinate, UTMBoundingBox

GPX_TEMPLATE = Path(__file__).parent / "template.gpx"
WMM_COEFFICIENTS_FILE = Path(__file__).parent / "wmm" / "WMMHR_2025.COF"
FONT_FAMILY = "Noto Sans, sans-serif"

layouts = {
    "A4": { 
        "portrait": Layout(
            paper_size=np.array([210.0, 297.0]),
            map_size=np.array([175.0, 210.0])
        ),
        "landscape": Layout(
            paper_size=np.array([297.0, 210.0]),
            map_size=np.array([252.0, 123.0])
        )
    }
}
    
def crop_screenshot(image):
    # Crop the image to remove the top and bottom borders
    # Working with BlueStacks 5
    height, width = image.shape[:2]
    return image[48:height-10, 0:width]

def match_template(img, template, method=cv.TM_SQDIFF_NORMED):
    h, w = template.shape  # Get the width and height of the template image
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take the minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)  # Calculate the bottom right corner of the rectangle
    difference = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].astype(np.int32) - template  # Calculate the difference between the matched area and the template
    return(top_left, bottom_right, np.mean(difference))  # Return the coordinates of the rectangle

def cluster_2d_points(points: np.ndarray, radius: float) -> list[list[np.ndarray]]:
    clusters = []
    visited = [False] * len(points)

    for i in range(len(points)):
        if visited[i]:
            continue

        # Start a new cluster
        cluster = [points[i]]
        visited[i] = True

        # Check for all points to see if they are within the radius
        for j in range(i + 1, len(points)):
            if not visited[j]:
                distance = np.linalg.norm(points[i][:2] - points[j][:2])
                if distance <= radius:
                    cluster.append(points[j])
                    visited[j] = True

        clusters.append(cluster)
    return sorted(clusters, key=lambda c: len(c), reverse=True)

def match_image(img1, img2, match_size):
    # match the top border of img2
    gimg1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gimg2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    h, w, _ = img2.shape
    translations = []
    translations.append([[], 0])
    for i in range(0, w, match_size):
        top_left, bottom_right, diff = match_template(gimg1, gimg2[0:match_size, i:i+match_size])
        if abs(diff) < 1.0:
            translations[-1][0].append((top_left[0] - i, top_left[1], abs(diff)))
    translations[-1][1] = len(translations[-1][0]) / (w // match_size)
    translations.append([[], 0])
    for i in range(0, w, match_size):
        top_left, bottom_right, diff = match_template(gimg1, gimg2[h-match_size:h, i:i+match_size])
        if abs(diff) < 1.0:
            translations[-1][0].append((top_left[0] - i, top_left[1] - h + match_size, abs(diff)))
    translations[-1][1] = len(translations[-1][0]) / (w // match_size)
    translations.append([[], 0])
    for i in range(0, h, match_size):
        top_left, bottom_right, diff = match_template(gimg1, gimg2[i:i+match_size, 0:match_size])
        if abs(diff) < 1.0:
            translations[-1][0].append((top_left[0], top_left[1] - i, abs(diff)))
    translations[-1][1] = len(translations[-1][0]) / (h // match_size)
    translations.append([[], 0])
    for i in range(0, h, match_size):
        top_left, bottom_right, diff = match_template(gimg1, gimg2[i:i+match_size, w-match_size:w])
        if abs(diff) < 1.0:
            translations[-1][0].append((top_left[0] - w + match_size, top_left[1] - i, abs(diff)))
    translations[-1][1] = len(translations[-1][0]) / (h // match_size)

    translations.sort(key=lambda x: x[1], reverse=True)
    result = np.array(translations[0][0])
    result = cluster_2d_points(result, radius=2.0)[0]
    result.sort(key=lambda x: x[2])
    return result

def draw_star_path(radius: float) -> list[svg.PathData]:
    inner_radius = radius * (3 - math.sqrt(5)) / 2
    result = [svg.M(radius, 0)]
    for angle in range(36, 360, 72):
        result.append(svg.L(inner_radius * math.cos(math.radians(angle)), inner_radius * math.sin(math.radians(angle))))
        result.append(svg.L(radius * math.cos(math.radians(angle + 36)), radius * math.sin(math.radians(angle + 36))))
    result.append(svg.Z())

def read_map(map_root: Path) -> np.ndarray:
    images = list(map_root.glob("Screenshot_*.png"))
    images.sort()
    imgs = [crop_screenshot(cv.imread(str(image))) for image in images]
    regions = [np.array([0, 0, imgs[0].shape[1], imgs[0].shape[0]])]
    for i in tqdm(range(1, len(imgs))):
        img1 = imgs[i-1]
        img2 = imgs[i]
        h, w = img2.shape[:2]
        translation = regions[i-1][:2] + match_image(img1, img2, 128)[0][:2].astype(np.int32)
        regions.append(np.array([translation[0], translation[1], translation[0] + w, translation[1] + h]))
    regions = np.array(regions)
    left, top, right, bottom = regions[:, 0].min(), regions[:, 1].min(), regions[:, 2].max(), regions[:, 3].max()
    canvas = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    for i in range(len(imgs)):
        img = imgs[i]
        region = regions[i]
        canvas[region[1]-top:region[3]-top, region[0]-left:region[2]-left] = np.concatenate((img, np.full((img.shape[0], img.shape[1], 1), 255)), axis=2)  # Add alpha channel with full opacity
    return canvas

def get_corners(canvas: cv.typing.MatLike) -> list[np.ndarray]:
    canvas_bgr = cv.cvtColor(canvas, cv.COLOR_BGRA2BGR)

    hsv = cv.cvtColor(canvas_bgr, cv.COLOR_BGR2HSV)

    lower_magenta = np.array([150, 160, 160])
    upper_magenta = np.array([170, 255, 255])
    mask = cv.inRange(hsv, lower_magenta, upper_magenta)
    dst = cv.cornerHarris(mask, 4, 5, 0.04)
    ys, xs = np.where(dst > 0.5 * dst.max())
    positions = list(zip(xs, ys))
    top_left = np.array(min(positions, key=lambda p: p[0] ** 2 + p[1] ** 2))
    top_right = np.array(min(positions, key=lambda p: (p[0] - canvas.shape[1]) ** 2 + p[1] ** 2))
    bottom_left = np.array(min(positions, key=lambda p: (p[1] - canvas.shape[0]) ** 2 + p[0] ** 2))
    bottom_right = np.array(min(positions, key=lambda p: ((p[0] - canvas.shape[1]) ** 2 + (p[1] - canvas.shape[0]) ** 2)))

    return [top_left, top_right, bottom_right, bottom_left]

class TopoMap:
    def __init__(self, layout: Layout, label: str, center: UTMCoordinate, scale: Fraction = Fraction(1, 25000)):
        """
        Initialize the TopoMapGenerator with a center point and grid size.
        
        :param label: Label for the map, typically the name of the area or location.
        :param center: UTMCoordinate object representing the center of the map.
        :param grid_size: Size of the grid in meters.
        :param scale: Scale of the map, default is 1:25000. It should be a Fraction object.
        """
        assert(scale > 0), "Scale must be a positive value."
        assert(scale <= Fraction(1, 1000)), "Scale must be less than or equal to 1:1000 for practical map generation."
        self.layout = layout
        self.label = label
        map_size =  layout.map_size * 1.1 / scale / 1000
        bottom = center.northing - int(map_size[1] / 2)
        top = int(math.ceil(bottom + map_size[1]))
        left = center.easting - int(map_size[0] / 2)
        right = int(math.ceil(left + map_size[0]))
        self.boundingbox = UTMBoundingBox(
            zone_number=center.zone_number, 
            zone_letter=center.zone_letter, 
            left=left, 
            top=top, 
            right=right, 
            bottom=bottom)
        self.gridBoundingBox = UTMBoundingBox(
            zone_number=center.zone_number, 
            zone_letter=center.zone_letter, 
            left=int(math.floor(left / 1000)) * 1000,  # Extend the bounding box by 1 km to include grid lines
            top=int(math.ceil(top / 1000)) * 1000,
            right=int(math.ceil(right / 1000)) * 1000,
            bottom=int(math.floor(bottom / 1000)) * 1000
        )
        self.grid_size = ((self.gridBoundingBox.right - self.gridBoundingBox.left) // 1000, (self.gridBoundingBox.top - self.gridBoundingBox.bottom) // 1000)
        self.center = center
        map_center_lat, map_center_lon = self.boundingbox.center.to_latlon()
        map_top_center_lat, map_top_center_lon = self.boundingbox.top_center.to_latlon()
        enu_top_center = enu.geodetic2enu(map_top_center_lat, map_top_center_lon, 0, map_center_lat, map_center_lon, 0)
        self.tn2gn = math.atan2(enu_top_center[0], enu_top_center[1])
        self.create_time = datetime.datetime.now()
        decimal_year = pygeomag.calculate_decimal_year(self.create_time)
        geo_mag = pygeomag.GeoMag(coefficients_file=str(WMM_COEFFICIENTS_FILE.relative_to(Path().absolute())), high_resolution=True)
        result = geo_mag.calculate(map_center_lat, map_center_lon, 0, decimal_year)
        self.declination = result.d
        self.scale = scale

    def create_boundary_gpx(self, gpx_file: Path) -> None:
        tree = etree.XML(GPX_TEMPLATE.read_bytes())
        extensions = tree.getchildren()[0]
        tree.remove(extensions)
        metadata = etree.Element("metadata")
        meta_name = etree.Element("name")
        meta_name.text = f"Map Boundaries of {self.label}"
        metadata.append(meta_name)
        meta_time = etree.Element("time")
        meta_time.text = self.create_time.isoformat(timespec="milliseconds" if self.create_time.microsecond else "seconds").replace("+00:00", "Z")
        metadata.append(meta_time)
        tree.append(metadata)
        points = [self.boundingbox.top_left.to_latlon(), self.boundingbox.top_right.to_latlon(), self.boundingbox.bottom_right.to_latlon(), self.boundingbox.bottom_left.to_latlon(), self.boundingbox.top_left.to_latlon()]
        track = etree.Element("trk")
        trk_name = etree.Element("name")
        trk_name.text = f"Map Boundaries of {self.label}"
        track.append(trk_name)
        trackseg = etree.Element("trkseg")
        for point in points:
            trkpt = etree.Element("trkpt", {"lat": f"{point[0]:.6f}", "lon": f"{point[1]:.6f}"})
            trackseg.append(trkpt)
        trkpt = etree.Element("trkpt", {"lat": f"{points[0][0]:.6f}", "lon": f"{points[0][1]:.6f}"})
        trackseg.append(trkpt)
        track.append(trackseg)
        tree.append(track)
        tree.append(extensions)
        gpx_file.write_bytes(etree.tostring(tree, encoding="utf-8", xml_declaration=True, pretty_print=True))

    def add_map(self, canvas: np.ndarray, corners: list[np.ndarray]) -> None:
        image_width, image_height = canvas.shape[1], canvas.shape[0]
        rotated_map_width = self.grid_size[0] * 1000 * 1000 * self.scale * abs(math.cos(self.tn2gn)) + self.grid_size[1] * 1000 * 1000 * self.scale * abs(math.sin(self.tn2gn))
        rotated_map_height = rotated_map_width * image_height / image_width
        self.rotated_map_size = np.array([rotated_map_width, rotated_map_height])
        self.corners = [corner.astype(np.float32) / np.array([image_width, image_height]) * self.rotated_map_size for corner in corners]
        with TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir) / "temp_map.png"
            cv.imwrite(str(file_name), canvas)
            encoded_image = base64.b64encode(file_name.read_bytes()).decode('utf-8')
            self.image_data = f"data:image/png;base64,{encoded_image}"
            

    def draw_compass(self, tn2gn:float, tn2mn:float) -> svg.G:
        compass_size = self.layout.bottom_margin * 0.6
        line_length = compass_size * 0.85
        tn2gn_rad = math.radians(tn2gn)
        tn2mn_rad = math.radians(tn2mn)
        star = svg.Path(
            d = draw_star_path((compass_size - line_length) * 0.2),
            fill="#000000",
            transform=[
                svg.Translate(0, compass_size - line_length),
                svg.Rotate(-90, 0, 0),
            ]
        )

        label_text_size = compass_size / 15.0
        label_text_margin = label_text_size / 2.0
        angle_text_size = compass_size / 20.0
        angle_text_margin = angle_text_size / 2.0


        tn = svg.Path(
            d = [
                svg.M(0, compass_size),
                svg.L(0, compass_size - line_length),
            ],
            stroke="#000000",
            stroke_width=0.2
        )
        tn_text = svg.Text(
            fill= "#000000",
            stroke= "#000000",
            stroke_width=0.1,
            font_family=FONT_FAMILY,
            font_size=label_text_size,
            text="TN",
            x = label_text_margin,
            y = compass_size - line_length + label_text_margin
        )
        mn_end_x = line_length * math.sin(tn2mn_rad)
        mn_end_y = compass_size - line_length * math.cos(tn2mn_rad)
        mn = svg.Path(
            d=[
                svg.M(0, compass_size),
                svg.L(mn_end_x, mn_end_y)
            ],
            stroke="#000000",
            stroke_width=0.1
        )
        if mn_end_x < 0:
            mn_text_x = mn_end_x - label_text_margin
            mn_text_anchor = "end"
        else:
            mn_text_x = mn_end_x + label_text_margin
            mn_text_anchor = "start"
        mn_text = svg.Text(
            fill= "#000000",
            stroke= "#000000",
            stroke_width=0.1,
            font_family=FONT_FAMILY,
            font_size=label_text_size,
            text="MN",
            text_anchor= mn_text_anchor,
            x = mn_text_x,
            y = mn_end_y + label_text_size * 1.5
        )
        mn_angle = f"{tn2mn:.2f}°"
        if tn2mn < 0:
            mn_angle_x = line_length * 0.5 * math.sin(tn2mn_rad) - angle_text_margin
            mn_angle_anchor = "end"
        else:
            mn_angle_x = line_length * 0.5 * math.sin(tn2mn_rad) + angle_text_margin
            mn_angle_anchor = "start"
        mn_angle_y = compass_size - line_length * 0.5 * math.cos(tn2mn_rad) - angle_text_margin
        mn_angle_text = svg.Text(
            fill= "#000000",
            stroke= "#000000",
            stroke_width=0.1,
            font_family=FONT_FAMILY,
            font_size=angle_text_size,
            text=mn_angle,
            text_anchor=mn_angle_anchor,
            x = mn_angle_x,
            y = mn_angle_y
        )
        
        gn_end_x = line_length * 0.75 * math.sin(tn2gn_rad)
        gn_end_y = compass_size - line_length * 0.75 * math.cos(tn2gn_rad)
        gn = svg.Path(
            d=[
                svg.M(0, compass_size),
                svg.L(gn_end_x, gn_end_y)
            ],
            stroke="#000000",
            stroke_width=0.1
        )
        if gn_end_x < 0:
            gn_text_x = gn_end_x - label_text_margin
            gn_text_anchor = "end"
        else:
            gn_text_x = gn_end_x + label_text_margin
            gn_text_anchor = "start"
        gn_text = svg.Text(
            fill= "#000000",
            stroke= "#000000",
            stroke_width=0.1,
            font_family=FONT_FAMILY,
            font_size=label_text_size,
            text="GN",
            text_anchor=gn_text_anchor,
            x = gn_text_x,
            y = gn_end_y + label_text_margin
        )
        gn_angle = f"{tn2gn:.2f}°"
        if tn2gn < 0:
            gn_angle_x = line_length * 0.375 * math.sin(tn2gn_rad) - angle_text_margin
            gn_angle_anchor = "end"
        else:
            gn_angle_x = line_length * 0.375 * math.sin(tn2gn_rad) + angle_text_margin
            gn_angle_anchor = "start"
        gn_angle_y = compass_size - line_length * 0.375 * math.cos(tn2gn_rad) - angle_text_margin
        gn_angle_text = svg.Text(
            fill= "#000000",
            stroke= "#000000",
            stroke_width=0.1,
            font_family=FONT_FAMILY,
            font_size=angle_text_size,
            text=gn_angle,
            text_anchor=gn_angle_anchor,
            x = gn_angle_x,
            y = gn_angle_y
        )

        tn_mn_arc = svg.Path(
            d=[
                svg.M(0, compass_size - line_length*0.85),
                svg.Arc(line_length*0.85, line_length*0.85, 0, False, False, line_length * 0.85 * math.sin(tn2mn_rad), compass_size - line_length * 0.85 * math.cos(tn2mn_rad)),
            ],
            fill="none",
            stroke="#000000",
            stroke_width=0.1,
            stroke_dasharray=[0.1, 0.8],
            stroke_dashoffset=0,
            id="tn_mn_arc"  # Add an ID for the arc to be referenced if needed
        )
        tn_gn_arc = svg.Path(
            d=[
                svg.M(0, compass_size - line_length * 0.75),
                svg.Arc(line_length * 0.75, line_length * 0.75, 0, False, False, gn_end_x, gn_end_y)
            ],
            fill="none",
            stroke="#000000",
            stroke_width=0.1,
            stroke_dasharray=[0.1, 0.8],
            stroke_dashoffset=0,
            id="tn_gn_arc"  # Add an ID for the arc to be referenced if needed
        )
        compass_group = svg.G(
            elements = [
                star,
                tn,
                tn_text,
                mn,
                mn_text,
                mn_angle_text,
                gn,
                gn_text,
                gn_angle_text,
                tn_mn_arc,
                tn_gn_arc
            ],
            transform_origin="right top",
            transform=[
                svg.Translate(self.layout.paper_size[0] - self.layout.right_margin - compass_size * 0.5, self.layout.paper_size[1] - self.layout.bottom_margin + (self.layout.bottom_margin - compass_size) / 4)
            ],
            id="compass"
        )
        return compass_group

    def draw_grid(self) -> svg.G:
        horizontal_lines = []
        top_left = self.corners[0]
        top_right = self.corners[1]
        bottom_left = self.corners[3]
        bottom_right = self.corners[2]
        for i in range(self.grid_size[1] + 1):
            start = top_left + (bottom_left - top_left) * i / self.grid_size[1]
            end = top_right + (bottom_right - top_right) * i / self.grid_size[1]
            horizontal_lines.append(
                svg.Path(
                    d=[
                        svg.M(start[0], start[1]),
                        svg.L(end[0], end[1])
                    ],
                    stroke="#000000",
                    stroke_width=0.1,
                )
            )
        vertical_lines = []
        for i in range(self.grid_size[0] + 1):
            start = top_left + (top_right - top_left) * i / self.grid_size[0]
            end = bottom_left + (bottom_right - bottom_left) * i / self.grid_size[0]
            vertical_lines.append(
                svg.Path(
                    d=[
                        svg.M(start[0], start[1]),
                        svg.L(end[0], end[1])
                    ],
                    stroke="#000000",
                    stroke_width=0.1,
                )
            )
        grid_group = svg.G(
            elements=horizontal_lines + vertical_lines,
            id="grid",
        )
        return grid_group
    
    def draw_frame(self) -> tuple[svg.ClipPath, svg.G]:
        frame = svg.Rect(
            stroke="#000000",
            stroke_width=1.0,
            fill="none",
            width=self.layout.map_size[0],
            height=self.layout.map_size[1],
        )
        clip=svg.ClipPath(
            elements=[
                frame
            ],
            id="frame_clip"
        )
        utm_center2map_center_northing = self.boundingbox.center.northing - self.gridBoundingBox.center.northing
        utm_center2map_center_easting = self.boundingbox.center.easting - self.gridBoundingBox.center.easting
        enu_center2map_center_northing = utm_center2map_center_northing * math.cos(self.tn2gn) - utm_center2map_center_easting * math.sin(self.tn2gn)
        enu_center2map_center_easting = utm_center2map_center_northing * math.sin(self.tn2gn) + utm_center2map_center_easting * math.cos(self.tn2gn)
        grid_translation_x = self.layout.map_size[0] / 2  - self.rotated_map_size[0] / 2 - enu_center2map_center_easting * 1000 * self.scale
        grid_translation_y = self.layout.map_size[1] / 2 - self.rotated_map_size[1] / 2 + enu_center2map_center_northing * 1000 * self.scale
        label_text_size = self.layout.left_margin / 5
        label_prefix_size = label_text_size * 0.8
        top_label_y = -label_text_size * 0.8
        bottom_label_y = self.layout.map_size[1] + label_text_size * 1.5
        left_label_x = -label_text_size * 0.8
        right_label_x = self.layout.map_size[0] + label_text_size * 0.8
        grid_labels = []
        top_left = self.corners[0]
        top_right = self.corners[1]
        bottom_left = self.corners[3]
        bottom_right = self.corners[2]
        grid_translation = np.array([grid_translation_x, grid_translation_y])
        grid_east = self.gridBoundingBox.left // 1000
        grid_north = self.gridBoundingBox.top // 1000
        for i in range(self.grid_size[0] + 1):
            start = top_left + (top_right - top_left) * i / self.grid_size[0] + grid_translation
            end = bottom_left + (bottom_right - bottom_left) * i / self.grid_size[0] + grid_translation
            grad = end - start
            t_0 = -start[1] / grad[1]
            t_1 = (self.layout.map_size[1] - start[1]) / grad[1]
            easting = grid_east + i
            if t_0 >= 0:
                x_0 = start[0] + t_0 * grad[0]
                if x_0 >= 0 and x_0 <= self.layout.map_size[0]:
                    grid_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_prefix_size,
                            text=f"{easting // 100}",
                            x=x_0,
                            y=top_label_y,
                            text_anchor="end"
                        )
                    )
                    grid_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_text_size,
                            text=f"{easting % 100:02d}",
                            x=x_0,
                            y=top_label_y,
                            text_anchor="start"
                        )
                    )

            if t_1 <= 1.0:
                x_1 = start[0] + t_1 * grad[0]
                if x_1 >= 0 and x_1 <= self.layout.map_size[0]:
                    grid_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_prefix_size,
                            text=f"{easting // 100}",
                            x=x_1,
                            y=bottom_label_y,
                            text_anchor="end"
                        )
                    )
                    grid_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_text_size,
                            text=f"{easting % 100:02d}",
                            x=x_1,
                            y=bottom_label_y,
                            text_anchor="start"
                        )
                    )
        left_labels, right_labels = [], []
        for i in range(self.grid_size[1] + 1):
            start = top_left + (bottom_left - top_left) * i / self.grid_size[1] + grid_translation
            end = top_right + (bottom_right - top_right) * i / self.grid_size[1] + grid_translation
            grad = end - start
            t_0 = -start[0] / grad[0]
            t_1 = (self.layout.map_size[0] - start[0]) / grad[0]
            northing = grid_north - i
            if t_0 >= 0:
                y_0 = start[1] + t_0 * grad[1]
                if y_0 >= 0 and y_0 <= self.layout.map_size[1]:
                    left_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_prefix_size,
                            text=f"{northing // 100}",
                            x=self.layout.map_size[1] - y_0,
                            y=0,
                            text_anchor="end"
                        )
                    )
                    left_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_text_size,
                            text=f"{northing % 100:02d}",
                            x=self.layout.map_size[1] - y_0,
                            y=0,
                            text_anchor="start"
                        )
                    )

            if t_1 <= 1.0:
                y_1 = start[1] + t_1 * grad[1]
                if y_1 >= 0 and y_1 <= self.layout.map_size[1]:
                    right_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_prefix_size,
                            text=f"{northing // 100}",
                            x=y_1,
                            y=0,
                            text_anchor="end"
                        )
                    )
                    right_labels.append(
                        svg.Text(
                            fill="#000000",
                            stroke="#000000",
                            stroke_width=0.1,
                            font_family=FONT_FAMILY,
                            font_size=label_text_size,
                            text=f"{northing % 100:02d}",
                            x=y_1,
                            y=0,
                            text_anchor="start"
                        )
                    )
        grid_labels.append(svg.G(
            elements=left_labels,
            transform=[
                svg.Translate(left_label_x, self.layout.map_size[1]),
                svg.Rotate(-90, 0, 0),
            ]
        ))
        grid_labels.append(svg.G(
            elements=right_labels,
            transform=[
                svg.Translate(right_label_x, 0),
                svg.Rotate(90, 0, 0),
            ]
        ))

        frame_group = svg.G(
            elements=[
                frame,
                svg.G(
                    elements=[
                        svg.G(
                            elements=[
                                svg.Image(
                                    href=self.image_data,
                                    width=self.rotated_map_size[0],
                                    height=self.rotated_map_size[1],
                                    preserveAspectRatio=svg.PreserveAspectRatio("none")
                                ),
                                self.draw_grid()],
                            transform=[
                                svg.Translate(grid_translation_x, grid_translation_y),
                            ]
                        )
                    ],
                    clip_path="url(#frame_clip)"
                )] + grid_labels,
            id="frame",
            transform=[
                svg.Translate(self.layout.left_margin, self.layout.top_margin)  # Adjust position if needed
            ],
        )
        return clip, frame_group
    
    def draw_label(self) -> svg.G:
        font_size = self.layout.top_margin / 5
        label_y = self.layout.top_margin - font_size * 1.5
        label_group = svg.G(
            elements=[
                svg.Text(
                    fill="#000000",
                    stroke="#000000",
                    stroke_width=0.1,
                    font_family=FONT_FAMILY,
                    font_size=font_size,
                    text=f"{self.scale.numerator}:{self.scale.denominator}",
                    x=self.layout.left_margin,
                    y=label_y,
                    text_anchor="start"
                ),
                svg.Text(
                    fill="#000000",
                    stroke="#000000",
                    stroke_width=0.1,
                    font_family=FONT_FAMILY,
                    font_size=font_size,
                    text=f"{self.label}",
                    x=svg.Length(50, '%'),
                    y=label_y,
                    text_anchor="middle"
                ),
                svg.Text(
                    fill="#000000",
                    stroke="#000000",
                    stroke_width=0.1,
                    font_family=FONT_FAMILY,
                    font_size=font_size,
                    text=f"{self.create_time.year}.{self.create_time.month}",
                    x=self.layout.paper_size[0] - self.layout.right_margin,
                    y=label_y,
                    text_anchor="end"
                )
            ],
            id="label"
        )
        return label_group
    
    def draw_scale_bar(self) -> svg.G:
        full_scale_length = int(self.layout.map_size[0] / self.scale / 1000)
        log_scale = int(math.log10(full_scale_length))
        font_size = self.layout.bottom_margin * 0.6 / 15
        if log_scale >= 3:
            unit = "km"
            ratio = 1000
        else:
            unit = "m"
            ratio = 1
        scale_length = 10 ** (log_scale - 1)
        scale_bar_width = float(scale_length * 1000 * self.scale)
        scale_bar_height = font_size / 2
        scale_bar_y = font_size * 6
        scale_bar_text_y = scale_bar_y - font_size * 1.0
        scale_bar_unit_y = scale_bar_y + font_size * 1.5

        scale_bars = [svg.Rect(
            x=scale_bar_width * i,
            y=scale_bar_y - scale_bar_height, width=scale_bar_width, height=scale_bar_height,
            stroke="#000000",
            stroke_width=0.2,
            fill="#cccccc" if i % 2 == 0 else "none"
            ) for i in range(10)]
        scale_texts = [
            svg.Text(
                fill="#000000",
                stroke="#000000",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size=font_size,
                text=f"{i * scale_length / ratio}".removesuffix(".0"),
                x=scale_bar_width * i,
                y=scale_bar_text_y,
                text_anchor="middle")
                for i in range(0, 11, 5)
        ]
        scale_texts.append(
            svg.Text(
                fill="#000000",
                stroke="#000000",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size=font_size,
                text=unit,
                x=scale_bar_width * 10,
                y=scale_bar_unit_y,
                text_anchor="end"
            )
        )
        scale_texts.append(
            svg.Text(
                fill="#000000",
                stroke="#000000",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size= font_size,
                text=f"WGS84 UTM Zone {self.center.zone_number}{self.center.zone_letter}",
                x=0,
                y=scale_bar_unit_y + font_size * 1.5,
                text_anchor="start"
            )
        )
        scale_texts.append(
            svg.Text(
                fill="#808080",
                stroke="#808080",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size= font_size,
                text=f"Parameters: {self.label} {self.center.zone_number}{self.center.zone_letter} {self.center.easting} {self.center.northing} {self.scale.numerator}:{self.scale.denominator}",
                x=0,
                y=scale_bar_unit_y + font_size * 3,
                text_anchor="start"
            )
        )
        scale_texts.append(
            svg.Text(
                fill="#808080",
                stroke="#808080",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size= font_size,
                text=f"Generated by OsmAnd Topographic Map Generator https://github.com/acui/osmand_topo_map_generator",
                x=0,
                y=scale_bar_unit_y + font_size * 4.5,
                text_anchor="start"
            )
        )
        scale_texts.append(
            svg.Text(
                fill="#808080",
                stroke="#808080",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size= font_size,
                text=f"Map is provided by OsmAnd Maps & Navigation https://osmand.net/",
                x=0,
                y=scale_bar_unit_y + font_size * 6,
                text_anchor="start"
            )
        )
        scale_texts.append(
            svg.Text(
                fill="#808080",
                stroke="#808080",
                stroke_width=0.1,
                font_family=FONT_FAMILY,
                font_size= font_size,
                text=f"Licensed under CC BY-NC-ND 2.0 https://github.com/osmandapp/Osmand/blob/master/LICENSE",
                x=0,
                y=scale_bar_unit_y + font_size * 7.5,
                text_anchor="start"
            )
        )
        scale_bar_group = svg.G(
            elements=scale_bars + scale_texts,
            id="scale_bar",
            transform=[
                svg.Translate(self.layout.left_margin, self.layout.paper_size[1] - self.layout.bottom_margin)
            ]
        )
        return scale_bar_group

    def draw_map(self) -> None:
        # h, w = self.canvas.shape[:2]
        clip, frame_group = self.draw_frame()

        self.svg = svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, self.layout.paper_size[0], self.layout.paper_size[1]),
            width=svg.Length(self.layout.paper_size[0], "mm"),
            height=svg.Length(self.layout.paper_size[1], "mm"),
            elements=[
                clip,
                self.draw_label(),
                frame_group,
                self.draw_scale_bar(),
                self.draw_compass(math.degrees(self.tn2gn), self.declination)
            ]
        )

if __name__ == "__main__":
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, CliPositionalArg
    from typing import get_args
    import re

    class Arguments(BaseSettings, cli_parse_args=True):
        name: CliPositionalArg[str] = Field(description="Name of the map")
        zone_number_letter: CliPositionalArg[str] = Field(description="UTM zone number and letter (e.g. 33T)")
        easting: CliPositionalArg[int] = Field(description="Easting coordinate in UTM")
        northing: CliPositionalArg[int] = Field(description="Northing coordinate in UTM")
        scale: CliPositionalArg[str] = Field(description="Scale of the map (e.g. 1:25000)")
        map_root: CliPositionalArg[Path] = Field(description="Root directory for the working directory")
        paper_size: Literal["A4"] = "A4"
        orientation: Literal["portrait", "landscape"] = "portrait"
        
        @field_validator("zone_number_letter", mode="after")
        @classmethod
        def validate_zone_number_letter(cls, v:str) -> str:
            if len(v) != 3:
                raise ValueError("Zone number and letter must be in the format 'NNL' (e.g. 33T)")
            zone_number = int(v[:2])
            zone_letter = v[2].upper()
            if zone_letter not in get_args(UTMZoneLetter):
                raise ValueError(f"Zone letter must be one of {get_args(UTMZoneLetter)}")
            return v
        
        @field_validator("scale", mode="after")
        @classmethod
        def validate_scale(cls, v:str) -> str:
            m = re.match(r"^(\d+):(\d+)$", v)
            if not m:
                raise ValueError("Scale must be in the format 'MMM:NNN' (e.g. 1:25000)")
            scale_numerator = int(m.group(1))
            scale_denominator = int(m.group(2))
            if scale_numerator <= 0 or scale_denominator <= 0:
                raise ValueError("Scale must be positive")
            return v
        
        @field_validator("easting", mode="after")
        @classmethod
        def validate_easting(cls, v:int) -> int:
            if v < 100000 or v > 900000:
                raise ValueError("Easting must be between 100000 and 900000")
            return v
        
        @field_validator("northing", mode="after")
        @classmethod
        def validate_northing(cls, v:int) -> int:
            if v < 0 or v > 10000000:
                raise ValueError("Northing must be between 0 and 10000000")
            return v

    args = Arguments()

    poi = UTMCoordinate(
        zone_number=int(args.zone_number_letter[:2]),
        zone_letter=args.zone_number_letter[2],
        easting=args.easting,
        northing=args.northing
    )

    if not args.map_root.exists():
        args.map_root.mkdir(parents=True, exist_ok=True)
    topo_map = TopoMap(
        layouts[args.paper_size][args.orientation],
        args.name,
        poi,
        Fraction(*map(int, args.scale.split(":")))
    )
    bounding_box = topo_map.gridBoundingBox
    gpx_file = args.map_root / f"{args.name}.gpx"
    topo_map.create_boundary_gpx(gpx_file)
    print(f"Copy the {gpx_file} to OsmAnd")
    print(f"Take the screenshots and save them into the directory: {args.map_root.absolute()}")
    input("Press Enter to continue")
    canvas = read_map(args.map_root)
    corners = get_corners(canvas)
    merged_image_file = args.map_root / f"{args.name}_merged.png"
    # canvas = cv.imread(str(merged_image_file))
    cv.imwrite(str(merged_image_file), canvas)
    print(f"Map image saved to {merged_image_file.name}.")
    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]
    while True:
        print(f"Top-left: {top_left}, Top-right: {top_right}, Bottom-right: {bottom_right}, Bottom-left: {bottom_left}")
        have_result = False
        while True:
            result = input("Is this correct? (y/n): ").strip().lower()
            if result == 'y':
                have_result = True
                break
            if result == 'n':
                break
        if have_result:
            break
        while True:
            top_left = input("Enter the top-left corner of the grid in merged_image_file (format: x,y): ")
            try:
                top_left = np.array(list(map(int, top_left.split(","))))
                break
            except ValueError:
                continue
        while True:
            top_right = input("Enter the top-right corner of the grid in merged_image_file (format: x,y): ")
            try:
                top_right = np.array(list(map(int, top_right.split(","))))
                if top_right[0] <= top_left[0]:
                    print("Top-right corner must be to the right of corner.")
                    continue
                break
            except ValueError:
                continue
        while True:
            
            bottom_right = input("Enter the bottom-right corner of the grid in merged_image_file (format: x,y): ")
            try:
                bottom_right = np.array(list(map(int, bottom_right.split(","))))
                if bottom_right[0] < top_left[0] or bottom_right[1] <= top_left[1]:
                    print("Bottom-right corner must be to the right and below the top-left corner.")
                    continue
                break
            except ValueError:
                continue

        while True:
                
            bottom_left = input("Enter the bottom-left corner of the grid in merged_image_file (format: x,y): ")
            try:
                bottom_left = tuple(map(int, bottom_left.split(",")))
                if bottom_left[1] <= top_left[1]:
                    print("Bottom-left corner must be below the top-left corner.")
                    continue
                break
            except ValueError:
                continue

    top = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    bottom = max( top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    left = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    right = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    canvas = canvas[top:bottom, left:right]
    top_left = np.array([top_left[0] - left, top_left[1] - top])
    top_right = np.array([top_right[0] - left, top_right[1] - top])
    bottom_right = np.array([bottom_right[0] - left, bottom_right[1] - top])
    bottom_left = np.array([bottom_left[0] - left, bottom_left[1] - top])

    topo_map.add_map(canvas, 
        corners=[
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ]
    )
    topo_map.draw_map()
    (args.map_root / f"{args.name}.svg").write_text(str(topo_map.svg))











