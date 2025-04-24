from typing import Literal
from pydantic import BaseModel
import utm

UTMZoneLetter = Literal["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]

class UTMCoordinate(BaseModel):
    """
    A dataclass to represent UTM coordinates.
    """
    zone_number: int
    zone_letter: UTMZoneLetter
    easting: int
    northing: int

    def to_latlon(self):
        return utm.to_latlon(self.easting, self.northing, self.zone_number, self.zone_letter)
    
    def __str__(self):
        return f"{self.zone_number}{self.zone_letter} {self.easting} {self.northing}"

class UTMBoundingBox(BaseModel):
    zone_number: int
    zone_letter: UTMZoneLetter
    left: int
    top: int
    right: int
    bottom: int

    @property
    def top_left(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.left,
            northing=self.top
        )

    @property
    def top_right(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.right,
            northing=self.top
        )

    @property
    def bottom_left(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.left,
            northing=self.bottom
        )

    @property
    def bottom_right(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.right,
            northing=self.bottom
        )
    
    @property
    def center(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=(self.right + self.left + 1) // 2,
            northing=(self.top + self.bottom + 1) // 2
        )
    
    @property
    def top_center(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=(self.right + self.left + 1) // 2,
            northing=self.top
        )
    
    @property
    def left_center(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.left,
            northing=(self.top + self.bottom + 1) // 2
        )
    
    @property
    def bottom_center(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=(self.right + self.left + 1) // 2,
            northing=self.bottom
        )
    
    @property
    def right_center(self) -> UTMCoordinate:
        return UTMCoordinate(
            zone_number=self.zone_number,
            zone_letter=self.zone_letter,
            easting=self.right,
            northing=(self.top + self.bottom + 1) // 2
        )