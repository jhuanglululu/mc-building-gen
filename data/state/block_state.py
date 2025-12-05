from dataclasses import dataclass
from enum import Enum

StairFacing = Enum("StairFacing", ["NONE", "NORTH", "EAST", "SOUTH", "WEST"])
Half = Enum("Half", ["NONE", "TOP", "BOTTOM"])
Shape = Enum("Shape", [
    "NONE", "STRAIGHT",
    "INNER_LEFT", "INNER_RIGHT",
    "OUTER_RIGHT", "OUTER_LEFT"
])
Waterlogged = Enum("WaterLogged", ["NONE", "TRUE", "FALSE"])


@dataclass
class BlockState:
    facing: StairFacing = StairFacing.NONE
    half: Half = Half.NONE
    shape: Shape = Shape.NONE
    waterlogged: Waterlogged = Waterlogged.NONE
