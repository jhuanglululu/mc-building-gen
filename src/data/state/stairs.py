import numpy as np

from data.state.block_state import BlockState, Half, Shape, StairFacing, Waterlogged
from data.state.helpers import state_parser

_facing, next = state_parser([
    StairFacing.NORTH,
    StairFacing.EAST,
    StairFacing.SOUTH,
    StairFacing.WEST,
])

_half, next = state_parser([
    Half.BOTTOM,
    Half.TOP
], next)

_shape, next = state_parser([
    Shape.STRAIGHT,
    Shape.INNER_LEFT,
    Shape.INNER_RIGHT,
    Shape.OUTER_RIGHT,
    Shape.OUTER_LEFT
], next)

_waterlogged, _ = state_parser([
    Waterlogged.FALSE,
    Waterlogged.TRUE
], next)

def stairs(state: np.uint16) -> BlockState:
    return BlockState(
        facing=_facing(state),
        half=_half(state),
        shape=_shape(state),
        waterlogged=_waterlogged(state)
    )
