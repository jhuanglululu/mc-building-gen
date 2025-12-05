from data.state.block_state import BlockState
from data.state.helpers import apply_to
from data.state.stairs import stairs

apply_to(targets=[
    'oak_stairs',
    'birch_stairs',
    'spruce_stairs',
    'stone_stairs',
    'cobblestone_stairs',
    'stone_brick_stairs'
])(stairs)

__all__ = [
    'BlockState'
]
