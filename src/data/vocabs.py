BLOCKS = [
    # transformer tokens
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "air",
    # oak
    "oak_log",
    "oak_planks",
    "oak_stairs",
    "oak_leaves",
    # birch
    "birch_log",
    "birch_planks",
    "birch_stairs",
    "birch_leaves",
    # spruce
    "spruce_log",
    "spruce_planks",
    "spruce_stairs",
    "spruce_leaves",
    # stone
    "stone",
    "stone_stairs",
    "cobblestone",
    "cobblestone_stairs",
    "stone_bricks",
    "stone_brick_stairs",
    # ores
    "coal_ore",
    "iron_ore",
    "gold_ore",
    "diamond_ore",
    # other
    "glass",
    "glass_pane",
]

BLOCK_TO_ID = {block: id for id, block in enumerate(BLOCKS)}
ID_TO_BLOCK = {id: block for id, block in enumerate(BLOCKS)}


def export_blocks(path: str):
    import json

    with open(path, "w+") as f:
        _ = f.write(json.dumps(BLOCK_TO_ID, indent=4))
