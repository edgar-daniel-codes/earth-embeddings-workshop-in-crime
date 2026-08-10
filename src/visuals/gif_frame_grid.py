### Summer Internship - Earth Embeddings
### Visuals - GIF to per-frame PNG grid
### By Edgar Daniel


"""

Turn every animated GIF under ``docs/figures`` and ``docs/resources`` into a
single static PNG that lays out its frames on an optimally arranged grid.

Each frame is composited onto the house background so partial/transparent GIF
frames collapse to flat images, and the grid shape is chosen to keep the whole
mosaic as close to square as possible given the frame size. Outputs are written
next to each source GIF as ``<name>_frames_grid.png`` (the GIFs are left
untouched).

Run:  python -m src.visuals.gif_frame_grid

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow both `python -m src.visuals.gif_frame_grid` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image as PILImage
from PIL import ImageSequence

from src.utils.style import DEFAULT as PALETTE

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "docs" / "resources",
)

GRID_SUFFIX = "_frames_grid"     # appended to the GIF stem for the output PNG
FRAME_PADDING = 6                # gap (px) between frames and around the grid
BACKGROUND = PALETTE.background  # padding / transparency fill (house white)


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


# Frame extraction


def extract_frames(gif_path: str | Path) -> list[PILImage.Image]:
    """Return every frame of an animated GIF as standalone RGB images.

    Frames are iterated in order (so Pillow resolves GIF disposal/partial
    frames) and composited onto the house background, collapsing palette
    transparency into a flat RGB frame ready to tile.
    """
    frames: list[PILImage.Image] = []
    with PILImage.open(gif_path) as gif:
        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")
            canvas = PILImage.new("RGBA", rgba.size, BACKGROUND)
            canvas.alpha_composite(rgba)
            frames.append(canvas.convert("RGB"))
    return frames


# Grid layout


def optimal_grid(n_frames: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    """(rows, cols) whose mosaic aspect ratio is closest to square.

    Scans every column count and scores each candidate by how far the
    resulting canvas is from a 1:1 aspect, breaking ties toward the layout
    that wastes the fewest empty cells.
    """
    best_key: tuple[float, int] | None = None
    best_shape = (n_frames, 1)

    for cols in range(1, n_frames + 1):
        rows = math.ceil(n_frames / cols)
        ratio = (cols * frame_w) / (rows * frame_h)
        squareness = max(ratio, 1.0 / ratio)      # 1.0 == perfectly square
        empty_cells = rows * cols - n_frames
        key = (round(squareness, 6), empty_cells)

        if best_key is None or key < best_key:
            best_key = key
            best_shape = (rows, cols)

    return best_shape


def frames_to_grid(
    frames: list[PILImage.Image],
    padding: int = FRAME_PADDING,
    background: str = BACKGROUND,
) -> PILImage.Image:
    """Tile ``frames`` onto a single padded, near-square PNG canvas."""
    if not frames:
        raise ValueError("No frames to arrange.")

    cell_w = max(f.width for f in frames)
    cell_h = max(f.height for f in frames)
    rows, cols = optimal_grid(len(frames), cell_w, cell_h)

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    canvas = PILImage.new("RGB", (grid_w, grid_h), background)

    for idx, frame in enumerate(frames):
        row, col = divmod(idx, cols)
        # Center each frame in its cell (frames are usually uniform, but be safe).
        x = padding + col * (cell_w + padding) + (cell_w - frame.width) // 2
        y = padding + row * (cell_h + padding) + (cell_h - frame.height) // 2
        canvas.paste(frame, (x, y))

    return canvas


# Orchestration


def gif_to_grid(
    gif_path: str | Path,
    out_path: str | Path | None = None,
    padding: int = FRAME_PADDING,
    background: str = BACKGROUND,
) -> Path:
    """Convert one GIF into a per-frame grid PNG and save it.

    ``out_path=None`` writes ``<gif stem>_frames_grid.png`` next to the GIF.
    Returns the written path.
    """
    gif_path = Path(gif_path)
    frames = extract_frames(gif_path)
    if not frames:
        raise ValueError(f"No frames found in {gif_path}.")

    grid = frames_to_grid(frames, padding=padding, background=background)

    if out_path is None:
        out_path = gif_path.with_name(f"{gif_path.stem}{GRID_SUFFIX}.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path


def find_gifs(source_dirs=SOURCE_DIRS) -> list[Path]:
    """All GIFs under the given directories, searched recursively."""
    gifs: list[Path] = []
    for directory in source_dirs:
        gifs.extend(sorted(Path(directory).rglob("*.gif")))
    return gifs


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    logger.info("Start GIF -> per-frame grid conversion.")

    gifs = find_gifs()
    logger.info(f"Found {len(gifs)} GIF(s) under docs/figures and docs/resources.")

    for gif in gifs:
        try:
            out = gif_to_grid(gif)
            logger.info(f"Saved frame grid: {out}")
        except Exception as e:
            logger.error(f"Error processing {gif}. {e}")

    logger.info("GIF -> per-frame grid conversion ended.")
