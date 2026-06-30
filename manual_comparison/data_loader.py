#!/usr/bin/env python3
"""
data_loader.py

Parses manual count sheet names, scans AI count files, and matches them
into ComparisonPair objects. Only pairs where both manual and AI data exist
for the same tile and capture date are returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ComparisonPair:
    tile_id: str
    date: datetime
    manual_sheet: str
    ai_file: Path
    ai_alive_sheet: str
    ai_dead_sheet: Optional[str]
    species: Optional[str] = None
    season: Optional[str] = None
    manual_alive: Optional[np.ndarray] = field(default=None, repr=False)
    manual_dead: Optional[np.ndarray] = field(default=None, repr=False)
    ai_alive: Optional[np.ndarray] = field(default=None, repr=False)
    ai_dead: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        short = self.tile_id[-8:] if len(self.tile_id) > 8 else self.tile_id
        return f"{short}_{self.date.strftime('%Y-%m-%d')}"

    @property
    def has_dead(self) -> bool:
        return self.manual_dead is not None and self.ai_dead is not None

    @property
    def tile_group(self) -> str:
        """'T05' for legacy tiles, 'Tile ID' for newer numeric-ID tiles."""
        return "T05" if not self.tile_id.isdigit() else "Tile ID"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> datetime:
    """
    Parse date from the numeric suffix of a manual sheet name.
    Handles YYYYMMDD... (8+ digit) and YYMMDD... (6 digit) formats.
    """
    year_prefix = int(ts[:4])
    if 2000 <= year_prefix <= 2100:
        return datetime.strptime(ts[:8], "%Y%m%d")
    return datetime.strptime("20" + ts[:6], "%Y%m%d")


def parse_manual_sheet_name(sheet_name: str) -> Optional[tuple[str, datetime]]:
    """
    Return (tile_id, capture_date) parsed from a manual count sheet name,
    or None for non-data sheets (Template, Sheet1, etc.).

    Expected formats:
      T05_CG1-202411122300        → tile_id='T05',               date=2024-11-12
      982091078941976_CG1-251222  → tile_id='982091078941976',   date=2025-12-22
      982091078351764_CG1-25122315402 → tile_id='982091078351764', date=2025-12-23
    """
    m = re.match(r"^(.+)_CG1-(\d+)", sheet_name)
    if not m:
        return None
    tile_id = m.group(1)
    try:
        date = _parse_timestamp(m.group(2))
    except ValueError:
        return None
    return tile_id, date


# ---------------------------------------------------------------------------
# AI file indexing
# ---------------------------------------------------------------------------

def _ai_tile_key(filepath: Path) -> str:
    """
    Extract the tile portion from an AI data filename.
    e.g. '2025Dec-982091078941976_data.xlsx' → '982091078941976'
         '2024Oct-MIS5T05_data.xlsx'         → 'MIS5T05'
    """
    stem = filepath.stem.removesuffix("_data")  # e.g. '2025Dec-982091078941976'
    return stem.split("-", 1)[-1]               # e.g. '982091078941976'


def _tiles_match(manual_id: str, ai_key: str) -> bool:
    """
    True when manual tile ID and AI file tile key refer to the same tile.
    Handles exact match and suffix match (e.g. 'T05' is a suffix of 'MIS5T05').
    """
    return (manual_id == ai_key
            or ai_key.endswith(manual_id)
            or manual_id.endswith(ai_key))


def _build_ai_index(ai_dir: Path) -> dict[str, dict]:
    """
    Scan ai_dir for *_data.xlsx files and return a nested dict:
      {
        tile_key: {
          "_species": str | None,
          "_season":  str | None,
          iso_date:   (filepath, alive_sheet, dead_sheet_or_None),
          ...
        }
      }
    Species and season are read from the TileInfo sheet (row 1, cols 1 and 2).
    """
    index: dict[str, dict] = {}
    for ai_file in sorted(ai_dir.glob("*_data.xlsx")):
        key = _ai_tile_key(ai_file)
        xl = pd.ExcelFile(ai_file)
        alive_sheets = {s for s in xl.sheet_names if s.startswith("CM-ALIVE-")}
        dead_sheets  = {s for s in xl.sheet_names if s.startswith("CM-DEAD_CORAL-")}

        species, season = None, None
        if "TileInfo" in xl.sheet_names:
            ti = pd.read_excel(xl, sheet_name="TileInfo", header=None)
            if len(ti) > 1:
                raw_sp = ti.iloc[1, 1]
                raw_se = ti.iloc[1, 2]
                species = str(raw_sp).strip() if pd.notna(raw_sp) else None
                season  = str(raw_se).strip() if pd.notna(raw_se) else None

        index[key] = {"_species": species, "_season": season}
        for sheet in alive_sheets:
            iso = sheet.removeprefix("CM-ALIVE-")          # e.g. '2025-12-22'
            dead = f"CM-DEAD_CORAL-{iso}" if f"CM-DEAD_CORAL-{iso}" in dead_sheets else None
            index[key][iso] = (ai_file, sheet, dead)
    return index


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------

def find_matched_pairs(
    manual_file: Path,
    ai_dir: Path,
) -> tuple[list[ComparisonPair], list[str]]:
    """
    Match manual count sheets to AI count files by tile ID and capture date.

    Returns:
        matched_pairs  – list of ComparisonPair (only where both sides exist)
        unmatched_msgs – human-readable descriptions of unmatched manual sheets
    """
    ai_index = _build_ai_index(ai_dir)
    xl_manual = pd.ExcelFile(manual_file)

    pairs: list[ComparisonPair] = []
    unmatched: list[str] = []

    for sheet in xl_manual.sheet_names:
        parsed = parse_manual_sheet_name(sheet)
        if parsed is None:
            continue
        tile_id, date = parsed
        iso = date.strftime("%Y-%m-%d")

        matched_key = next((k for k in ai_index if _tiles_match(tile_id, k)), None)
        if matched_key is None:
            unmatched.append(f"'{sheet}': no AI file found for tile '{tile_id}'")
            continue
        if iso not in ai_index[matched_key]:
            unmatched.append(f"'{sheet}': AI file has no data for date {iso}")
            continue

        ai_file, alive_sheet, dead_sheet = ai_index[matched_key][iso]
        pairs.append(ComparisonPair(
            tile_id=tile_id,
            date=date,
            manual_sheet=sheet,
            ai_file=ai_file,
            ai_alive_sheet=alive_sheet,
            ai_dead_sheet=dead_sheet,
            species=ai_index[matched_key].get("_species"),
            season=ai_index[matched_key].get("_season"),
        ))

    return pairs, unmatched


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------

def _safe_float(arr: np.ndarray) -> np.ndarray:
    """Convert to float, replacing non-numeric and NaN values with 0."""
    try:
        out = arr.astype(float)
    except (ValueError, TypeError):
        out = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").values.astype(float)
    return np.nan_to_num(out, nan=0.0)


def load_matrices(pair: ComparisonPair, manual_file: Path) -> None:
    """
    Load the 20×20 count matrices into pair in-place.

    Manual layout (pandas, header=None):
      - Row 0:     column-index header (NaN, 0, 1, …, 19)
      - Rows 1–20: row index in col 0, alive counts in cols 1–20,
                   dead row-index in col 22, dead counts in cols 23–42
    AI layout (pandas, header=None):
      - Row 0:     column-index header (0, 1, …, 19)
      - Rows 1–20: counts in cols 0–19
    """
    df = pd.read_excel(manual_file, sheet_name=pair.manual_sheet, header=None)

    pair.manual_alive = _safe_float(df.iloc[1:21, 1:21].values)

    # Dead counts are only present in Tile ID sheets (col 43 populated → shape[1] == 43)
    if df.shape[1] >= 43:
        dead_raw = _safe_float(df.iloc[1:21, 23:43].values)
        if np.nansum(dead_raw) > 0:
            pair.manual_dead = dead_raw

    df_ai = pd.read_excel(pair.ai_file, sheet_name=pair.ai_alive_sheet, header=None)
    pair.ai_alive = _safe_float(df_ai.iloc[1:21, 0:20].values)

    if pair.ai_dead_sheet:
        df_ai_dead = pd.read_excel(pair.ai_file, sheet_name=pair.ai_dead_sheet, header=None)
        ai_dead = _safe_float(df_ai_dead.iloc[1:21, 0:20].values)
        # Only attach AI dead when manual dead also exists to allow paired comparison
        if pair.manual_dead is not None:
            pair.ai_dead = ai_dead
