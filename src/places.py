"""A tiny gazetteer of places inside the 4 Montreal tiles.

Used by the Streamlit app (area picker) and by the offline planner, which
cannot look up place names the way an LLM can. Coordinates are approximate
centres (lon, lat, WGS84).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    name: str
    lon: float
    lat: float
    aliases: tuple[str, ...]


PLACES = [
    Place("Centre of the coverage", -73.6126, 45.4999, ("centre", "center", "middle", "this area", "this lidar area")),
    Place("Université de Montréal", -73.6150, 45.5035,
          ("universite de montreal", "university of montreal", "udem", "roger-gaudry", "roger gaudry")),
    Place("Saint Joseph's Oratory", -73.6178, 45.4925,
          ("oratory", "oratoire", "saint joseph", "saint-joseph", "st joseph", "st. joseph")),
    Place("Côte-des-Neiges / Queen-Mary", -73.6230, 45.4945,
          ("cote-des-neiges", "cote des neiges", "queen-mary", "queen mary")),
]

DEFAULT_PLACE = PLACES[0]


def normalize_text(text: str) -> str:
    """Lowercase and strip accents, so 'Université' matches 'universite'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def find_place(text: str) -> Place | None:
    """First place whose name or alias appears in `text`. The default centre only matches on purpose."""
    text = normalize_text(text)
    for place in PLACES[1:] + PLACES[:1]:
        if any(alias in text for alias in (normalize_text(place.name), *place.aliases)):
            return place
    return None


def get_place(name: str) -> Place:
    return next(p for p in PLACES if p.name == name)
