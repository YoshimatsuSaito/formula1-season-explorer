import sys
import json
from functools import cache
from pathlib import Path
from typing import Any

current_file = Path(__file__)
current_dir = current_file.parent
sys.path.append(str(current_dir.parent))

from modules.utils import load_config


@cache
def load_geo_data(gp_name: str) -> dict[str, Any]:
    """Load geo data of the grand prix circuit"""
    dict_config = load_config(current_dir / "../config/config.yml")

    with open(Path(current_dir.parent) / dict_config["geo_data_dir"] / dict_config["gp_circuits"][gp_name], "r") as geojson_file:
        geojson_data = json.load(geojson_file)

    return geojson_data
