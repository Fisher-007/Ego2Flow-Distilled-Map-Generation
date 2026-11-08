"""Reusable source and generation routines for map construction."""

import argparse


from concurrent.futures import ThreadPoolExecutor


import json


from pathlib import Path


import subprocess


from urllib.parse import urlencode


from urllib.request import urlopen


import xml.etree.ElementTree as ET


S3 = "https://argoverse.s3.amazonaws.com/"


PREFIX = "datasets/av2/sensor/train/"


NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


CITIES = {
    "ATX": "Austin",
    "DTW": "Detroit",
    "MIA": "Miami",
    "PAO": "Palo Alto",
    "PIT": "Pittsburgh",
    "WDC": "Washington, D.C.",
}


def listing(prefix, delimiter=None):
    """Read all pages without including unneeded object metadata."""
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    objects, prefixes = [], []
    while True:
        with urlopen(S3 + "?" + urlencode(params), timeout=60) as response:
            root = ET.fromstring(response.read())
        objects.extend(
            {"key": item.findtext("s:Key", namespaces=NS),
             "size": int(item.findtext("s:Size", namespaces=NS))}
            for item in root.findall("s:Contents", NS)
        )
        prefixes.extend(p.text for p in root.findall("s:CommonPrefixes/s:Prefix", NS))
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            return objects, prefixes
        params["continuation-token"] = token


def map_location(log_id, split="train"):
    prefix = f"datasets/av2/sensor/{split}/"
    objects, _ = listing(prefix + log_id + "/map/")
    vector = next(obj for obj in objects if "/log_map_archive_" in obj["key"])
    city = vector["key"].split("____")[1].split("_")[0]
    return {
        "city_code": city,
        "city": CITIES[city],
        "log_id": log_id,
        "split": split,
        "source": "s3://argoverse/" + prefix + log_id + "/",
        "vector_map_url": S3 + vector["key"],
        "vector_map_bytes": vector["size"],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

