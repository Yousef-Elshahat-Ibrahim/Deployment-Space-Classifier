import io
from pathlib import Path

import requests
import pandas as pd
from alerce.core import Alerce

alerce_client = Alerce()

DATA_DIR = Path("events_data")
DATA_DIR.mkdir(exist_ok=True)

BROKER_RENAME_MAPS = {
    "alerce": {
    },
    "fink": {
        "diaObjectId": "oid",
        "midpointMjdTai": "mjd",
        "band": "band_name",
    },
}


def _cache_path(oid, broker):
    return DATA_DIR / f"{oid}_{broker}.csv"


def find_file_local(oid, broker):
    path = _cache_path(oid, broker)
    try:
        df = pd.read_csv(path)
        # print('File exist locally')
    except FileNotFoundError:
        # print("The file doesn't exist locally. Fetching the file...")
        df = fetch_object(oid, broker)
    return df


def fetch_alerce(oid):
    lightcurve = alerce_client.query_lightcurve(oid, survey='lsst')
    df = pd.DataFrame(lightcurve["detections"])
    df = df.rename(columns=BROKER_RENAME_MAPS["alerce"])
    df.to_csv(_cache_path(oid, "alerce"), index=False)
    return df


def fetch_fink(oid):
    r = requests.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": oid,
            "columns": "r:diaObjectId,r:midpointMjdTai,r:band,r:psfFlux,r:psfFluxErr,r:ra,r:dec,r:psfFlux_flag,r:pixelFlags_bad,r:pixelFlags_saturated",
            "output-format": "json",
        },
    )
    df = pd.read_json(io.BytesIO(r.content))
    df = df.rename(columns=lambda c: c.replace("r:", ""))
    df = df.rename(columns=BROKER_RENAME_MAPS["fink"])
    df.to_csv(_cache_path(oid, "fink"), index=False)
    return df


BROKER_FETCHERS = {
    "alerce": fetch_alerce,
    "fink": fetch_fink,
}


def fetch_object(oid, broker, **kwargs):
    if broker not in BROKER_FETCHERS:
        raise ValueError(f"Unknown broker '{broker}'. Choose from: {list(BROKER_FETCHERS)}")
    return BROKER_FETCHERS[broker](oid, **kwargs)

import io
import requests
import pandas as pd

def search_fink_objects(tag="extragalactic_new_candidate", n_candidates=300, min_detections=30, top_n=10):
    # 1. pull a wide pool of candidate object IDs matching the tag
    r = requests.post(
        "https://api.lsst.fink-portal.org/api/v1/tags",
        json={"tag": tag, "columns": "r:diaObjectId", "n": str(n_candidates)},
    )
    pool = pd.read_json(io.BytesIO(r.content))
    object_ids = pool["r:diaObjectId"].astype(str).unique().tolist()

    if not object_ids:
        return []

    # 2. one batched request to count real detections per object
    r1 = requests.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": ",".join(object_ids),
            "columns": "r:diaObjectId",
            "output-format": "json",
        },
    )
    counts = pd.read_json(io.BytesIO(r1.content))["r:diaObjectId"].astype(str).value_counts()

    # 3. keep only well-observed objects, sorted by detection count
    well_observed = counts[counts >= min_detections].sort_values(ascending=False)
    return well_observed.head(top_n).index.tolist()

def search_objects(survey="lsst", classifier="stamp_classifier_rubin_beta_20260421",
                    class_name="AGN", probability=0.89, n_det=40, page_size=100):
    results = alerce_client.query_objects(
        survey=survey, classifier=classifier, class_name=class_name,
        probability=probability, n_det=n_det, page_size=page_size, format="pandas"
    )
    return results["oid"].tolist()