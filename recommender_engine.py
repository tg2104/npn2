# =========================
# Hotel Recommender (Your Logic, Wired Up)
# Works on: Hotel_Reviews_AllCols.csv
# =========================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

import boto3
import pandas as pd

# -------------------------
# 0) Load data (from S3)
# -------------------------
BUCKET_NAME = "npn.tg"  # change to your bucket name
FILE_KEY = "Hotel_Reviews_AllCols.csv"   # file name inside S3

s3 = boto3.client("s3")

def load_data():
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
    df = pd.read_csv(obj["Body"])  # "Body" is a streaming object
    return df

df = load_data()

# Mirror your pipeline’s variable names
df1 = df.copy()  # review-level table used by hybrid_recommender


# Mirror your pipeline’s variable names
df1 = df.copy()  # review-level table used by hybrid_recommender

# -------------------------
# 1) Build hotel table + location utils (EXACT CODE)
# -------------------------

HOTEL_COL = "name"

hotels = (
    df.groupby(HOTEL_COL, as_index=False)
      .agg({"city": "first", "latitude": "median", "longitude": "median"})
)
hotels["latitude"]  = pd.to_numeric(hotels["latitude"], errors="coerce")
hotels["longitude"] = pd.to_numeric(hotels["longitude"], errors="coerce")
hotels = hotels[
    hotels["latitude"].between(-90, 90) & hotels["longitude"].between(-180, 180)
].reset_index(drop=True)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def city_centroid(city_name: str):
    sub = hotels[hotels["city"].fillna("").str.strip().str.lower()
                 == str(city_name).strip().lower()]
    if sub.empty:
        return None
    return float(sub["latitude"].median()), float(sub["longitude"].median())

def recommend_nearest_hotels(city: str, latitude: float=None, longitude: float=None, within_km: float=None):

    if not city or str(city).strip() == "":
        raise ValueError("city is required.")

    if latitude is None or longitude is None:
        centroid = city_centroid(city)
        if centroid is None:
            raise ValueError(f"City '{city}' not found in hotel table.")
        user_lat, user_lon = centroid
    else:
        user_lat, user_lon = float(latitude), float(longitude)

    dfh = hotels.copy()
    dfh["distance_km"] = haversine_km(user_lat, user_lon, dfh["latitude"].values, dfh["longitude"].values)

    if within_km is not None:
        dfh = dfh[dfh["distance_km"] <= float(within_km)].copy()
        if dfh.empty:
            return dfh

    # Proximity score (closer => higher)
    dfh["location_score"] = 1.0 / (1.0 + dfh["distance_km"])

    # Small boost for exact city match
    same_city = (dfh["city"].fillna("").str.strip().str.lower() == str(city).strip().lower()).astype(int)
    dfh["location_score"] *= (1 + 0.05 * same_city)

    keep_cols = [HOTEL_COL, "city", "latitude", "longitude", "distance_km", "location_score"]
    return (dfh.sort_values("location_score", ascending=False)[keep_cols]
              .head(5)
              .reset_index(drop=True))

# -------------------------
# 2) Category map + resolver (EXACT CODE)
# -------------------------

SUB_TO_PARENT = {
    # Hotels
    "corporate lodging": "Hotels",
    "business hotels": "Hotels",
    "budget hotels": "Hotels",
    "family-friendly hotels": "Hotels",
    "resorts": "Hotels",
    "resort": "Hotels",
    "motels": "Hotels",
    "lodging": "Hotels",

    # Restaurants
    "bars": "Restaurants",
    "bar": "Restaurants",
    "caterers": "Restaurants",
    "catering": "Restaurants",
    "cafes": "Restaurants",
    "fast food": "Restaurants",
    "restaurants": "Restaurants",
    "restaurant": "Restaurants",

    # Travel
    "airport": "Travel",
    "airports": "Travel",
    "travel agency": "Travel",
    "tour operators": "Travel",
    "concierge service": "Travel",
    "tourist attractions": "Travel",
    "attractions": "Travel",

    # Entertainment
    "banquet halls": "Entertainment",
    "banquet facilities": "Entertainment",
    "banquet rooms": "Entertainment",
    "golf courses": "Entertainment",
    "theaters": "Entertainment",
    "cinema": "Entertainment",
    "nightlife": "Entertainment",
    "clubs": "Entertainment",

    # Shopping
    "malls": "Shopping",
    "retail": "Shopping",
    "market": "Shopping",
    "souvenirs": "Shopping",
}

def resolve_categories(desired_categories, cat_cols):
    """
    Returns:
      parent_cols: list of top-level encoded columns to check (subset of cat_cols)
      raw_terms: list of raw substrings to match against df1['categories']
    """
    if desired_categories is None:
        return [], []
    if isinstance(desired_categories, str):
        desired_categories = [desired_categories]

    # Handle "All" shortcut (no category filtering)
    if any(str(x).strip().lower() == "all" for x in desired_categories):
        return [], []  # signal no filtering

    # Build maps for case-insensitive column matching
    col_map = {c.lower(): c for c in cat_cols}

    parent_cols = set()
    raw_terms = []

    for term in desired_categories:
        t = str(term).strip().lower()
        # If a top-level category is directly provided (e.g., "hotels")
        if t in col_map:
            parent_cols.add(col_map[t])
            continue
        # If it's a known subcategory, map up to parent mega-category
        if t in SUB_TO_PARENT:
            parent = SUB_TO_PARENT[t].lower()
            if parent in col_map:
                parent_cols.add(col_map[parent])
            else:
                # parent not in encoded columns; treat as raw term
                raw_terms.append(t)
        else:
            # fallback to raw substring matching
            raw_terms.append(t)

    return list(parent_cols), raw_terms

# -------------------------
# 3) Provide mlb & df_final (no logic change; just wiring)
#    We read the one-hot category columns directly from the CSV.
# -------------------------
_expected_cat_cols = ["Hotels", "Restaurants", "Travel", "Entertainment", "Shopping", "Other"]
_present_cat_cols = [c for c in _expected_cat_cols if c in df1.columns]

mlb = MultiLabelBinarizer()
mlb.classes_ = np.array(_present_cat_cols)  # so hybrid_recommender can use mlb.classes_ unchanged

# df_final is only used to group by name and average the encoded category columns
# SAFETY: if none of the expected category columns exist, keep just 'name' to avoid KeyErrors later.
if _present_cat_cols:
    df_final = df1[["name"] + _present_cat_cols].copy()
else:
    df_final = df1[["name"]].copy()

# -------------------------
# 4) Hybrid recommender (EXACT CODE + stability guards)
# -------------------------

def hybrid_recommender(
    user_city,
    min_rating,
    min_sentiment,
    min_recency,
    desired_categories=None,     # e.g., "Hotels" or ["Hotels","Entertainment"]
    category_logic="any",        # "any" or "all"
    desired_segments=None,       # e.g., "Family" or ["Family","Business"] or "All"
    user_lat=None,
    user_lon=None,
    within_km=None,
    top_k=5
):
    # ---------- 0) Preconditions ----------
    # We expect df1 to already contain:
    # - reviews.rating, sentiment_score (per review), reviews.date
    # - categories (raw), and df_final with one-hot categories via mlb.classes_
    # - primary_segment (one of: Family, Couple, Business, Friends, Pet, Solo, or All fallback)

    # ---------- 1) Build hotel-level features ----------
    df1['reviews.date'] = pd.to_datetime(df1['reviews.date'], errors='coerce')
    earliest_date = df1['reviews.date'].min()
    df1['reviews.date'] = df1['reviews.date'].fillna(earliest_date)

    hotel_features = (
        df1.groupby("name", as_index=False)
           .agg({
               "city": "first",
               "latitude": "median",
               "longitude": "median",
               "reviews.rating": "mean",
               "sentiment_score": "mean",    # TF-IDF + XGBoost prob (0..1)
               "reviews.date": "max"
           })
    )

    # Recency score (per hotel using most-recent review)
    DATE_COL = "reviews.date"
    hotel_features[DATE_COL] = hotel_features[DATE_COL].dt.tz_localize(None)
    today = pd.Timestamp("now").tz_localize(None)
    hotel_features["days_since_review"] = (today - hotel_features[DATE_COL]).dt.days
    alpha = 0.001
    hotel_features["recency_score"] = np.exp(
        -alpha * hotel_features["days_since_review"].fillna(hotel_features["days_since_review"].max())
    )

    # Merge one-hot categories (per hotel)
    cat_cols = list(mlb.classes_)
    if cat_cols:
        categories_encoded = df_final.groupby("name")[cat_cols].mean().reset_index()
    else:
        # SAFETY: if no category columns exist, merge a minimal frame to avoid KeyErrors
        categories_encoded = df1[["name"]].drop_duplicates().reset_index(drop=True)
    hotel_features = hotel_features.merge(categories_encoded, on="name", how="left")

    # ---------- 2) Segment preference (NEW) ----------
    # Normalize primary_segment to title-case for consistency
    if "primary_segment" not in df1.columns:
        raise KeyError("primary_segment not found in df1. Please create it first.")

    df1["_ps"] = df1["primary_segment"].fillna("All").astype(str).str.title()

    # Normalize desired_segments to a list
    if desired_segments is None:
        desired_segments_norm = None
    elif isinstance(desired_segments, str):
        desired_segments_norm = [desired_segments.strip().title()]
    else:
        desired_segments_norm = [str(s).strip().title() for s in desired_segments]

    # Per-hotel segment match score: share of reviews whose primary_segment ∈ desired_segments
    if desired_segments_norm is None or ("All" in desired_segments_norm):
        seg_match = df1.groupby("name", as_index=False) \
                       .agg(segment_match_score=("primary_segment", lambda x: 1.0))  # no filtering: everyone gets 1
    else:
        seg_match = (
            df1.assign(_match=df1["_ps"].isin(desired_segments_norm).astype(float))
               .groupby("name", as_index=False)["_match"].mean()
               .rename(columns={"_match": "segment_match_score"})
        )

    hotel_features = hotel_features.merge(seg_match, on="name", how="left")
    hotel_features["segment_match_score"] = hotel_features["segment_match_score"].fillna(0.0)

    # ---------- 3) Distance + location score ----------
    hotel_features = hotel_features.dropna(subset=["latitude", "longitude"])
    if user_lat is not None and user_lon is not None:
        hotel_features["distance_km"] = haversine_km(
            user_lat, user_lon,
            hotel_features["latitude"].values, hotel_features["longitude"].values
        )
    else:
        centroid = city_centroid(user_city)  # 'hotels' is your hotel table
        if centroid is not None:
            user_lat, user_lon = centroid
            hotel_features["distance_km"] = haversine_km(
                user_lat, user_lon,
                hotel_features["latitude"].values, hotel_features["longitude"].values
            )
        else:
            hotel_features["distance_km"] = np.nan

    # SAFETY: only apply within_km if we have any finite distances
    if within_km is not None:
        dist_series = hotel_features["distance_km"]
        has_any_distance = np.isfinite(dist_series).any()
        if has_any_distance:
            hotel_features = hotel_features[dist_series <= float(within_km)]

    hotel_features["location_score"] = 1.0 / (1.0 + hotel_features["distance_km"].fillna(1e6))

    # ---------- 4) City-first ----------
    city_hotels = hotel_features[
        hotel_features["city"].fillna("").str.strip().str.lower() == str(user_city).strip().lower()
    ].copy()
    if city_hotels.empty:
        print(f"⚠️ No hotels found in {user_city}. Expanding search.")
        city_hotels = hotel_features.copy()

    # ---------- 5) Category filter ----------
    parent_cols, raw_terms = resolve_categories(desired_categories, cat_cols)

    def apply_category_filter(df_hotels):
        if desired_categories is None:
            return df_hotels
        if isinstance(desired_categories, str) and desired_categories.strip().lower() == "all":
            return df_hotels
        if isinstance(desired_categories, list) and any(str(x).strip().lower() == "all" for x in desired_categories):
            return df_hotels

        df2 = df_hotels.reset_index(drop=True).copy()

        # encoded (top-level) masks
        if parent_cols:
            enc_mask_any = (df2[parent_cols] > 0).any(axis=1)
            enc_mask_all = (df2[parent_cols] > 0).all(axis=1)
        else:
            enc_mask_any = pd.Series(False, index=df2.index)
            enc_mask_all = pd.Series(False, index=df2.index)

        # raw substring mask (SAFETY: handle missing 'categories')
        if "categories" in df1.columns:
            raw_cat_per_hotel = (
                df1.groupby("name", as_index=False)["categories"]
                   .agg(lambda x: " | ".join(set(map(str, x))))
                   .rename(columns={"categories": "raw_categories_concat"})
            )
        else:
            raw_cat_per_hotel = df1[["name"]].drop_duplicates()
            raw_cat_per_hotel["raw_categories_concat"] = ""

        df2 = df2.merge(raw_cat_per_hotel, on="name", how="left")
        if raw_terms:
            raw_str = df2["raw_categories_concat"].fillna("").str.lower()
            raw_mask_any = raw_str.apply(lambda s: any(term in s for term in raw_terms))
            raw_mask_all = raw_str.apply(lambda s: all(term in s for term in raw_terms))
        else:
            raw_mask_any = pd.Series(False, index=df2.index)
            raw_mask_all = pd.Series(False, index=df2.index)

        if category_logic == "all":
            if parent_cols and raw_terms:
                keep_mask = enc_mask_all & raw_mask_all
            elif parent_cols:
                keep_mask = enc_mask_all
            elif raw_terms:
                keep_mask = raw_mask_all
            else:
                keep_mask = pd.Series(True, index=df2.index)
        else:  # "any"
            if parent_cols or raw_terms:
                keep_mask = enc_mask_any | raw_mask_any
            else:
                keep_mask = pd.Series(True, index=df2.index)

        return df2[keep_mask].drop(columns=["raw_categories_concat"], errors="ignore")

    city_hotels = apply_category_filter(city_hotels)

    # ---------- 6) Threshold filters (rating/sentiment/recency) ----------
    filtered = city_hotels[
        (city_hotels["reviews.rating"].fillna(0) >= float(min_rating)) &
        (city_hotels["sentiment_score"].fillna(0) >= float(min_sentiment)) &
        (city_hotels["recency_score"].fillna(0) >= float(min_recency))
    ]

    # If too few, fill from elsewhere (still honoring category + segment prefs)
    if filtered.empty:
        print(f"⚠️ No hotels found in {user_city} met thresholds — showing closest matches overall.")
        pool = apply_category_filter(hotel_features.copy())
        filtered = pool.sort_values("distance_km").head(top_k).copy()
    elif len(filtered) < top_k:
        print(f"⚠️ Only {len(filtered)} hotels in {user_city} met thresholds. Filling with nearest from other cities.")
        needed = top_k - len(filtered)
        pool = apply_category_filter(hotel_features[~hotel_features["name"].isin(filtered["name"])].copy())
        extra = pool.sort_values("distance_km").head(needed)
        filtered = pd.concat([filtered, extra], ignore_index=True)
    else:
        print(f"✅ Showing hotels in {user_city} that meet user thresholds.")

    # ---------- 7) Similarity-based ranking (now includes segment score) ----------
    sim_cols = ["reviews.rating", "sentiment_score", "recency_score", "location_score", "segment_match_score"] + cat_cols

    # SAFETY: if nothing left (or no feature columns), avoid cosine on empty matrix
    if filtered.empty:
        return filtered  # no rows to score

    sim_cols = [c for c in sim_cols if c in filtered.columns]
    if not sim_cols:
        # No valid feature columns available; set a default constant score to keep flow
        filtered = filtered.copy()
        filtered["hybrid_score"] = 0.0
    else:
        X = filtered[sim_cols].fillna(0)
        if X.shape[0] == 0:
            return filtered  # nothing to score
        # Cosine similarity requires at least one sample and one feature
        if X.shape[1] == 0:
            filtered = filtered.copy()
            filtered["hybrid_score"] = 0.0
        else:
            sim_matrix = cosine_similarity(X)
            filtered = filtered.copy()
            filtered["hybrid_score"] = sim_matrix.mean(axis=1)

    # ---------- 8) Output ----------
    cols_out = [
        "name", "city", "reviews.rating", "sentiment_score", "recency_score",
        "segment_match_score", "distance_km", "hybrid_score"
    ]
    if desired_categories is not None and parent_cols:
        cols_out += parent_cols

    # SAFETY: keep only columns that exist
    cols_out = [c for c in cols_out if c in filtered.columns]

    return (
        filtered.sort_values("hybrid_score", ascending=False)
                .head(top_k)[cols_out]
                .reset_index(drop=True)
    )
