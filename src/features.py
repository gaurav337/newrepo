import pandas as pd
import numpy as np
import warnings
import hashlib

warnings.filterwarnings("ignore")

# lookup dictionaries for categorical traffic and environmental variables
# standard base32 geohash alphabet: 0123456789bcdefghjkmnpqrstuvwxyz
CHAR_TO_CODE = {c: i for i, c in enumerate("0123456789bcdefghjkmnpqrstuvwxyz")}
ROAD_TYPE_MAP = {"Highway": 0, "Street": 1, "Residential": 2, "Unknown": 3}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3, "Unknown": 4}

def parse_time(timestamp_string):
    # maps hours and minutes into clean 15-minute slots across the day
    hours, minutes = map(int, timestamp_string.split(':'))
    return hours * 4 + minutes // 15

def stable_hash(prefix_string):
    # MD5 hash of string to keep it perfectly reproducible across processes/runs
    h_hex = hashlib.md5(str(prefix_string).encode('utf-8')).hexdigest()
    return int(h_hex, 16) % (10**6)

def decode_geohash(geohash):
    # Decode geohash to (latitude, longitude) using pure Python
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for char in geohash:
        if char not in base32:
            continue
        val = base32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                # longitude bit
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if val & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                # latitude bit
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if val & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

def calculate_smoothed_layer(past_df, cluster_level, structural_prior_map, global_prior_value, smoothing_weight):
    # groups data and runs bayesian smoothing down the hierarchical spatial tree
    layer_aggregates = past_df.groupby(cluster_level)["demand"].agg(["sum", "count"])
    smoothed_values_map = {}
    
    for key, row in layer_aggregates.iterrows():
        spatial_parent = key[:-1] if cluster_level != "p2" else None
        prior = structural_prior_map.get(spatial_parent, global_prior_value) if spatial_parent else global_prior_value
        smoothed_values_map[key] = (row["sum"] + smoothing_weight * prior) / (row["count"] + smoothing_weight)
        
    return smoothed_values_map

def get_hierarchical_ratios_vectorized(day49_past, day48_past, target_df, lambda_vals=None):
    # Default bayesian smoothing lambda parameter is set to 150.0 for all hierarchical levels
    if lambda_vals is None:
        lambda_vals = {"geohash": 150.0, "p5": 150.0, "p4": 150.0, "p3": 150.0, "p2": 150.0}
        
    day49_past_df = day49_past.copy()
    day48_past_df = day48_past.copy()
    
    # slice geographical string lengths using vector syntax
    for df in [day49_past_df, day48_past_df]:
        df["p5"] = df["geohash"].str[:5]
        df["p4"] = df["geohash"].str[:4]
        df["p3"] = df["geohash"].str[:3]
        df["p2"] = df["geohash"].str[:2]
        
    global_mean_day49 = day49_past_df["demand"].mean() if len(day49_past_df) > 0 else 0.0
    global_mean_day48 = day48_past_df["demand"].mean() if len(day48_past_df) > 0 else 0.0
    
    # map layer distributions sequentially across day 48 tracks
    p2_day48 = calculate_smoothed_layer(day48_past_df, "p2", {}, global_mean_day48, lambda_vals["p2"])
    p3_day48 = calculate_smoothed_layer(day48_past_df, "p3", p2_day48, global_mean_day48, lambda_vals["p3"])
    p4_day48 = calculate_smoothed_layer(day48_past_df, "p4", p3_day48, global_mean_day48, lambda_vals["p4"])
    p5_day48 = calculate_smoothed_layer(day48_past_df, "p5", p4_day48, global_mean_day48, lambda_vals["p5"])
    geo_day48 = calculate_smoothed_layer(day48_past_df, "geohash", p5_day48, global_mean_day48, lambda_vals["geohash"])

    # map layer distributions sequentially across day 49 tracks
    p2_day49 = calculate_smoothed_layer(day49_past_df, "p2", {}, global_mean_day49, lambda_vals["p2"])
    p3_day49 = calculate_smoothed_layer(day49_past_df, "p3", p2_day49, global_mean_day49, lambda_vals["p3"])
    p4_day49 = calculate_smoothed_layer(day49_past_df, "p4", p3_day49, global_mean_day49, lambda_vals["p4"])
    p5_day49 = calculate_smoothed_layer(day49_past_df, "p5", p4_day49, global_mean_day49, lambda_vals["p5"])
    geo_day49 = calculate_smoothed_layer(day49_past_df, "geohash", p5_day49, global_mean_day49, lambda_vals["geohash"])

    # isolate array boundaries to execute rapid indexing lookups without row loops
    target_geohashes = target_df["geohash"].values
    target_p5 = target_df["geohash"].str[:5].values
    target_p4 = target_df["geohash"].str[:4].values
    target_p3 = target_df["geohash"].str[:3].values
    target_p2 = target_df["geohash"].str[:2].values
    
    hierarchical_ratios_list = []
    for idx in range(len(target_df)):
        g, p5, p4, p3, p2 = target_geohashes[idx], target_p5[idx], target_p4[idx], target_p3[idx], target_p2[idx]
        
        # dynamic resolution fallbacks for unseen/newly opened city grids
        demand_estimate_49 = geo_day49.get(g, p5_day49.get(p5, p4_day49.get(p4, p3_day49.get(p3, p2_day49.get(p2, global_mean_day49)))))
        demand_estimate_48 = geo_day48.get(g, p5_day48.get(p5, p4_day48.get(p4, p3_day48.get(p3, p2_day48.get(p2, global_mean_day48)))))
        
        hierarchical_ratios_list.append(demand_estimate_49 / (demand_estimate_48 + 1e-6))
        
    return hierarchical_ratios_list

def create_features_oot(df, ref_df, day49_past_override, lambda_vals=None, global_impute_dict=None):
    df = df.copy()
    
    if "slot" not in df.columns:
        df["slot"] = df["timestamp"].apply(parse_time)
        
    if "slot" not in ref_df.columns:
        ref_df = ref_df.copy()
        ref_df["slot"] = ref_df["timestamp"].apply(parse_time)
        
    if global_impute_dict is None:
        global_impute_dict = {"temp_median": ref_df["Temperature"].median()}

    # native array missing field assignments
    df["Temperature"] = df["Temperature"].fillna(global_impute_dict["temp_median"])
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")

    # 1. Decode lat/lon from geohash
    lat_lon = [decode_geohash(g) for g in df["geohash"]]
    df["lat"] = [x[0] for x in lat_lon]
    df["lon"] = [x[1] for x in lat_lon]

    # 2. Rotated coordinates at 30, 45, and 60 degrees for better spatial learning
    for angle in [30, 45, 60]:
        rad = np.radians(angle)
        df[f"rot_{angle}_x"] = df["lon"] * np.cos(rad) - df["lat"] * np.sin(rad)
        df[f"rot_{angle}_y"] = df["lon"] * np.sin(rad) + df["lat"] * np.cos(rad)

    # 3. Distances to the top 3 high-demand hotspots from day 48
    hotspots = ref_df.groupby("geohash")["demand"].sum().nlargest(3).index.tolist()
    while len(hotspots) < 3:
        hotspots.append("qp02zt")  # sensible default fallback

    hotspot_coords = [decode_geohash(h) for h in hotspots]
    for i, (h_lat, h_lon) in enumerate(hotspot_coords):
        df[f"dist_hotspot_{i}"] = np.sqrt((df["lat"] - h_lat)**2 + (df["lon"] - h_lon)**2)

    baseline_demand_map = ref_df.set_index(["geohash", "slot"])["demand"].to_dict()
    geohash_mean_map = ref_df.groupby("geohash")["demand"].mean().to_dict()
    geohash_std_map = ref_df.groupby("geohash")["demand"].std().to_dict()
    
    global_traffic_mean = ref_df["demand"].mean()
    global_traffic_std = ref_df["demand"].std()
    
    # fast array pair indexing to avoid multi-key row lookups
    geohash_slot_pairs = list(zip(df["geohash"], df["slot"]))
    df["base_demand"] = [baseline_demand_map.get(pair, geohash_mean_map.get(pair[0], global_traffic_mean)) for pair in geohash_slot_pairs]
        
    df["geo_mean"] = df["geohash"].map(geohash_mean_map).fillna(global_traffic_mean)
    df["geo_std"] = df["geohash"].map(geohash_std_map).fillna(global_traffic_std)
    df["geo_cv"] = df["geo_std"] / (df["geo_mean"] + 1e-6)

    # index sub-character arrays to process positional identifiers
    for i in range(6):
        df[f"geo_char_{i}"] = df["geohash"].str[i].map(CHAR_TO_CODE).fillna(-1).astype(int)

    for length in [2, 3, 4, 5]:
        df[f"geo_pref_{length}"] = df["geohash"].str[:length].apply(stable_hash)

    # map domain categorical parameters into clean execution binaries with fixed mappings
    df["RoadType_cat"] = df["RoadType"].map(ROAD_TYPE_MAP).fillna(3).astype(int)
    df["Weather_cat"] = df["Weather"].map(WEATHER_MAP).fillna(4).astype(int)
    df["LargeVehicles_bin"] = df["LargeVehicles"].map({"Allowed": 1, "Not Allowed": 0}).fillna(-1)
    df["Landmarks_bin"] = df["Landmarks"].map({"Yes": 1, "No": 0}).fillna(-1)

    # target core traffic feature combinations
    df["temp_x_lanes"] = df["Temperature"] * df["NumberofLanes"]
    df["base_x_temp"] = df["base_demand"] * df["Temperature"]
    df["base_x_lanes"] = df["base_demand"] * df["NumberofLanes"]
    
    # execute pipeline calibration fallback ratios
    day48_past = ref_df[ref_df["slot"] <= 4]
    df["loo_early_ratio"] = get_hierarchical_ratios_vectorized(
        day49_past_override, day48_past, target_df=df, lambda_vals=lambda_vals
    )

    drop_columns_targets = ["Index", "timestamp", "geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks", "day", "slot"]
    return df.drop(columns=drop_columns_targets, errors="ignore")

def create_features(df, ref_df, is_train=True, lambda_vals=None, global_impute_dict=None):
    # Common helper that automatically obtains/resolves day49_past_override
    df = df.copy()
    if "slot" not in df.columns:
        df["slot"] = df["timestamp"].apply(parse_time)
        
    if is_train:
        day49_past_override = df[df["slot"] <= 4]
    else:
        # Load day49 early morning slots from train.csv to compute test fallback ratios
        train_df = pd.read_csv("data/train.csv")
        if "slot" not in train_df.columns:
            train_df["slot"] = train_df["timestamp"].apply(parse_time)
        day49_past_override = train_df[(train_df["day"] == 49) & (train_df["slot"] <= 4)]
        
    return create_features_oot(df, ref_df, day49_past_override, lambda_vals=lambda_vals, global_impute_dict=global_impute_dict)