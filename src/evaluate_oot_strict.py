import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from features import create_features_oot

def calculate_time_slot(timestamp_string):
    # splits 24h clock strings into 15-minute intervals (96 slots a day)
    hours, minutes = map(int, timestamp_string.split(':'))
    return hours * 4 + minutes // 15

def run_strict_oot_pipeline():
    print("reading baseline datasets...")
    traffic_data = pd.read_csv("data/train.csv")
    
    traffic_data["slot"] = traffic_data["timestamp"].apply(calculate_time_slot)
    
    day48_baseline = traffic_data[traffic_data["day"] == 48].copy()
    day49_active = traffic_data[traffic_data["day"] == 49].copy()
    
    # split morning slots from target evaluation windows
    day49_morning_past = day49_active[day49_active["slot"] <= 4].copy()
    day49_target_future = day49_active[day49_active["slot"].between(5, 8)].copy()
    
    target_geohashes = day49_target_future["geohash"].values
    actual_traffic_demand = day49_target_future["demand"].values
    
    spatial_kfold = GroupKFold(n_splits=5)
    out_of_fold_seen_predictions = np.zeros(len(day49_target_future))
    out_of_fold_unseen_predictions = np.zeros(len(day49_target_future))
    
    # extract valid model feature columns using a small sample slice
    feature_check_df = create_features_oot(
        day49_target_future.iloc[:5], 
        ref_df=day48_baseline, 
        day49_past_override=day49_morning_past
    )
    model_features = [col for col in feature_check_df.columns if col != "demand"]
    
    traffic_model_params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }
    
    print(f"kicking off spatial cross validation over {len(model_features)} features...")
    
    for fold, (train_indices, val_indices) in enumerate(spatial_kfold.split(day49_target_future, actual_traffic_demand, groups=target_geohashes)):
        train_fold_data = day49_target_future.iloc[train_indices].copy()
        val_fold_data = day49_target_future.iloc[val_indices].copy()
        
        # isolate morning historical records for training locations only
        active_train_geohashes = set(train_fold_data["geohash"].unique())
        train_fold_morning_past = day49_morning_past[day49_morning_past["geohash"].isin(active_train_geohashes)].copy()
        
        train_features_raw = create_features_oot(
            train_fold_data, 
            ref_df=day48_baseline, 
            day49_past_override=train_fold_morning_past
        )
        X_train = train_features_raw[model_features].fillna(train_features_raw[model_features].median())
        
        # calculate residuals against base demand configuration
        y_train = train_fold_data["demand"].values - X_train["base_demand"].values
        
        traffic_predictor = lgb.LGBMRegressor(**traffic_model_params)
        traffic_predictor.fit(X_train, y_train)
        
        # evaluation path 1: seen locations (retains morning history mapping)
        validation_seen_features = create_features_oot(
            val_fold_data, 
            ref_df=day48_baseline, 
            day49_past_override=day49_morning_past
        )
        X_val_seen = validation_seen_features[model_features].fillna(X_train.median())
        predicted_seen_residuals = traffic_predictor.predict(X_val_seen)
        
        out_of_fold_seen_predictions[val_indices] = np.clip(
            X_val_seen["base_demand"].values + predicted_seen_residuals, 0, 1
        )
        
        # evaluation path 2: unseen locations (simulates completely new/blackout grids)
        # remove test junction locations from day 48 history mapping matrix
        validation_geohashes_set = set(val_fold_data["geohash"])
        unseen_day48_reference = day48_baseline[~day48_baseline["geohash"].isin(validation_geohashes_set)].copy()
        
        validation_unseen_features = create_features_oot(
            val_fold_data, 
            ref_df=unseen_day48_reference, 
            day49_past_override=train_fold_morning_past
        )
        X_val_unseen = validation_unseen_features[model_features].fillna(X_train.median())
        predicted_unseen_residuals = traffic_predictor.predict(X_val_unseen)
        
        out_of_fold_unseen_predictions[val_indices] = np.clip(
            X_val_unseen["base_demand"].values + predicted_unseen_residuals, 0, 1
        )
        
        print(f"   -> completed split fold {fold + 1}/5 testing tracks.")
        
    print("\nFINAL GRIDLOCK INTELLIGENCE MODEL STANDINGS:")
    print(f"   SEEN ZONES R2 SCORE   : {r2_score(actual_traffic_demand, out_of_fold_seen_predictions):.5f}")
    print(f"   UNSEEN ZONES R2 SCORE : {r2_score(actual_traffic_demand, out_of_fold_unseen_predictions):.5f}")

if __name__ == "__main__":
    run_strict_oot_pipeline()