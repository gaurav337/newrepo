import os
import pickle
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from features import parse_time, create_features_oot

def main():
    print("loading historical training data tracks...")
    traffic_master_df = pd.read_csv("data/train.csv")
    
    day48_baseline = traffic_master_df[traffic_master_df["day"] == 48].copy()
    day49_active = traffic_master_df[traffic_master_df["day"] == 49].copy()
    
    if "slot" not in day48_baseline.columns:
        day48_baseline["slot"] = day48_baseline["timestamp"].apply(parse_time)
    if "slot" not in day49_active.columns:
        day49_active["slot"] = day49_active["timestamp"].apply(parse_time)
        
    # Split day 49 into morning past (slots 0-4) and future targets (slots 5-8)
    day49_morning_past = day49_active[day49_active["slot"] <= 4].copy()
    day49_target_future = day49_active[day49_active["slot"].between(5, 8)].copy()
    
    future_geohashes = day49_target_future["geohash"].values
    actual_traffic_demand = day49_target_future["demand"].values
    
    # Extract feature names first using a temporary feature extraction
    temp_features_raw = create_features_oot(
        day49_target_future.iloc[:10], ref_df=day48_baseline, day49_past_override=day49_morning_past
    )
    temporal_exclusion_columns = ["slot", "hour", "minute", "sin_time", "cos_time"]
    model_features = [col for col in temp_features_raw.columns if col != "demand" and col not in temporal_exclusion_columns]
    
    output_directory_path = "models_vD_honest"
    os.makedirs(output_directory_path, exist_ok=True)
    with open(f"{output_directory_path}/feature_cols.pkl", "wb") as model_file:
        pickle.dump(model_features, model_file)
        
    spatial_group_kfold = GroupKFold(n_splits=5)
    
    # structural dictionaries to house out of fold forecasts
    model_keys = ["lgb", "xgb", "cat"]
    oof_predictions = {m: np.zeros(len(day49_target_future)) for m in model_keys}
    
    # Huber loss model configurations
    model_settings = {
        "lgb": (lgb.LGBMRegressor, {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
            "random_state": 42, "verbose": -1, "objective": "huber", "alpha": 0.9
        }),
        "xgb": (xgb.XGBRegressor, {
            "n_estimators": 250, "learning_rate": 0.05, "max_depth": 6,
            "random_state": 42, "verbosity": 0, "objective": "reg:pseudohubererror"
        }),
        "cat": (CatBoostRegressor, {
            "iterations": 400, "learning_rate": 0.05, "depth": 6,
            "random_seed": 42, "verbose": 0, "loss_function": "Huber:delta=1.0"
        })
    }
    
    print("==================================================")
    print(" TRAINING HONEST PIPELINE (UNSEEN GEOHASH CV)")
    print("==================================================")
    
    for fold, (train_indices, val_indices) in enumerate(spatial_group_kfold.split(day49_target_future, actual_traffic_demand, groups=future_geohashes)):
        train_fold_data = day49_target_future.iloc[train_indices].copy()
        val_fold_data = day49_target_future.iloc[val_indices].copy()
        
        train_fold_geohashes = set(train_fold_data["geohash"].unique())
        val_fold_geohashes = set(val_fold_data["geohash"].unique())
        
        # enforce leak-free past data (mask validation geohashes from the training-fold past statistics)
        train_fold_morning_past = day49_morning_past[day49_morning_past["geohash"].isin(train_fold_geohashes)].copy()
        
        # 1. Build training features
        train_features_raw = create_features_oot(
            train_fold_data, ref_df=day48_baseline, day49_past_override=train_fold_morning_past
        )
        train_features = train_features_raw[model_features].fillna(train_features_raw[model_features].median())
        
        baseline_demand_train = train_features["base_demand"].values
        train_residuals = train_fold_data["demand"].values - baseline_demand_train
        
        # 2. Build validation features using train-fold past (so validation geohashes fallback strictly)
        val_features_raw = create_features_oot(
            val_fold_data, ref_df=day48_baseline, day49_past_override=train_fold_morning_past
        )
        val_features = val_features_raw[model_features].fillna(train_features.median())
        baseline_demand_validation = val_features["base_demand"].values
        
        # 3. Fit models and predict
        for model_key, (model_constructor, training_parameters) in model_settings.items():
            predictor_instance = model_constructor(**training_parameters)
            predictor_instance.fit(train_features, train_residuals)
            
            predicted_validation_residuals = predictor_instance.predict(val_features)
            final_predictions_vector = np.clip(baseline_demand_validation + predicted_validation_residuals, 0, 1)
            
            oof_predictions[model_key][val_indices] = final_predictions_vector
            
            model_storage_path = f"{output_directory_path}/{model_key}_fold_{fold}.pkl"
            with open(model_storage_path, "wb") as model_file:
                pickle.dump(predictor_instance, model_file)
                
    calculated_scores = {}
    for model_key in model_keys:
        calculated_scores[model_key] = r2_score(actual_traffic_demand, oof_predictions[model_key])
        
    uniform_ensemble_preds = (
        oof_predictions["lgb"] +
        oof_predictions["xgb"] +
        oof_predictions["cat"]
    ) / 3.0
    calculated_scores["ensemble"] = r2_score(actual_traffic_demand, uniform_ensemble_preds)
    
    # Scipy SLSQP optimization to find the best blend weights on OOF predictions
    def objective_func(weights):
        w_lgb, w_xgb, w_cat = weights
        blend = (
            w_lgb * oof_predictions["lgb"] +
            w_xgb * oof_predictions["xgb"] +
            w_cat * oof_predictions["cat"]
        )
        return np.mean((blend - actual_traffic_demand) ** 2)
        
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    bounds = [(0, 1)] * 3
    opt_res = minimize(objective_func, [1/3, 1/3, 1/3], method='SLSQP', bounds=bounds, constraints=cons)
    best_weights = opt_res.x
    
    blended_slsqp_predictions = (
        best_weights[0] * oof_predictions["lgb"] +
        best_weights[1] * oof_predictions["xgb"] +
        best_weights[2] * oof_predictions["cat"]
    )
    calculated_scores["ensemble_slsqp"] = r2_score(actual_traffic_demand, blended_slsqp_predictions)
    
    # Save optimized blend weights
    with open(f"{output_directory_path}/blend_weights.pkl", "wb") as weight_file:
        pickle.dump(best_weights, weight_file)
        
    print("\n=== HONEST EVALUATION SCORES (STRICT OOT UNSEEN) ===")
    print(f"LightGBM OOF R2:                  {calculated_scores['lgb']:.5f}")
    print(f"XGBoost OOF R2:                   {calculated_scores['xgb']:.5f}")
    print(f"CatBoost OOF R2:                  {calculated_scores['cat']:.5f}")
    print(f"Uniform Ensemble OOF R2:          {calculated_scores['ensemble']:.5f}")
    print(f"SLSQP Optimized Ensemble OOF R2:  {calculated_scores['ensemble_slsqp']:.5f}")
    print(f"Optimized weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}")

if __name__ == "__main__":
    main()