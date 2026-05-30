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

from features import create_features

def train_target_version(version_name, features_list, day49_prepared, geohashes, y_raw, target_type, clip_val=None):
    print(f"\n==================================================")
    print(f" TRAINING VERSION: {version_name} ({target_type} target)")
    print(f"==================================================")
    
    feature_matrix = day49_prepared[features_list].copy()
    feature_matrix = feature_matrix.fillna(feature_matrix.median())
    
    baseline_demand_vector = feature_matrix["base_demand"].values
    raw_target_array = y_raw.values
    
    # process structural mathematical transformations matching target options
    if target_type == "direct":
        target_values = raw_target_array
    elif target_type == "ratio":
        target_values = raw_target_array / np.maximum(baseline_demand_vector, 1e-6)
        if clip_val is not None:
            target_values = np.clip(target_values, 0, clip_val)
    elif target_type == "residual":
        target_values = raw_target_array - baseline_demand_vector
    else:
        raise ValueError("unrecognized operational target string")
        
    output_directory_path = f"models_{version_name}"
    os.makedirs(output_directory_path, exist_ok=True)
    with open(f"{output_directory_path}/feature_cols.pkl", "wb") as model_file:
        pickle.dump(feature_matrix.columns.tolist(), model_file)
        
    # save training feature medians for consistent test imputation
    medians = feature_matrix.median().to_dict()
    with open(f"{output_directory_path}/train_medians.pkl", "wb") as med_file:
        pickle.dump(medians, med_file)
        
    # operational baseline hyperparameter maps with Huber loss
    model_settings = {
        "lgb": (lgb.LGBMRegressor, {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1,
            "objective": "huber", "alpha": 0.9
        }),
        "xgb": (xgb.XGBRegressor, {
            "n_estimators": 250, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0,
            "objective": "reg:pseudohubererror"
        }),
        "cat": (CatBoostRegressor, {
            "iterations": 400, "learning_rate": 0.05, "depth": 6, "random_seed": 42, "verbose": 0,
            "loss_function": "Huber:delta=1.0"
        })
    }
    
    spatial_kfold = GroupKFold(n_splits=5)
    
    # structural dictionary to house out of fold forecast matrices
    oof_predictions_registry = {model_key: np.zeros(len(feature_matrix)) for model_key in model_settings}
    
    for fold, (train_indices, val_indices) in enumerate(spatial_kfold.split(feature_matrix, target_values, groups=geohashes)):
        X_train, y_train = feature_matrix.iloc[train_indices], target_values[train_indices]
        X_val, y_val = feature_matrix.iloc[val_indices], target_values[val_indices]
        
        baseline_demand_validation = baseline_demand_vector[val_indices]
        
        # loop model definitions cleanly over structural setting arrays
        for model_key, (model_constructor, training_parameters) in model_settings.items():
            predictor_instance = model_constructor(**training_parameters)
            predictor_instance.fit(X_train, y_train)
            transformed_predictions = predictor_instance.predict(X_val)
            
            # reverse map targets back into real-world traffic metrics
            if target_type == "direct":
                final_predictions_vector = np.clip(transformed_predictions, 0, 1)
            elif target_type == "ratio":
                final_predictions_vector = np.clip(transformed_predictions * baseline_demand_validation, 0, 1)
            elif target_type == "residual":
                final_predictions_vector = np.clip(baseline_demand_validation + transformed_predictions, 0, 1)
                
            oof_predictions_registry[model_key][val_indices] = final_predictions_vector
            
            model_storage_path = f"{output_directory_path}/{model_key}_fold_{fold}.pkl"
            with open(model_storage_path, "wb") as model_file:
                pickle.dump(predictor_instance, model_file)
                
    calculated_scores = {}
    for model_key in model_settings:
        calculated_scores[model_key] = r2_score(raw_target_array, oof_predictions_registry[model_key])
        
    # compile averaged ensemble matrix configurations
    blended_ensemble_predictions = (
        oof_predictions_registry["lgb"] + 
        oof_predictions_registry["xgb"] + 
        oof_predictions_registry["cat"]
    ) / 3.0
    calculated_scores["ensemble"] = r2_score(raw_target_array, blended_ensemble_predictions)
    
    # Scipy SLSQP optimization to find the best blend weights
    def objective_func(weights):
        w_lgb, w_xgb, w_cat = weights
        blend = (
            w_lgb * oof_predictions_registry["lgb"] +
            w_xgb * oof_predictions_registry["xgb"] +
            w_cat * oof_predictions_registry["cat"]
        )
        return np.mean((blend - raw_target_array) ** 2)
        
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    bounds = [(0, 1)] * 3
    opt_res = minimize(objective_func, [1/3, 1/3, 1/3], method='SLSQP', bounds=bounds, constraints=cons)
    best_weights = opt_res.x
    
    blended_slsqp_predictions = (
        best_weights[0] * oof_predictions_registry["lgb"] +
        best_weights[1] * oof_predictions_registry["xgb"] +
        best_weights[2] * oof_predictions_registry["cat"]
    )
    calculated_scores["ensemble_slsqp"] = r2_score(raw_target_array, blended_slsqp_predictions)
    
    # Save optimized blend weights
    with open(f"{output_directory_path}/blend_weights.pkl", "wb") as weight_file:
        pickle.dump(best_weights, weight_file)
    
    print(f"\nResults for {version_name} ({target_type}):")
    print(f"  LightGBM OOF R2: {calculated_scores['lgb']:.5f}")
    print(f"  XGBoost OOF R2:  {calculated_scores['xgb']:.5f}")
    print(f"  CatBoost OOF R2: {calculated_scores['cat']:.5f}")
    print(f"  Uniform Ensemble OOF R2: {calculated_scores['ensemble']:.5f}")
    print(f"  SLSQP Optimized Ensemble OOF R2: {calculated_scores['ensemble_slsqp']:.5f}")
    print(f"  Optimized weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}")
    
    return calculated_scores

def main():
    print("loading historical traffic datasets...")
    traffic_master_df = pd.read_csv("data/train.csv")
    
    day48_baseline_df = traffic_master_df[traffic_master_df["day"] == 48].copy()
    day49_active_df = traffic_master_df[traffic_master_df["day"] == 49].copy()
    
    print("extracting spatial feature columns matrices...")
    day49_engineered_df = create_features(day49_active_df, ref_df=day48_baseline_df, is_train=True)
    spatial_geohashes = day49_active_df["geohash"].values
    target_traffic_demand = day49_engineered_df["demand"]
    
    # baseline track version A configurations (includes time-dependent steps)
    features_space_A = day49_engineered_df.drop(columns=["demand", "day"], errors="ignore").columns.tolist()
    
    # baseline track version B configurations (excludes time tracking parameters)
    temporal_exclusion_columns = ["slot", "hour", "minute", "sin_time", "cos_time"]
    features_space_B = [col for col in features_space_A if col not in temporal_exclusion_columns]
    
    # kick off evaluation sweeps natively down sequence pipelines
    metrics_version_A = train_target_version("vA", features_space_A, day49_engineered_df, spatial_geohashes, target_traffic_demand, "direct")
    metrics_version_B = train_target_version("vB", features_space_B, day49_engineered_df, spatial_geohashes, target_traffic_demand, "direct")
    metrics_version_C5 = train_target_version("vC5", features_space_B, day49_engineered_df, spatial_geohashes, target_traffic_demand, "ratio", clip_val=5.0)
    metrics_version_D = train_target_version("vD", features_space_B, day49_engineered_df, spatial_geohashes, target_traffic_demand, "residual")

if __name__ == "__main__":
    main()