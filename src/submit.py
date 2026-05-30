import os
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from features import create_features

def main():
    print("loading training split and evaluation inference data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    day48_baseline = train_df[train_df["day"] == 48].copy()
    
    print("generating inference dataset vectors...")
    test_prepared_df = create_features(test_df, ref_df=day48_baseline, is_train=False)
    
    # pull mapped model headers
    with open("models_vD/feature_cols.pkl", "rb") as model_file:
        saved_features_list = pickle.load(model_file)
        
    feature_matrix = test_prepared_df[saved_features_list]
    
    # load saved training medians for leak-free imputation
    print("loading training medians for imputation...")
    with open("models_vD/train_medians.pkl", "rb") as med_file:
        train_medians = pickle.load(med_file)
        
    feature_matrix = feature_matrix.fillna(train_medians)
    
    print(f"feature matrix geometry path: {feature_matrix.shape}")
    
    baseline_demand_vector = feature_matrix["base_demand"].values
    blended_residual_predictions = np.zeros(len(feature_matrix))
    
    # load optimized blend weights
    print("loading SLSQP optimized blend weights...")
    with open("models_vD/blend_weights.pkl", "rb") as weight_file:
        blend_weights = pickle.load(weight_file)
        
    w_lgb, w_xgb, w_cat = blend_weights
    print(f"Applying blend weights: LGB={w_lgb:.4f}, XGB={w_xgb:.4f}, CAT={w_cat:.4f}")
    
    print("loading model structures and gathering residual predictions (vD engine)...")
    for fold in range(5):
        # LightGBM
        with open(f"models_vD/lgb_fold_{fold}.pkl", "rb") as model_file:
            lgb_model = pickle.load(model_file)
        pred_lgb = lgb_model.predict(feature_matrix)
        
        # XGBoost
        with open(f"models_vD/xgb_fold_{fold}.pkl", "rb") as model_file:
            xgb_model = pickle.load(model_file)
        pred_xgb = xgb_model.predict(feature_matrix)
        
        # CatBoost
        with open(f"models_vD/cat_fold_{fold}.pkl", "rb") as model_file:
            cat_model = pickle.load(model_file)
        pred_cat = cat_model.predict(feature_matrix)
        
        # Blend fold predictions using optimized weights
        blended_residual_predictions += (
            w_lgb * pred_lgb +
            w_xgb * pred_xgb +
            w_cat * pred_cat
        )
        
    blended_residual_predictions /= 5.0
    
    # reconstruct target predictions from the baseline matrix
    final_traffic_predictions = np.clip(baseline_demand_vector + blended_residual_predictions, 0, 1)
    
    # build submission matrix matching index configurations
    submission_df = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": final_traffic_predictions
    })
    
    # sort by Index before saving
    submission_df = submission_df.sort_values(by="Index")
    
    print("\n=== Submission Diagnostics ===")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Min prediction: {submission_df['demand'].min():.5f}")
    print(f"Max prediction: {submission_df['demand'].max():.5f}")
    print(f"Mean prediction: {submission_df['demand'].mean():.5f}")
    print(submission_df.head())
    
    # auto-create folder if not exists
    os.makedirs("submissions", exist_ok=True)
    
    output_target_path = "submissions/ensemble_submission.csv"
    submission_df.to_csv(output_target_path, index=False)
    print(f"\nsaved updated target execution frame to {output_target_path}")

if __name__ == "__main__":
    main()