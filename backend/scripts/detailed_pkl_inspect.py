import pickle
import joblib
from pathlib import Path
import json

def inspect_model(pkl_path):
    """Load and inspect the pickle file in detail"""
    path = Path(pkl_path)
    
    try:
        obj = joblib.load(path)
    except Exception as e:
        print(f"Failed to load with joblib: {e}")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e2:
            print(f"Failed to load with pickle: {e2}")
            return
    
    print(f"\n{'='*80}")
    print(f"FILE: {path}")
    print(f"{'='*80}\n")
    
    print(f"Object Type: {type(obj)}")
    print(f"Object Type Name: {type(obj).__module__}.{type(obj).__name__}\n")
    
    # Get all attributes
    all_attrs = dir(obj)
    print(f"Total attributes: {len(all_attrs)}\n")
    
    # Print public attributes (not starting with _)
    public_attrs = [a for a in all_attrs if not a.startswith('_')]
    print(f"Public attributes and methods: {public_attrs}\n")
    
    # Try to get params
    if hasattr(obj, 'get_params'):
        try:
            params = obj.get_params()
            print("get_params():")
            for key, val in sorted(params.items()):
                print(f"  {key}: {val}")
            print()
        except Exception as e:
            print(f"Error getting params: {e}\n")
    
    # Try to get estimators (for ensemble models)
    if hasattr(obj, 'estimators_'):
        try:
            print(f"Number of estimators: {len(obj.estimators_)}")
            print(f"First few estimators: {obj.estimators_[:3]}\n")
        except Exception as e:
            print(f"Error accessing estimators_: {e}\n")
    
    # Try to get feature importances (for tree-based models)
    if hasattr(obj, 'feature_importances_'):
        try:
            fi = obj.feature_importances_
            print(f"Feature importances shape: {fi.shape}")
            print(f"Feature importances (top 10): {sorted(enumerate(fi), key=lambda x: x[1], reverse=True)[:10]}\n")
        except Exception as e:
            print(f"Error accessing feature_importances_: {e}\n")
    
    # Try to get n_features_in
    if hasattr(obj, 'n_features_in_'):
        print(f"Number of input features: {obj.n_features_in_}\n")
    
    # Try to get feature names
    if hasattr(obj, 'feature_names_in_'):
        try:
            print(f"Feature names: {obj.feature_names_in_}\n")
        except Exception as e:
            print(f"Error accessing feature_names_in_: {e}\n")
    
    # For StandardScaler - print mean and scale
    if hasattr(obj, 'mean_'):
        try:
            print(f"Scaler mean_: {obj.mean_}")
            print()
        except Exception as e:
            print(f"Error accessing mean_: {e}\n")
    
    if hasattr(obj, 'scale_'):
        try:
            print(f"Scaler scale_: {obj.scale_}")
            print()
        except Exception as e:
            print(f"Error accessing scale_: {e}\n")
    
    if hasattr(obj, 'var_'):
        try:
            print(f"Scaler var_: {obj.var_}")
            print()
        except Exception as e:
            print(f"Error accessing var_: {e}\n")
    
    print("="*80)


if __name__ == "__main__":
    inspect_model("backend/model/house_price_model.pkl")
    inspect_model("backend/model/scaler.pkl")
