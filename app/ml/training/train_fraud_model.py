"""
Train fraud detection model on Kaggle dataset
Uses XGBoost classifier with features from claim and fraud score tables
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from sqlalchemy import create_engine


def load_training_data():
    """Load claims and fraud scores from database"""

    print("📥 Loading training data from database...")

    engine = create_engine(settings.DATABASE_URL)

    query = """
    SELECT
        c.claim_number,
        c.claim_type,
        c.status,
        c.estimated_damage,
        c.deductible,
        c.claim_metadata,
        f.fraud_score,
        f.ml_model_score,
        f.risk_level,
        f.requires_investigation,
        p.policy_type,
        p.coverage_limit,
        p.deductible as policy_deductible
    FROM claims c
    JOIN fraud_scores f ON c.id = f.claim_id
    JOIN policies p ON c.policy_id = p.id
    """

    df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(df)} records")

    return df


def extract_features(df):
    """Extract features from claims data"""

    print("🔧 Extracting features...")

    # Parse JSON metadata
    metadata_df = pd.json_normalize(df['claim_metadata'])

    # Combine with main df
    df = pd.concat([df.drop('claim_metadata', axis=1), metadata_df], axis=1)

    # Create binary fraud target (fraud_score >= 0.5)
    df['is_fraud'] = (df['fraud_score'] >= 0.5).astype(int)

    # Encode categorical variables
    categorical_cols = ['claim_type', 'status', 'policy_type', 'risk_level']

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Extract numerical features from metadata
    numerical_features = [
        'estimated_damage',
        'deductible',
        'coverage_limit',
        'policy_deductible',
        'claim_type_encoded',
        'status_encoded',
        'policy_type_encoded',
        'risk_level_encoded'
    ]

    # Add metadata features if they exist
    metadata_features = ['age', 'driver_rating']
    for feat in metadata_features:
        if feat in df.columns:
            numerical_features.append(feat)

    # Handle missing values
    for feat in numerical_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(df[feat].median())

    # Select features that exist
    available_features = [f for f in numerical_features if f in df.columns]

    X = df[available_features]
    y = df['is_fraud']

    print(f"✅ Features: {available_features}")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")

    return X, y, label_encoders, available_features


def train_model(X, y):
    """Train XGBoost classifier"""

    print("\n🏋️  Training XGBoost model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Model Performance:")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

    # Feature importance
    print("\n📈 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, X_test, y_test, y_pred_proba


def save_model(model, label_encoders, features):
    """Save trained model and encoders"""

    print("\n💾 Saving model...")

    model_dir = Path("app/ml/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fraud_detector_v1.pkl"

    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'features': features,
        'version': '1.0',
        'model_type': 'XGBoost'
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved to: {model_path}")

    return model_path


def main():
    """Main training pipeline"""

    print("\n" + "="*60)
    print("🛡️  ClaimGuard Fraud Detection Model Training")
    print("="*60 + "\n")

    # Load data
    df = load_training_data()

    # Extract features
    X, y, label_encoders, features = extract_features(df)

    # Train model
    model, X_test, y_test, y_pred_proba = train_model(X, y)

    # Save model
    model_path = save_model(model, label_encoders, features)

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Features used: {len(features)}")
    print("\nNext steps:")
    print("1. Review model performance metrics above")
    print("2. Use the model in fraud detection service")
    print("3. Integrate with Gradio dashboard")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
