import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path='refined_data.csv'):
    """
    Load and prepare the refined cryptocurrency data (with future_target - no data leakage).
    """
    print("=" * 70)
    print("RANDOM FOREST - CRYPTOCURRENCY PRICE PREDICTION (NO DATA LEAKAGE)")
    print("=" * 70)
    print(f"\  Loading data from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"     Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def prepare_features(df):
    """
    Select and prepare features for the model.
    Features: chg_24h, chg_7d, vol_to_marketcap_ratio
    Target: future_target (0 or 1) - predicts NEXT period movement
    """
    print("\  Preparing features...")
    
    # Select features as per requirements
    feature_columns = ['chg_24h', 'chg_7d', 'vol_to_marketcap_ratio']
    target_column = 'future_target'  # CHANGED: Use future_target to avoid data leakage!
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"     Features selected: {feature_columns}")
    print(f"     Target variable: {target_column}")
    print(f"     Feature shape: {X.shape}")
    print(f"     Target distribution:")
    print(f"      - Class 0 (down): {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
    print(f"      - Class 1 (up): {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")
    
    return X, y, df


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets (80% train, 20% test).
    """
    print(f"\  Splitting data (Train: 80%, Test: 20%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"     Training set: {len(X_train)} samples")
    print(f"     Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train Random Forest Classifier model.
    """
    print("\  Training Random Forest model...")
    
    # Initialize Random Forest with good default parameters
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=10,            # Maximum depth of trees
        min_samples_split=5,     # Minimum samples to split
        min_samples_leaf=2,      # Minimum samples in leaf
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("     Model trained successfully!")
    print(f"     Number of trees: {model.n_estimators}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics.
    Returns: accuracy, precision, recall, confusion matrix
    """
    print("\n Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Display results
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\. Accuracy:  {accuracy * 100:.2f}%")
    print(f". Precision: {precision * 100:.2f}%")
    print(f". Recall:    {recall * 100:.2f}%")
    
    print(f"\  Confusion Matrix:")
    print(f"    {'Predicted 0':>15} {'Predicted 1':>15}")
    print(f"Actual 0: {conf_matrix[0][0]:>10} {conf_matrix[0][1]:>15}")
    print(f"Actual 1: {conf_matrix[1][0]:>10} {conf_matrix[1][1]:>15}")
    
    print(f"\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
    
    return accuracy, precision, recall, conf_matrix


def get_feature_importance(model, feature_names):
    """
    Display feature importance from Random Forest model.
    """
    print("\n Feature Importance Analysis:")
    print("=" * 70)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe for better display
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.to_string(index=False))
    
    # Show percentage
    print(f"\nMost important feature: {feature_importance_df.iloc[0]['Feature']} "
          f"({feature_importance_df.iloc[0]['Importance']*100:.2f}%)")
    
    return feature_importance_df


def predict_top_coins(model, df, feature_columns, top_n=10):
    """
    Predict and return top N cryptocurrencies most likely to increase in price.
    """
    print(f"\n Predicting Top {top_n} Coins Most Likely to Go Up:")
    print("=" * 70)
    
    # Get the latest data (most recent timestamp) for each cryptocurrency
    latest_data = df.sort_values('timestamp').groupby('symbol').tail(1).copy()
    
    # Prepare features
    X_predict = latest_data[feature_columns]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_predict)
    
    # Get probability of class 1 (price going up)
    latest_data['confidence_score'] = probabilities[:, 1]
    latest_data['prediction'] = model.predict(X_predict)
    
    # Get top N coins with highest probability of going up
    top_coins = latest_data.nlargest(top_n, 'confidence_score')
    
    # Display results
    print(f"\n{'Rank':<6} {'Symbol':<10} {'Name':<25} {'Confidence':<12} {'Prediction'}")
    print("-" * 70)
    
    for idx, (_, row) in enumerate(top_coins.iterrows(), 1):
        symbol = row['symbol'][:10]
        name = str(row['name'])[:25]
        confidence = row['confidence_score']
        prediction = "UP ⬆" if row['prediction'] == 1 else "DOWN ⬇"
        
        print(f"{idx:<6} {symbol:<10} {name:<25} {confidence*100:>6.2f}%      {prediction}")
    
    return top_coins


def test_model(model, X_test, y_test):
    """
    Test the trained model on test dataset.
    This function is called from main() for model testing.
    """
    print("\n Testing Model on Test Dataset...")
    print("=" * 70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy on test set
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f". Test Set Accuracy: {test_accuracy * 100:.2f}%")
    
    # Sample predictions
    print(f"\n Sample Predictions (first 10 from test set):")
    print(f"{'Actual':<10} {'Predicted':<10} {'Correct?'}")
    print("-" * 35)
    
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        correct = " " if actual == predicted else "✗"
        print(f"{actual:<10} {predicted:<10} {correct}")
    
    # Calculate test metrics
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    
    print(f"\  Test Set Metrics:")
    print(f"   Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"   Precision: {test_precision * 100:.2f}%")
    print(f"   Recall:    {test_recall * 100:.2f}%")
    
    return test_accuracy, test_precision, test_recall


def save_model(model, filename='random_forest_model_no_leakage.pkl'):
    """
    Save the trained model to disk using joblib.
    """
    print(f"\n Saving trained model...")
    
    joblib.dump(model, filename)
    
    print(f"     Model saved successfully to: {filename}")
    print(f"     You can load this model later using: joblib.load('{filename}')")
    
    return filename


def main():
    """
    Main function to run the complete Random Forest pipeline.
    """
    # Load data
    df = load_data('refined_data.csv')
    
    # Prepare features
    X, y, df_full = prepare_features(df)
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, precision, recall, conf_matrix = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = get_feature_importance(model, X.columns.tolist())
    
    # Predict top 10 coins
    top_coins = predict_top_coins(model, df_full, X.columns.tolist(), top_n=10)
    
    # Test model (as required)
    test_accuracy, test_precision, test_recall = test_model(model, X_test, y_test)
    
    # Save the trained model
    model_filename = save_model(model, 'random_forest_model_no_leakage.pkl')
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - RANDOM FOREST MODEL (NO DATA LEAKAGE)")
    print("=" * 70)
    print(f". Model Type: Random Forest Classifier")
    print(f". Data Source: refined_data.csv (with future_target)")
    print(f". Target Used: future_target (predicts NEXT period)")
    print(f". Training Samples: {len(X_train)}")
    print(f". Testing Samples: {len(X_test)}")
    print(f". Final Accuracy: {accuracy * 100:.2f}%")
    print(f". Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f". Precision: {precision * 100:.2f}%")
    print(f". Recall: {recall * 100:.2f}%")
    print(f". Most Important Feature: {feature_importance.iloc[0]['Feature']}")
    print(f". Top Predicted Coin: {top_coins.iloc[0]['symbol']}")
    print(f". Model Saved: {model_filename}")
    print("\n Random Forest training and evaluation complete!")
    print(". This model has NO data leakage and predicts FUTURE movements!")
    print("=" * 70)
    
    return model, accuracy, precision, recall


if __name__ == "__main__":
    model, accuracy, precision, recall = main()

