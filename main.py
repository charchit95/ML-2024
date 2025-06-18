import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def load_monks_data(problem):
    columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
    train = pd.read_csv(f'monks-{problem}.train', sep=' ', header=None, names=columns)
    test = pd.read_csv(f'monks-{problem}.test', sep=' ', header=None, names=columns)
    
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    X_train = train.drop('class', axis=1)
    y_train = train['class']
    X_test = test.drop('class', axis=1)
    y_test = test['class']
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    return X_train_encoded, X_test_encoded

def evaluate_models(X_train, y_train, X_test, y_test, problem, results_file):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    with open(results_file, 'a') as f:
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'report': report
            }
            
            # Write to file
            f.write(f"\n{'='*50}\n")
            f.write(f"{name} - Monk {problem} Problem\n")
            f.write(f"{'='*50}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            
            # Print to console
            print(f"\n{name} - Monk {problem} Problem")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
    
    return results

def main():
    results_file = "monk-result.txt"
    
    # Clear previous results if file exists
    open(results_file, 'w').close()
    
    for problem in [1, 2, 3]:
        print(f"\n{'='*50}")
        print(f"Analyzing MONK-{problem} Problem")
        print(f"{'='*50}")
        
        X_train, y_train, X_test, y_test = load_monks_data(problem)
        X_train_encoded, X_test_encoded = preprocess_data(X_train, X_test)
        results = evaluate_models(X_train_encoded, y_train, X_test_encoded, y_test, problem, results_file)
        
        plt.figure(figsize=(10, 5))
        plt.bar(results.keys(), [res['accuracy'] for res in results.values()])
        plt.title(f'Model Comparison - MONK-{problem}')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.savefig(f'monk_{problem}_accuracy.png')  # Save plot as image
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()
    print("\nResults saved to 'monk-result.txt'")