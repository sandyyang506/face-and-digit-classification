import time
import numpy as np
import matplotlib.pyplot as plt
import os

from data import get_dataset
from perceptron import Perceptron

def run_experiment(model_name, data_type, input_size, num_classes, iteration):
    
    print(f"\nRunning {model_name.__name__} on {data_type}s: \n")
    if data_type == "Digit":
        img_train, lbl_train = get_dataset(data_type, split='training')
    else:
        img_train, lbl_train = get_dataset(data_type, split='train')
    img_test, lbl_test = get_dataset(data_type, split='test')
    
    total_train_samples = len(img_train)
    
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {
        "percent": [],
        "mean_acc": [],
        "std_acc": [],
        "mean_time": []
    }
    
    for percent in percentages:
        subset_size = int(total_train_samples * percent)
        
        iteration_accuracies = []
        iteration_times = []
        
        for i in range(iteration):
            shuffled_indices = np.random.permutation(total_train_samples)
            selected_indices = shuffled_indices[:subset_size]
            
            X_train = img_train[selected_indices]
            y_train = lbl_train[selected_indices]
            
            model = model_name(input_size=input_size, num_classes=num_classes)
            
            start_time = time.time()
            model.train(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            iteration_times.append(training_time)
            
            correct_predictions = 0
            total_test_samples = len(img_test)
            
            for j in range(total_test_samples):
                features = img_test[j]
                actual_label = lbl_test[j]
                
                prediction = model.predict(features)
                if prediction == actual_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_test_samples
            iteration_accuracies.append(accuracy)
            
        results["percent"].append(int(percent * 100))
        results["mean_acc"].append(np.mean(iteration_accuracies))
        results["std_acc"].append(np.std(iteration_accuracies))
        results["mean_time"].append(np.mean(iteration_times))
        
        print(f"{int(percent*100)}% Training Accuracy: {results['mean_acc'][-1]:.4f}")
  
    plot_results(results, data_type, model_name.__name__)


def plot_results(res, data_type, model_name):
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(res["percent"], res["mean_acc"], yerr=res["std_acc"], fmt='-o', capsize=5)
    plt.title(f"{model_name} {data_type} Accuracy Graph")
    plt.xlabel("Training Size %")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"graphs/{model_name.lower()}_{data_type}_accuracy.png")
    plt.close()

    # Time Plot
    plt.figure(figsize=(8, 5))
    plt.plot(res["percent"], res["mean_time"], '-o', color='orange')
    plt.title(f"{model_name} {data_type} Training Time Graph")
    plt.xlabel("Training Size %")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.savefig(f"graphs/{model_name.lower()}_{data_type}_time.png")
    plt.close()

def main():
    # Digit input size = 28 * 20 = 560
    # Face input size = 60 * 68 = 4080
    print("Starting Experiments: ")
    run_experiment(Perceptron, 'Digit', 560, 10, 5)
    run_experiment(Perceptron, 'Face', 4080, 2, 5)

if __name__ == "__main__":
    main()

    