import numpy as np
from src.model import NeuralNetwork
from src.data_loader import load_processed_mnist
import os

def train_network(nn, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=64, initial_lr=0.1, save_dir=r"outputs"):
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print(f"Training for up to {epochs} epochs with batch size {batch_size}, {num_batches} batches...")
    for epoch in range(epochs):
        learning_rate = initial_lr * (0.9 ** epoch)
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        total_loss = 0
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, num_samples)
            X_batch = X_train_shuffled[start:end]
            Y_batch = Y_train_shuffled[start:end]
            A1, A2, A3, Z1, Z2, Z3 = nn.forward_propagation(X_batch)
            loss = nn.compute_loss(A3, Y_batch)
            total_loss += loss * (end - start)
            grads = nn.backward_propagation(X_batch, Y_batch, A1, A2, A3, Z1, Z2, Z3)
            nn.update_parameters(grads, learning_rate, l2_lambda=0.01)
        avg_train_loss = total_loss / num_samples
        
        _, _, A3_val, _, _, _ = nn.forward_propagation(X_val)
        val_loss = nn.compute_loss(A3_val, Y_val)
        
        print(f"Epoch {epoch + 1}/{epochs}, LR: {learning_rate:.6f}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "W1.npy"), nn.W1)
            np.save(os.path.join(save_dir, "W2.npy"), nn.W2)
            np.save(os.path.join(save_dir, "W3.npy"), nn.W3)
            np.save(os.path.join(save_dir, "b1.npy"), nn.b1)
            np.save(os.path.join(save_dir, "b2.npy"), nn.b2)
            np.save(os.path.join(save_dir, "b3.npy"), nn.b3)
            print(f"Best weights saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}; no improvement in validation loss.")
                break

def evaluate_network(nn, X_test, Y_test, num_samples_to_display=5):
    _, _, A3, _, _, _ = nn.forward_propagation(X_test)
    predictions = np.argmax(A3, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    test_loss = nn.compute_loss(A3, Y_test)
    print("\nSample Predictions (Predicted vs True):")
    for i in range(min(num_samples_to_display, X_test.shape[0])):
        print(f"Sample {i + 1}: Predicted = {predictions[i]}, True = {true_labels[i]}")
    print(f"\nEvaluation Summary:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct Predictions: {np.sum(predictions == true_labels)} out of {X_test.shape[0]}")
    return accuracy

if __name__ == "__main__":
    X_train_full, Y_train_full, X_test, Y_test = load_processed_mnist(processed_dir=r"data\processed")
    num_train = int(0.9 * X_train_full.shape[0])  
    indices = np.random.permutation(X_train_full.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    X_train, Y_train = X_train_full[train_idx], Y_train_full[train_idx]
    X_val, Y_val = X_train_full[val_idx], Y_train_full[val_idx]
    
    save_dir = r"outputs"
    if os.path.exists(os.path.join(save_dir, "W1.npy")):
        print("Loading existing weights...")
        nn = NeuralNetwork(load_from_dir=save_dir)
    else:
        print("Initializing new weights...")
        nn = NeuralNetwork()
    
    train_network(nn, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=64, initial_lr=0.1, save_dir=save_dir)
    accuracy = evaluate_network(nn, X_test, Y_test, num_samples_to_display=5)