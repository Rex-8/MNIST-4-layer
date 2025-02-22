import numpy as np
import gzip
import os

def process_raw_to_npy(data_dir=r"data\raw", processed_dir=r"data\processed"):
    raw_paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    }
    processed_paths = {
        "train_images": os.path.join(processed_dir, "train_images.npy"),
        "train_labels": os.path.join(processed_dir, "train_labels.npy"),
        "test_images": os.path.join(processed_dir, "test_images.npy"),
        "test_labels": os.path.join(processed_dir, "test_labels.npy")
    }
    for path in raw_paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw file not found: {path}")
    train_images = load_images(raw_paths["train_images"])
    test_images = load_images(raw_paths["test_images"])
    train_labels = load_labels(raw_paths["train_labels"])
    test_labels = load_labels(raw_paths["test_labels"])
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)
    os.makedirs(processed_dir, exist_ok=True)
    np.save(processed_paths["train_images"], train_images)
    np.save(processed_paths["train_labels"], train_labels)
    np.save(processed_paths["test_images"], test_images)
    np.save(processed_paths["test_labels"], test_labels)
    print(f"Preprocessed data saved to {processed_dir}")

def load_processed_mnist(processed_dir=r"data\processed"):
    processed_paths = {
        "train_images": os.path.join(processed_dir, "train_images.npy"),
        "train_labels": os.path.join(processed_dir, "train_labels.npy"),
        "test_images": os.path.join(processed_dir, "test_images.npy"),
        "test_labels": os.path.join(processed_dir, "test_labels.npy")
    }
    for path in processed_paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed file not found: {path}. Run python data_loader.py first."
            )
    train_images = np.load(processed_paths["train_images"])
    train_labels = np.load(processed_paths["train_labels"])
    test_images = np.load(processed_paths["test_images"])
    test_labels = np.load(processed_paths["test_labels"])
    return train_images, train_labels, test_images, test_labels

def load_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        if rows != 28 or cols != 28:
            raise ValueError(f"Unexpected dimensions: {rows}x{cols}")
        pixels = num_images * rows * cols
        data = f.read(pixels)
        images = np.frombuffer(data, dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    return images

def load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        num_labels = int.from_bytes(f.read(4), 'big')
        data = f.read(num_labels)
        labels = np.frombuffer(data, dtype=np.uint8)
    return labels

def one_hot_encode(labels):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, 10), dtype=np.float32)
    one_hot[np.arange(num_labels), labels] = 1.0
    return one_hot

if __name__ == "__main__":
    process_raw_to_npy()
    train_images, train_labels, test_images, test_labels = load_processed_mnist()
    print("Training images shape:", train_images.shape)
    print("Training labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
    print("Sample image (first 10 pixels):", train_images[0, :10])
    print("Sample label (one-hot):", train_labels[0])