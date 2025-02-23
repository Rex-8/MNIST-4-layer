import tkinter as tk
import numpy as np
from src.model import NeuralNetwork

class MNISTDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        
        # Load the trained model
        self.nn = NeuralNetwork(load_from_dir=r"outputs")
        
        # Canvas setup (smooth drawing surface)
        self.canvas_size = 280  # 280x280 pixels
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(pady=10)
        
        # Initialize drawing array (28x28, grayscale 0-1)
        self.grid_size = 28
        self.pixel_size = self.canvas_size // self.grid_size  # ~10 pixels per 28x28 cell
        self.drawing = np.zeros((28, 28), dtype=np.float32)
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        
        # Prediction label
        self.pred_label = tk.Label(root, text="Predicted Digit: None", font=("Arial", 14))
        self.pred_label.pack(pady=5)
        
        # Buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict, font=("Arial", 12))
        self.predict_button.pack(pady=5)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear, font=("Arial", 12))
        self.clear_button.pack(pady=5)

    def draw(self, event):
        x, y = event.x, event.y
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            radius = 7  
            self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill="black", outline=""
            )
            # Map to 28x28 grid and update nearby pixels
            row = y // self.pixel_size
            col = x // self.pixel_size
            # Affect a smaller area for precision
            for r in range(max(0, row-1), min(28, row+2)):  # 3x3 area
                for c in range(max(0, col-1), min(28, col+2)):
                    if 0 <= r < 28 and 0 <= c < 28:
                        # Grayscale increment
                        self.drawing[r, c] = min(self.drawing[r, c] + 0.5, 1.0)
                        # Update canvas with grayscale
                        gray_value = int(255 * (1 - self.drawing[r, c]))
                        color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"
                        self.canvas.create_rectangle(
                            c * self.pixel_size, r * self.pixel_size,
                            (c + 1) * self.pixel_size, (r + 1) * self.pixel_size,
                            fill=color, outline=""
                        )

    def clear(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="white", outline="")
        self.drawing = np.zeros((28, 28), dtype=np.float32)
        self.pred_label.config(text="Predicted Digit: None")

    def predict(self):
        input_vector = self.drawing.flatten().reshape(1, 784)
        _, _, A3, _, _, _ = self.nn.forward_propagation(input_vector)
        predicted_digit = np.argmax(A3)
        self.pred_label.config(text=f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDigitRecognizer(root)
    root.mainloop()