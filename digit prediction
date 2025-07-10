import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os
import tensorflow as tf

MODEL_FILENAME = 'mnist_cnn.h5'

# Train and save a simple model if not present
def train_and_save_model():
    print('Training a new model (this will take about 1 minute)...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test), verbose=2)
    model.save(MODEL_FILENAME)
    print('Model trained and saved.')

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        train_and_save_model()
    return tf.keras.models.load_model(MODEL_FILENAME)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Digit Draw & Predict')
        self.resizable(False, False)
        self.canvas_width = 200
        self.canvas_height = 200
        self.model = load_model()
        self.create_widgets()
        self.last_x, self.last_y = None, None
        self.image1 = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image1)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10, padx=10)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        self.predict_btn = tk.Button(self, text='Predict', command=self.predict)
        self.predict_btn.grid(row=1, column=0, pady=10)
        self.clear_btn = tk.Button(self, text='Clear', command=self.clear)
        self.clear_btn.grid(row=1, column=1, pady=10)

        self.result_label = tk.Label(self, text='Draw a digit and click Predict', font=('Arial', 14))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=r*2, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=r*2)
        else:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete('all')
        self.image1 = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.result_label.config(text='Draw a digit and click Predict')

    def preprocess(self):
        # Resize to 28x28, invert, normalize
        img = self.image1.copy()
        img = img.resize((28, 28), Image.LANCZOS)
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img

    def predict(self):
        img = self.preprocess()
        pred = self.model.predict(img)
        digit = np.argmax(pred)
        conf = np.max(pred)
        self.result_label.config(text=f'Prediction: {digit} (Confidence: {conf:.2f})')

if __name__ == '__main__':
    app = App()
    app.mainloop()