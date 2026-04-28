import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

from model.simple_cnn import SimpleCNN



model = SimpleCNN(num_classes=10, hyper_mode=True)
dummy_input = tf.zeros((1, 28, 28, 1), dtype=tf.float32)
model(dummy_input, training=False)

checkpoint_dir = 'runs/mnist_simplecnn_hyper/checkpoints/best' 
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

class DigitRecognizerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer (SimpleCNN) & Probability Chart")
        
        self.root.geometry("650x450") 
        
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        tk.Label(left_frame, text="Draw a digit (0-9):", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.canvas = tk.Canvas(left_frame, width=280, height=280, bg='black', cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (280, 280), 'black')
        self.draw_img = ImageDraw.Draw(self.image)
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Predict", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=self.predict).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Clear / Redraw", font=("Arial", 12), command=self.clear).grid(row=0, column=1, padx=10)
        
        self.lbl_result = tk.Label(left_frame, text="Predict: ---", font=("Arial", 16, "bold"), fg="blue")
        self.lbl_result.pack()

        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(right_frame, text="Model Confidence (%)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.chart_canvas = tk.Canvas(right_frame, width=250, height=350, bg="#f0f0f0")
        self.chart_canvas.pack()
        
        self.bars = []
        self.labels = []
        self.pct_texts = []
        
        bar_height = 25
        spacing = 30
        self.max_bar_width = 180
        
        for i in range(10):
            y_pos = 20 + i * spacing
            
            lbl = self.chart_canvas.create_text(15, y_pos + bar_height/2, text=str(i), font=("Arial", 12, "bold"))
            
            bar = self.chart_canvas.create_rectangle(35, y_pos, 35, y_pos + bar_height, fill="gray", outline="")
            
            pct = self.chart_canvas.create_text(40, y_pos + bar_height/2, text="0.0%", font=("Arial", 10), anchor=tk.W)
            
            self.labels.append(lbl)
            self.bars.append(bar)
            self.pct_texts.append(pct)

    def draw(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw_img.ellipse([x-r, y-r, x+r, y+r], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 'black')
        self.draw_img = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Predict: ---")
        
        for i in range(10):
            self.chart_canvas.coords(self.bars[i], 35, 20 + i*30, 35, 20 + i*30 + 25)
            self.chart_canvas.itemconfig(self.bars[i], fill="gray")
            self.chart_canvas.itemconfig(self.pct_texts[i], text="0.0%")
            self.chart_canvas.coords(self.pct_texts[i], 40, 20 + i*30 + 12.5)

    def predict(self):
        img_resized = self.image.resize((28, 28), Image.Resampling.BICUBIC)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_batch = np.expand_dims(img_array, axis=(0, -1))
        
        logits = model(img_batch, training=False)
        probs = tf.nn.softmax(logits).numpy()[0]
        
        pred_label = np.argmax(probs)
        confidence = probs[pred_label] * 100
        
        self.lbl_result.config(text=f"Predict: {pred_label} ({confidence:.1f}%)")
        
        for i in range(10):
            prob = probs[i]
            pct_val = prob * 100
            
            bar_w = prob * self.max_bar_width
            y_pos = 20 + i * 30
            bar_height = 25
            
            self.chart_canvas.coords(self.bars[i], 35, y_pos, 35 + bar_w, y_pos + bar_height)
            
            color = "#4CAF50" if i == pred_label else "#ADD8E6"
            self.chart_canvas.itemconfig(self.bars[i], fill=color)
            
            self.chart_canvas.itemconfig(self.pct_texts[i], text=f"{pct_val:.1f}%")
            
            text_x = 35 + bar_w + 5 if bar_w > 0 else 40
            self.chart_canvas.coords(self.pct_texts[i], text_x, y_pos + bar_height/2)

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.root.mainloop()