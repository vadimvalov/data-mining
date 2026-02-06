import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

class InputField:
    def __init__(self, root, on_recognize):
        self.root = root
        self.on_recognize = on_recognize
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind('<B1-Motion>', lambda e: [
            self.canvas.create_oval(e.x-7, e.y-7, e.x+7, e.y+7, fill='black', outline='black'),
            self.draw.ellipse([e.x-7, e.y-7, e.x+7, e.y+7], fill='black')
        ])
        
        tk.Button(root, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(root, text="Recognize", command=self.recognize).pack(side=tk.LEFT)
        self.label = tk.Label(root, text="-", font=("Arial", 16))
        self.label.pack()
        
    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="-")
        
    def get_image_array(self):
        img = np.array(self.image.resize((28, 28)))
        img = 255 - img
        img = img / 255.0
        return img
    
    def recognize(self):
        img = self.get_image_array()
        result = self.on_recognize(img)
        self.label.config(text=f"{result}")

if __name__ == "__main__":
    InputField(tk.Tk(), lambda img: "?").root.mainloop()