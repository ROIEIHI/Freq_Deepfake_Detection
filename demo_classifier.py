"""
Deepfake Detection Demo - Proof of Concept
==========================================

A user-friendly GUI to classify images as Real or Fake using a trained Xception model.
Perfect for classroom demonstrations.

Usage:
    python demo_classifier.py
    python demo_classifier.py --model path/to/best_xception.keras

Author: Deepfake Detection Project
"""

import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# Suppress TF warnings for cleaner demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (299, 299)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700


class DeepfakeDetectorApp:
    """GUI Application for Deepfake Detection Demo."""
    
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.current_image_path = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("🔍 Deepfake Detection Demo")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)
        
        # Load model
        self.load_model()
        
        # Build UI
        self.build_ui()
        
    def load_model(self):
        """Load the trained Keras model."""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nMake sure you've trained the model first using train_xception.py")
            sys.exit(1)
    
    def build_ui(self):
        """Build the user interface."""
        
        # ─── Title ─────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 Deepfake Detection System",
            font=('Helvetica', 24, 'bold'),
            fg='#e94560',
            bg='#1a1a2e'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Powered by Xception Neural Network",
            font=('Helvetica', 12),
            fg='#888888',
            bg='#1a1a2e'
        )
        subtitle_label.pack()
        
        # ─── Image Display Area ────────────────────────────────────
        self.image_frame = tk.Frame(
            self.root,
            bg='#16213e',
            width=400,
            height=400,
            relief='ridge',
            bd=3
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="Click 'Select Image' to load a face image",
            font=('Helvetica', 14),
            fg='#888888',
            bg='#16213e',
            wraplength=350
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # ─── Buttons ───────────────────────────────────────────────
        button_frame = tk.Frame(self.root, bg='#1a1a2e')
        button_frame.pack(pady=20)
        
        # Style for buttons
        style = ttk.Style()
        style.configure(
            'Custom.TButton',
            font=('Helvetica', 12, 'bold'),
            padding=10
        )
        
        select_btn = tk.Button(
            button_frame,
            text="📁 Select Image",
            font=('Helvetica', 12, 'bold'),
            bg='#0f3460',
            fg='white',
            activebackground='#1a5276',
            activeforeground='white',
            padx=20,
            pady=10,
            cursor='hand2',
            command=self.select_image
        )
        select_btn.pack(side='left', padx=10)
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔬 Analyze",
            font=('Helvetica', 12, 'bold'),
            bg='#e94560',
            fg='white',
            activebackground='#c73e54',
            activeforeground='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled',
            command=self.analyze_image
        )
        self.analyze_btn.pack(side='left', padx=10)
        
        # ─── Result Display ────────────────────────────────────────
        self.result_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.result_frame.pack(pady=10, fill='x', padx=50)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 28, 'bold'),
            fg='white',
            bg='#1a1a2e'
        )
        self.result_label.pack()
        
        # ─── Confidence Bar ────────────────────────────────────────
        self.confidence_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.confidence_frame.pack(pady=5, fill='x', padx=100)
        
        # Real label
        self.real_label = tk.Label(
            self.confidence_frame,
            text="REAL",
            font=('Helvetica', 10, 'bold'),
            fg='#2ecc71',
            bg='#1a1a2e'
        )
        self.real_label.pack(side='left')
        
        # Progress bar container
        self.bar_container = tk.Frame(
            self.confidence_frame,
            bg='#333333',
            height=30
        )
        self.bar_container.pack(side='left', fill='x', expand=True, padx=10)
        
        self.confidence_bar = tk.Frame(
            self.bar_container,
            bg='#2ecc71',
            height=30
        )
        
        # Fake label
        self.fake_label = tk.Label(
            self.confidence_frame,
            text="FAKE",
            font=('Helvetica', 10, 'bold'),
            fg='#e74c3c',
            bg='#1a1a2e'
        )
        self.fake_label.pack(side='right')
        
        # ─── Confidence Percentage ─────────────────────────────────
        self.confidence_text = tk.Label(
            self.root,
            text="",
            font=('Helvetica', 14),
            fg='#888888',
            bg='#1a1a2e'
        )
        self.confidence_text.pack(pady=5)
        
        # ─── Footer ────────────────────────────────────────────────
        footer_label = tk.Label(
            self.root,
            text="Transfer Learning with Xception | Binary Classification",
            font=('Helvetica', 10),
            fg='#555555',
            bg='#1a1a2e'
        )
        footer_label.pack(side='bottom', pady=10)
    
    def select_image(self):
        """Open file dialog to select an image."""
        filetypes = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_image_path = filepath
            self.display_image(filepath)
            self.analyze_btn.config(state='normal')
            self.result_label.config(text="")
            self.confidence_text.config(text="")
            self.confidence_bar.place_forget()
    
    def display_image(self, filepath):
        """Display the selected image in the UI."""
        try:
            # Load and resize image for display
            img = Image.open(filepath)
            img = img.convert('RGB')
            
            # Calculate resize to fit frame while maintaining aspect ratio
            display_size = 380
            img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            self.image_label.config(
                text=f"Error loading image:\n{str(e)}",
                image=''
            )
    
    def analyze_image(self):
        """Run the model prediction on the selected image."""
        if not self.current_image_path:
            return
        
        try:
            # Update UI to show processing
            self.result_label.config(text="Analyzing...", fg='#f39c12')
            self.root.update()
            
            # Load and preprocess image
            img = Image.open(self.current_image_path)
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to array and preprocess
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)  # Scale to [-1, 1]
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Interpret result (0 = Real, 1 = Fake)
            is_fake = prediction >= 0.5
            confidence = prediction if is_fake else (1 - prediction)
            
            # Update UI with result
            self.show_result(is_fake, confidence, prediction)
            
        except Exception as e:
            self.result_label.config(
                text=f"Error: {str(e)}",
                fg='#e74c3c'
            )
    
    def show_result(self, is_fake, confidence, raw_prediction):
        """Display the classification result with visual feedback."""
        
        if is_fake:
            result_text = "⚠️ FAKE DETECTED"
            result_color = '#e74c3c'
            bar_color = '#e74c3c'
        else:
            result_text = "✅ REAL IMAGE"
            result_color = '#2ecc71'
            bar_color = '#2ecc71'
        
        # Update result label
        self.result_label.config(text=result_text, fg=result_color)
        
        # Update confidence text
        self.confidence_text.config(
            text=f"Confidence: {confidence*100:.1f}% | Raw Score: {raw_prediction:.4f}"
        )
        
        # Update confidence bar
        self.bar_container.update()
        bar_width = int(self.bar_container.winfo_width() * raw_prediction)
        
        self.confidence_bar.config(bg=bar_color)
        self.confidence_bar.place(x=0, y=0, width=bar_width, height=30)
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def find_latest_model(base_dir):
    """Find the most recent model file in results directories."""
    model_path = None
    latest_time = 0
    
    # Look for xception_results_* directories
    for item in os.listdir(base_dir):
        if item.startswith('xception_results_'):
            results_dir = os.path.join(base_dir, item)
            if os.path.isdir(results_dir):
                model_file = os.path.join(results_dir, 'best_xception.keras')
                if os.path.exists(model_file):
                    file_time = os.path.getmtime(model_file)
                    if file_time > latest_time:
                        latest_time = file_time
                        model_path = model_file
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection Demo - Classify images as Real or Fake',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained model (.keras file). If not provided, searches for latest model.'
    )
    args = parser.parse_args()
    
    # Find model path
    if args.model:
        model_path = args.model
    else:
        # Try to find the latest model automatically
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = find_latest_model(script_dir)
        
        if model_path is None:
            print("=" * 60)
            print("ERROR: No trained model found!")
            print("=" * 60)
            print("\nPlease either:")
            print("  1. Train a model first using: python train_xception.py")
            print("  2. Specify model path: python demo_classifier.py --model path/to/model.keras")
            print()
            sys.exit(1)
        
        print(f"Found model: {model_path}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    # Launch app
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION DEMO")
    print("=" * 60)
    print(f"Model: {model_path}")
    print("Starting GUI...")
    print("=" * 60 + "\n")
    
    app = DeepfakeDetectorApp(model_path)
    app.run()


if __name__ == '__main__':
    main()
