#!/usr/bin/env python3
"""
Quick dataset verification script
Run this to verify your dataset is properly structured before training
"""

import os
import numpy as np
import cv2
from collections import Counter

def test_dataset_structure():
    """Test if dataset is properly structured"""
    
    print("🔍 Testing Dataset Structure...")
    print("=" * 50)
    
    dataset_path = "Dataset"
    images_path = os.path.join(dataset_path, "images")
    annotations_path = os.path.join(dataset_path, "annotations")
    
    # Check if folders exist
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        print("Please download and extract the dataset to 'Dataset/' folder")
        return False
    
    if not os.path.exists(images_path):
        print("❌ Images folder not found!")
        return False
    
    if not os.path.exists(annotations_path):
        print("❌ Annotations folder not found!")
        return False
    
    print("✅ Dataset folders found")
    
    # Get file counts
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.npy')]
    
    print(f"📊 Found {len(image_files)} image files")
    print(f"📊 Found {len(annotation_files)} annotation files")
    
    # Test a few samples
    test_samples = min(5, len(image_files))
    print(f"\n🧪 Testing {test_samples} samples...")
    
    valid_samples = 0
    expressions = []
    valences = []
    arousals = []
    
    for i, img_file in enumerate(image_files[:test_samples]):
        img_id = img_file.split('.')[0]
        
        try:
            # Test image loading
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not load image {img_file}")
                continue
            
            h, w, c = img.shape
            print(f"  Image {img_id}: {w}x{h}x{c}")
            
            # Test annotation loading
            exp_path = os.path.join(annotations_path, f"{img_id}_exp.npy")
            val_path = os.path.join(annotations_path, f"{img_id}_val.npy")
            aro_path = os.path.join(annotations_path, f"{img_id}_aro.npy")
            lnd_path = os.path.join(annotations_path, f"{img_id}_lnd.npy")
            
            if not all(os.path.exists(p) for p in [exp_path, val_path, aro_path, lnd_path]):
                print(f"❌ Missing annotations for {img_id}")
                continue
            
            exp = np.load(exp_path)
            val = np.load(val_path)
            aro = np.load(aro_path)
            lnd = np.load(lnd_path)
            
            # Convert to proper numeric types
            exp = float(exp)
            val = float(val)
            aro = float(aro)
            lnd = lnd.astype(np.float32)
            
            print(f"  Annotations {img_id}: exp={exp}, val={val:.3f}, aro={aro:.3f}, landmarks={lnd.shape}")
            
            # Check if valid (not -2)
            if val != -2 and aro != -2:
                valid_samples += 1
                expressions.append(int(exp))
                valences.append(float(val))
                arousals.append(float(aro))
            else:
                print(f"  ⚠️  Invalid sample (val={val}, aro={aro})")
            
        except Exception as e:
            print(f"❌ Error processing {img_id}: {e}")
    
    print(f"\n✅ Valid samples: {valid_samples}/{test_samples}")
    
    if expressions:
        print(f"\n📈 Sample Statistics:")
        print(f"Expression distribution: {dict(Counter(expressions))}")
        print(f"Valence range: [{min(valences):.3f}, {max(valences):.3f}]")
        print(f"Arousal range: [{min(arousals):.3f}, {max(arousals):.3f}]")
    
    # Class mapping
    class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    print(f"\n🏷️  Expression Classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    print("\n" + "=" * 50)
    
    if valid_samples > 0:
        print("✅ Dataset appears to be properly structured!")
        print("🚀 You can now run the main training script:")
        print("   python facial_expression_recognition.py")
        return True
    else:
        print("❌ Dataset issues detected. Please check the dataset.")
        return False

if __name__ == "__main__":
    test_dataset_structure()
