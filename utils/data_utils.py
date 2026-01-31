"""
Utility functions untuk persiapan dan processing data
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil


def verify_dataset_structure(dataset_path):
    """
    Verifikasi struktur dataset YOLO
    
    Args:
        dataset_path: Path ke folder dataset
        
    Returns:
        Dictionary berisi informasi dataset
    """
    dataset_path = Path(dataset_path)
    
    info = {
        'valid': True,
        'messages': [],
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
    }
    
    # Check folders
    required_folders = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    
    for folder in required_folders:
        folder_path = dataset_path / folder
        if not folder_path.exists():
            info['valid'] = False
            info['messages'].append(f"Missing folder: {folder}")
    
    # Count files
    train_img_path = dataset_path / 'images' / 'train'
    val_img_path = dataset_path / 'images' / 'val'
    train_lbl_path = dataset_path / 'labels' / 'train'
    val_lbl_path = dataset_path / 'labels' / 'val'
    
    if train_img_path.exists():
        info['train_images'] = len(list(train_img_path.glob('*.*')))
    if val_img_path.exists():
        info['val_images'] = len(list(val_img_path.glob('*.*')))
    if train_lbl_path.exists():
        info['train_labels'] = len(list(train_lbl_path.glob('*.txt')))
    if val_lbl_path.exists():
        info['val_labels'] = len(list(val_lbl_path.glob('*.txt')))
    
    # Validate pairs
    if info['train_images'] != info['train_labels']:
        info['messages'].append(
            f"Train: {info['train_images']} images but {info['train_labels']} labels"
        )
    if info['val_images'] != info['val_labels']:
        info['messages'].append(
            f"Val: {info['val_images']} images but {info['val_labels']} labels"
        )
    
    if not info['messages']:
        info['messages'].append("Dataset structure is valid!")
    
    return info


def get_image_info(image_path):
    """
    Dapatkan informasi gambar
    
    Args:
        image_path: Path ke gambar
        
    Returns:
        Dictionary berisi info gambar
    """
    image_path = Path(image_path)
    img = Image.open(image_path)
    
    return {
        'filename': image_path.name,
        'size_mb': image_path.stat().st_size / (1024 * 1024),
        'resolution': img.size,  # (width, height)
        'format': img.format,
    }


def resize_images(input_dir, output_dir, size=(640, 480)):
    """
    Resize semua gambar di folder
    
    Args:
        input_dir: Folder input
        output_dir: Folder output
        size: Ukuran target (width, height)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"Resizing {len(image_files)} images to {size}...")
    
    for img_file in image_files:
        img = Image.open(img_file)
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        output_file = output_path / img_file.name
        img_resized.save(output_file)
        print(f"  Saved: {img_file.name}")


def split_dataset(image_dir, label_dir, train_ratio=0.8):
    """
    Split dataset menjadi train dan validation
    
    Args:
        image_dir: Folder berisi semua gambar
        label_dir: Folder berisi semua label
        train_ratio: Ratio untuk training (default: 0.8 = 80%)
    """
    from sklearn.model_selection import train_test_split
    
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    
    # Get all files
    image_files = sorted([f.name for f in image_path.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not image_files:
        print("No images found!")
        return
    
    # Split
    train_files, val_files = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=42
    )
    
    # Create train/val folders
    train_img_dir = image_path.parent / 'images' / 'train'
    val_img_dir = image_path.parent / 'images' / 'val'
    train_lbl_dir = label_path.parent / 'labels' / 'train'
    val_lbl_dir = label_path.parent / 'labels' / 'val'
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    # Copy files
    print(f"Copying {len(train_files)} files to train...")
    for file in train_files:
        src_img = image_path / file
        dst_img = train_img_dir / file
        shutil.copy(src_img, dst_img)
        
        # Copy label
        label_file = file.rsplit('.', 1)[0] + '.txt'
        src_lbl = label_path / label_file
        if src_lbl.exists():
            dst_lbl = train_lbl_dir / label_file
            shutil.copy(src_lbl, dst_lbl)
    
    print(f"Copying {len(val_files)} files to validation...")
    for file in val_files:
        src_img = image_path / file
        dst_img = val_img_dir / file
        shutil.copy(src_img, dst_img)
        
        # Copy label
        label_file = file.rsplit('.', 1)[0] + '.txt'
        src_lbl = label_path / label_file
        if src_lbl.exists():
            dst_lbl = val_lbl_dir / label_file
            shutil.copy(src_lbl, dst_lbl)
    
    print("Split completed!")
    print(f"Training: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
