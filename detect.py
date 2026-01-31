"""
Script untuk deteksi sampah pada gambar statis
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


class TrashDetector:
    """Kelas untuk deteksi sampah menggunakan YOLO"""
    
    def __init__(self, model_path='runs/detect/sampah_detection/weights/best.pt'):
        """
        Inisialisasi detector
        
        Args:
            model_path: Path ke model YOLO yang sudah dilatih
        """
        self.model_path = model_path
        self.class_names = {0: 'Organik', 1: 'Anorganik', 2: 'B3'}
        self.class_colors = {
            0: (0, 255, 0),      # Organik - Hijau
            1: (0, 165, 255),    # Anorganik - Orange
            2: (0, 0, 255)       # B3 - Merah
        }
        
        # Load model
        if Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Model dimuat dari: {model_path}")
        else:
            print(f"Warning: Model tidak ditemukan di {model_path}")
            print("Menggunakan model pretrained YOLOv11m...")
            self.model = YOLO('yolov11m.pt')
    
    def detect_image(self, image_path, conf_threshold=0.5):
        """
        Deteksi sampah pada gambar
        
        Args:
            image_path: Path ke gambar
            conf_threshold: Confidence threshold untuk deteksi
            
        Returns:
            Image dengan hasil deteksi
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Tidak bisa membaca gambar {image_path}")
            return None
        
        # Deteksi
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        # Visualisasi hasil
        annotated_image = image.copy()
        
        # Statistik deteksi
        stats = {cls: 0 for cls in self.class_names.values()}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.class_names.get(class_id, 'Unknown')
                
                # Update statistik
                stats[class_name] += 1
                
                # Gambar bounding box
                color = self.class_colors.get(class_id, (255, 255, 255))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Gambar label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan statistik
        print("\n" + "=" * 50)
        print(f"Hasil Deteksi: {image_path}")
        print("=" * 50)
        print(f"Organik    : {stats['Organik']} item")
        print(f"Anorganik  : {stats['Anorganik']} item")
        print(f"B3         : {stats['B3']} item")
        print(f"Total      : {sum(stats.values())} item")
        print("=" * 50)
        
        return annotated_image
    
    def detect_batch(self, image_dir, output_dir='outputs', conf_threshold=0.5):
        """
        Deteksi sampah pada semua gambar di folder
        
        Args:
            image_dir: Folder berisi gambar
            output_dir: Folder untuk menyimpan hasil
            conf_threshold: Confidence threshold
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Cari semua gambar
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for fmt in image_formats:
            image_files.extend(image_dir.glob(fmt))
            image_files.extend(image_dir.glob(fmt.upper()))
        
        if not image_files:
            print(f"Tidak ada gambar ditemukan di {image_dir}")
            return
        
        print(f"\nDitemukan {len(image_files)} gambar")
        print("Memproses...")
        
        for img_path in image_files:
            result_image = self.detect_image(str(img_path), conf_threshold)
            if result_image is not None:
                # Simpan hasil
                output_path = output_dir / f"detected_{img_path.name}"
                cv2.imwrite(str(output_path), result_image)
                print(f"Hasil disimpan: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deteksi Sampah dengan YOLO')
    parser.add_argument('--image', type=str, help='Path ke gambar untuk deteksi')
    parser.add_argument('--image-dir', type=str, help='Folder berisi gambar untuk batch processing')
    parser.add_argument('--model', type=str, default='runs/detect/sampah_detection/weights/best.pt',
                       help='Path ke model YOLO')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs', help='Folder output')
    
    args = parser.parse_args()
    
    # Inisialisasi detector
    detector = TrashDetector(args.model)
    
    if args.image:
        # Deteksi single image
        result = detector.detect_image(args.image, args.conf)
        if result is not None:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path / 'detected_image.jpg'), result)
            print(f"\nHasil disimpan ke: {output_path / 'detected_image.jpg'}")
    
    elif args.image_dir:
        # Batch processing
        detector.detect_batch(args.image_dir, args.output, args.conf)
    
    else:
        print("Gunakan --image untuk gambar tunggal atau --image-dir untuk batch processing")
        print("\nContoh:")
        print("  python detect.py --image path/to/image.jpg")
        print("  python detect.py --image-dir path/to/images/")


if __name__ == '__main__':
    main()
