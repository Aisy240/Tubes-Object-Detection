"""
Script untuk deteksi sampah pada video
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time


class VideoTrashDetector:
    """Kelas untuk deteksi sampah pada video"""
    
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
        
        # Frame buffer untuk menghitung FPS
        self.frame_times = deque(maxlen=30)
    
    def process_frame(self, frame, conf_threshold=0.5):
        """
        Proses satu frame untuk deteksi
        
        Args:
            frame: Frame dari video
            conf_threshold: Confidence threshold
            
        Returns:
            Frame dengan hasil deteksi, statistik
        """
        # Deteksi
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        annotated_frame = frame.copy()
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
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Gambar label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 8),
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, stats
    
    def draw_stats(self, frame, stats, fps):
        """
        Gambar statistik pada frame
        
        Args:
            frame: Frame video
            stats: Statistik deteksi
            fps: Frame per second
            
        Returns:
            Frame dengan statistik
        """
        # Background untuk statistik
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Organik: {stats['Organik']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Anorganik: {stats['Anorganik']}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"B3: {stats['B3']}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def detect_video(self, video_path, output_path=None, conf_threshold=0.5):
        """
        Deteksi sampah pada video file
        
        Args:
            video_path: Path ke video file
            output_path: Path untuk menyimpan hasil (opsional)
            conf_threshold: Confidence threshold
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Tidak bisa membuka video {video_path}")
            return
        
        # Property video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        
        # Siapkan writer jika ada output path
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print("\nMemproses video...")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize frame jika terlalu besar
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Deteksi
                start_time = time.time()
                annotated_frame, stats = self.process_frame(frame, conf_threshold)
                process_time = time.time() - start_time
                
                # Hitung FPS
                self.frame_times.append(process_time)
                avg_time = np.mean(list(self.frame_times))
                current_fps = 1 / avg_time if avg_time > 0 else 0
                
                # Gambar statistik
                annotated_frame = self.draw_stats(annotated_frame, stats, current_fps)
                
                # Tampilkan
                cv2.imshow('Trash Detection', annotated_frame)
                
                # Simpan jika ada writer
                if writer:
                    writer.write(annotated_frame)
                
                # Progress
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count}/{total_frames} frames")
                
                # Tekan 'q' untuk stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Deteksi dihentikan oleh user")
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"\nSelesai! Total frames: {frame_count}")
        if output_path:
            print(f"Video hasil disimpan ke: {output_path}")
    
    def detect_webcam(self, output_path=None, conf_threshold=0.5):
        """
        Deteksi sampah dari webcam
        
        Args:
            output_path: Path untuk menyimpan hasil (opsional)
            conf_threshold: Confidence threshold
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak bisa membuka webcam")
            return
        
        # Set resolusi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Siapkan writer jika ada output path
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print("Webcam aktif. Tekan 'q' untuk exit...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Deteksi
                start_time = time.time()
                annotated_frame, stats = self.process_frame(frame, conf_threshold)
                process_time = time.time() - start_time
                
                # Hitung FPS
                self.frame_times.append(process_time)
                avg_time = np.mean(list(self.frame_times))
                current_fps = 1 / avg_time if avg_time > 0 else 0
                
                # Gambar statistik
                annotated_frame = self.draw_stats(annotated_frame, stats, current_fps)
                
                # Tampilkan
                cv2.imshow('Trash Detection - Webcam', annotated_frame)
                
                # Simpan jika ada writer
                if writer:
                    writer.write(annotated_frame)
                
                # Tekan 'q' untuk exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print("Webcam ditutup")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deteksi Sampah pada Video/Webcam dengan YOLO')
    parser.add_argument('--video', type=str, help='Path ke video file')
    parser.add_argument('--webcam', action='store_true', help='Gunakan webcam')
    parser.add_argument('--model', type=str, default='runs/detect/sampah_detection/weights/best.pt',
                       help='Path ke model YOLO')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Path untuk menyimpan hasil video')
    
    args = parser.parse_args()
    
    # Inisialisasi detector
    detector = VideoTrashDetector(args.model)
    
    if args.video:
        # Deteksi video file
        detector.detect_video(args.video, args.output, args.conf)
    elif args.webcam:
        # Deteksi webcam
        detector.detect_webcam(args.output, args.conf)
    else:
        print("Gunakan --video untuk video file atau --webcam untuk webcam")
        print("\nContoh:")
        print("  python detect_video.py --video path/to/video.mp4")
        print("  python detect_video.py --webcam")
        print("  python detect_video.py --webcam --output outputs/result.mp4")


if __name__ == '__main__':
    main()
