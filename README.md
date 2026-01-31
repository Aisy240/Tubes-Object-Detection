
# Sistem Deteksi Sampah (Organik, Anorganik, B3)

Proyek ini menggunakan YOLOv8 untuk mendeteksi tiga jenis sampah:
- **Organik**: Sampah biodegradable (daun, makanan, kayu, dll)
- **Anorganik**: Sampah non-biodegradable (plastik, logam, kaca, dll)
- **B3**: Sampah Berbahaya dan Beracun (kimia, elektronik, dll)

## Struktur Folder

```
yolo-sampah_detection/
├── train.py                   # Script training model
├── detect.py                  # Script deteksi gambar
├── detect_video.py            # Script deteksi video/webcam
├── requirements.txt           # Dependencies
├── README.md                  # Dokumentasi
├── .gitignore                 # Git ignore
├── dataset/
│   ├── data.yaml              # Konfigurasi dataset
│   ├── images/
│   │   ├── train/             # Gambar training
│   │   └── val/               # Gambar validasi
│   └── labels/
│       ├── train/             # Label training (format YOLO)
│       └── val/               # Label validasi
├── models/                    # Simpan model .pt di sini
├── utils/
│   ├── __init__.py
│   └── data_utils.py          # Utility functions
├── outputs/                   # Hasil deteksi
└── runs/                      # Training results (auto-generated)
```

## Installation

1. **Install Python 3.8+**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Persiapkan dataset**:
   - Letakkan gambar training di `dataset/images/train/`
   - Letakkan gambar validasi di `dataset/images/val/`
   - Buat file label YOLO format di `dataset/labels/train/` dan `dataset/labels/val/`

   Format label YOLO:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   Dimana:
   - `class_id`: 0=Organik, 1=Anorganik, 2=B3
   - Koordinat dalam format normalized (0-1)

## Penggunaan

### 1. Training Model

```bash
python train.py
```

Parameter yang bisa disesuaikan di `train.py`:
- `epochs`: Jumlah epoch (default: 100)
- `imgsz`: Ukuran input image (default: 640)
- `batch`: Batch size (default: 16)
- `device`: 0 untuk GPU, -1 untuk CPU

### 2. Deteksi pada Gambar

Deteksi single image:
```bash
python detect.py --image path/to/image.jpg
```

Batch detection:
```bash
python detect.py --image-dir path/to/images/
```

Hasil akan disimpan di folder `outputs/`

### 3. Deteksi pada Video

Deteksi video file:
```bash
python detect_video.py --video path/to/video.mp4
```

Simpan hasil ke file:
```bash
python detect_video.py --video path/to/video.mp4 --output outputs/result.mp4
```

Deteksi dari webcam:
```bash
python detect_video.py --webcam
```

Simpan webcam ke file:
```bash
python detect_video.py --webcam --output outputs/webcam_result.mp4
```

## Format Label Dataset (YOLO)

Setiap gambar harus memiliki file `.txt` dengan label. Format:

```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.15 0.2
```

Penjelasan:
- Baris pertama: Objek organik di pusat (0.5, 0.5), lebar 0.3, tinggi 0.4
- Baris kedua: Objek anorganik di posisi (0.2, 0.8), lebar 0.15, tinggi 0.2

Tools untuk membuat label:
- [Roboflow](https://roboflow.com/)
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [CVAT](https://github.com/openvinotoolkit/cvat)

## Konfigurasi Advanced

### Mengubah ukuran model YOLO

Di `train.py`, ubah line:
```python
model = YOLO('yolov8m.pt')  # 'm' bisa diganti dengan:
# 'n' - nano (paling kecil, tercepat)
# 's' - small
# 'm' - medium (recommended)
# 'l' - large
# 'x' - xlarge (paling akurat, lambat)
```

### Mengubah confidence threshold

Default threshold: 0.5 (50%)

Untuk lebih ketat (lebih sedikit false positives):
```bash
python detect.py --image path/to/image.jpg --conf 0.7
python detect_video.py --video path/to/video.mp4 --conf 0.7
```

Untuk lebih santai (mendeteksi lebih banyak):
```bash
python detect.py --image path/to/image.jpg --conf 0.3
```

## Tips Pelatihan

1. **Dataset Quality**: Pastikan dataset seimbang untuk ketiga kelas
2. **Augmentation**: Gunakan different angles, lighting, dan backgrounds
3. **Validasi Split**: Gunakan 80% training dan 20% validation
4. **Monitoring**: Lihat loss dan akurasi di `runs/detect/sampah_detection/`

## Troubleshooting

**Problem**: Out of memory saat training
- Solution: Kurangi batch size atau gunakan model lebih kecil ('n' atau 's')

**Problem**: Model tidak akurat
- Solution: Tambah lebih banyak data, atau train lebih lama dengan jumlah epochs lebih besar

**Problem**: Deteksi lambat
- Solution: Gunakan model lebih kecil, atau turunkan imgsz

## Lisensi

Proyek ini menggunakan YOLOv8 dari Ultralytics (licensed under AGPL-3.0)

## Contact

Untuk pertanyaan atau saran, silakan buat issue di repository ini.
