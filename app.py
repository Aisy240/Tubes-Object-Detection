from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import subprocess

app = Flask(__name__)

# LOAD MODEL
model = YOLO("runs/detect/train/weights/best.pt")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULT_FOLDER, "predict"), exist_ok=True)
os.makedirs(os.path.join(RESULT_FOLDER, "video"), exist_ok=True)


# HELPER: AVI → MP4
def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path.replace(".avi", ".mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", avi_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        mp4_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp4_path


# DASHBOARD (FOTO + VIDEO)
@app.route("/", methods=["GET", "POST"])
def dashboard():
    image_result = None
    video_result = None

    # FOTO
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]

        if file.filename != "":
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            results = model.predict(
                source=input_path,
                save=True,
                project=RESULT_FOLDER,
                name="predict",
                exist_ok=True,
                conf=0.3
            )

            save_dir = results[0].save_dir
            files = os.listdir(save_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
            image_result = f"results/predict/{files[-1]}"

    #  VIDEO 
    if request.method == "POST" and "video" in request.files:
        file = request.files["video"]

        if file.filename != "":
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            model.predict(
                source=input_path,
                save=True,
                project=RESULT_FOLDER,
                name="video",
                exist_ok=True,
                conf=0.3
            )

            video_dir = os.path.join(RESULT_FOLDER, "video")
            files = os.listdir(video_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(video_dir, x)))
            latest_video = files[-1]

            # konversi ke MP4
            if latest_video.endswith(".avi"):
                avi_path = os.path.join(video_dir, latest_video)
                mp4_path = convert_avi_to_mp4(avi_path)
                video_result = "results/video/" + os.path.basename(mp4_path)
            else:
                video_result = "results/video/" + latest_video

    return render_template(
        "dashboard.html",
        image=image_result,
        video=video_result
    )


# MAIN
if __name__ == "__main__":
    app.run(debug=True)
