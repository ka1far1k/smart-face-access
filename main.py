import cv2
import time
from pathlib import Path


def load_detector():
    model_path = Path(cv2.data.haarcascades) / (
        "haarcascade_frontalface_default.xml"
    )

    xml_data = model_path.read_text(encoding="utf-8")

    storage = cv2.FileStorage(
        xml_data,
        cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
    )

    detector = cv2.CascadeClassifier()
    detector.read(storage.getFirstTopLevelNode())
    storage.release()

    if detector.empty():
        raise RuntimeError("Ошибка загрузки классификатора")

    return detector


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_time = time.perf_counter()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    end_time = time.perf_counter()

    inference_time = (end_time - start_time) * 1000

    return faces, inference_time


def draw_faces(frame, faces):
    for x, y, w, h in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


def draw_metrics(frame, fps, inference_time):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {inference_time:.2f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


def main():
    detector = load_detector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Ошибка подключения камеры")

    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                break

            faces, inference_time = detect_faces(frame, detector)

            draw_faces(frame, faces)

            current_time = time.perf_counter()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            draw_metrics(frame, fps, inference_time)

            cv2.imshow("Smart Face Access", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(
                "Smart Face Access", cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()