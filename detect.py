import sys
from google.cloud import vision


def detect_labels(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    for label in response.label_annotations:
        print(f"{label.description}: {label.score:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python detect.py 画像のパス")
        sys.exit(1)
    detect_labels(sys.argv[1])