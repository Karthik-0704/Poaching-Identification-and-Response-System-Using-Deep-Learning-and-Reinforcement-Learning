from ultralytics import YOLO
import os 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    model = YOLO("yolo11m.pt")
    print("Training Started\n")
    model.train(data = "./data.yaml", imgsz = 640, batch = 8, epochs = 100)