from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    train_results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        device="cuda",  

    )

    metrics = model.val(device="cuda") 