from ultralytics import YOLO
import cv2

# Load the trained model
weights_path = "runs/detect/train5/weights/best.pt"
model = YOLO(weights_path)

# Set the class names (important for interpreting predictions)
names = ['big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck', 'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
model.names = names  # While YOLO object might have names, explicitly setting it ensures consistency

# Load the video
video_path = 'videos/1stvideo.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'videos/output_video_with_counts.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Perform object detection
    results = model(frame)

    # Process the results
    detections = results.xyxy[0].cpu().numpy()

    class_counts = {name: 0 for name in names}
    if len(detections) > 0:
        for *xyxy, conf, cls in detections:
            class_id = int(cls)
            if 0 <= class_id < len(names):
                class_name = names[class_id]
                class_counts[class_name] += 1

    print(f"Frame {frame_number}:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")

    annotated_frame = frame.copy()
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{names[int(cls)]} {conf:.2f}'
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Check 'output_video_with_counts.mp4' for the video with detections (if enabled).")