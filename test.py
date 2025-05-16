from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train5/weights/best.pt")

test_images_dir = "test/images"

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)

    img = cv2.imread(image_path)

    if img is not None:
        results = model(img)  

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names

            for i in range(len(boxes)):
                box = boxes[i]
                confidence = confidences[i]
                class_id = classes[i]
                class_name = names[class_id]

                if confidence > 0.5:  
                    color = (0, 255, 0)  
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(img, label, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = os.path.join(output_dir, f"detected_{image_file}")
        cv2.imwrite(output_path, img)

        cv2.imshow("Detected Objects", img)
        cv2.waitKey(0) 
    else:
        print(f"Error: Could not read image at {image_path}")

cv2.destroyAllWindows()