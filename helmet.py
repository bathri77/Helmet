import cv2
import numpy as np
import pygame

# Set thresholds and paths
CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "yolov4-tiny-obj.cfg"
weights = "yolov4-tiny-obj.weights"
labels = open("obj.names").read().strip().split("\n")
alarm = "ALARM.wav"
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights)

def detect():
    cap = cv2.VideoCapture(0)
    ALARM_ON = False

    # Set the window size to a medium size (e.g., 800x600)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 800, 600)

    while True:
        ret, image = cap.read()
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        layer_outputs = net.forward(ln)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        font_scale = 1
        thickness = 2

        pygame.init()
        pygame.mixer.music.load(alarm)

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                color = [int(c) for c in colors[class_ids[i]]]

                # Determine the text and color based on the class
                if labels[class_ids[i]] == "no_helmet":
                    text = f"{labels[class_ids[i]]}!!: {confidences[i]:.2f}"
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=thickness)
                else:
                    text = f"{labels[class_ids[i]]}): {confidences[i]:.2f}"
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=thickness)

                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness + 2, lineType=cv2.LINE_AA)

                # Play the alarm sound for "no_helmet"
                if labels[class_ids[i]] == "no_helmet":
                    pygame.mixer.music.play(-1)
                else:
                    pygame.mixer.music.stop()

        else:
            pygame.mixer.music.stop()

        cv2.imshow("output", image)

        if cv2.waitKey(1) == 13:
            pygame.mixer.music.stop()
            break

    cap.release()
    cv2.destroyAllWindows()

detect()
