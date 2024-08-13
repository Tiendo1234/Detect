import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(
    r"C:\Users\Admin\OneDrive\Documents\A - Proteus\Mang no-ron nhan tao\best.pt"
)

# Set up video capture
cap = cv2.VideoCapture(
    r"C:\Users\Admin\OneDrive\Documents\A - Proteus\Mang no-ron nhan tao\lk_doc.mp4"
)

# Get point Mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
# Define the line coordinates
"""START = sv.Point(182, 254)
END = sv.Point(462, 254)"""

# Point start and end for Line
START = sv.Point(0, 500)
END = sv.Point(1000, 500)

# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

count_res = 0
count_cap = 0
count_ind = 0
count_dio = 0
count_led = 0
count_ic = 0
count_ot = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classtrack = results[0].boxes.cls.cpu().tolist()
        """print(track_ids)
        print(classtrack)"""
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # detections = sv.Detections.from_yolov8(results[0])

        # Plot the tracks and count objects crossing the line
        for i, (box, track_id, classed) in enumerate(zip(boxes, track_ids, classtrack)):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track_ids) > 50:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Check if the object crosses the line
            if START.x < x < END.x and abs(y - START.y) < 50:
                if i < len(
                    classtrack
                ):  # Ensure index is within the bounds of classtrack
                    class_info = classtrack[i]  # Use index instead of track_id
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = {
                            "class": class_info
                        }  # Add class_info to track_id in crossed_object dict
                # count
                count_res = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 0.0
                )  # loop for crossed_object.values and get values has 'class'==0.0
                count_cap = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 1.0
                )
                count_ind = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 2.0
                )
                count_dio = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 3.0
                )
                count_led = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 4.0
                )
                count_ic = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 5.0
                )
                count_ot = sum(
                    1 for value in crossed_objects.values() if value.get("class") == 6.0
                )
                # count_all = f"Objects crossed: {len(crossed_objects)}"
                print("track_id:", track_id)
                print("class_info", class_info)
                print(crossed_objects)

                # Annotate the object as it crosses the line
                cv2.rectangle(
                    annotated_frame,
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2)),
                    (244, 0, 0),
                    2,
                )

            # Draw the line on the frame
            cv2.line(
                annotated_frame, (START.x, START.y), (END.x, END.y), (255, 255, 255), 2
            )

            # Write the count of objects on each frame
            cv2.putText(
                annotated_frame,
                f"Objects crossed: {len(crossed_objects)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Res: {count_res}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 150),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Cap: {count_cap}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 120),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Inductor: {count_ind}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 90),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Diot: {count_dio}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 60),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Led: {count_led}",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 30),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"IC: {count_ic}",
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Other: {count_ot}",
                (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 100, 100),
                2,
            )

        # cv2.imshow("RGB",cv2.resize(annotated_frame,(800,600)))
        cv2.imshow("RGB", annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        # Write the frame with annotations to the output video
    else:
        break
# Release the video capture
cap.release()
