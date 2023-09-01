import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import numpy as np

import cv2

# Setting model parameters
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)

N_CLASS = 4

INP_FEATURES = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(INP_FEATURES, N_CLASS)

# Loading the trained weights
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth', map_location=torch.device('cpu')))

# Setting the model to evaluation mode
model.eval()

def drawBoxes(frame, boxes, scores, labels):
    # Preprocessing
    image = frame.copy()
    
    colors = {1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 0, 0)}
    label_names = {1: 'Green', 2: 'Yellow', 3: 'Red'}  # Change these to your class names
    
    for box, label, score in zip(boxes, labels, scores):
        image = cv2.rectangle(image,
                               (box[0], box[1]),
                               (box[2], box[3]),
                               colors[label], 2)
        text = f'{label_names[label]}: {score:.2f}'
        image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
    
    return image

def filterBoxes(output,nms_th=0.3,score_threshold=0.5):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    # Non Max Supression
    mask = nms(boxes,scores,nms_th)
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels


def displayPredictions(frame, output, nms_th=0.3, score_threshold=0.5):
    
    boxes, scores, labels = filterBoxes(output, nms_th, score_threshold)
    
    # Preprocessing
    image = frame.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    colors = {1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 0, 0)}
    label_names = {1: 'Green', 2: 'Yellow', 3: 'Red'}  # Change these to your class names
    
    box_drawn = False
    if len(boxes) > 0:
        box_drawn = True

    for box, label, score in zip(boxes, labels, scores):
        image = cv2.rectangle(image,
                               (box[0], box[1]),
                               (box[2], box[3]),
                               colors[label], 2)
        text = f'{label_names[label]}: {score:.2f}'
        image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        box_drawn = True
    
    return image, box_drawn, boxes, scores, labels

def process_frame(frame, model, nms_th=0.3, score_threshold=0.4):
    # Convert the BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Increase the saturation by a factor, e.g., 1.5
    hsv[:,:,1] = hsv[:,:,1]*1.5

    # Convert the HSV image back to BGR
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert the BGR image to RGB
    frame = frame[:, :, ::-1]

    # Convert the frame to a torch tensor
    frame = torch.from_numpy(frame.copy())

    # Permute the frame to [C, H, W] from [H, W, C]
    frame = frame.permute(2, 0, 1)

    # Normalize the frame
    frame = frame / 255.0

    # Add a batch dimension
    frame = frame.unsqueeze(0)
        
    # Convert the frame to float
    frame = frame.float()

    # Run the model inference on the frame
    with torch.no_grad():
        results = model(frame)

    # Visualize the results on the frame
    annotated_frame, box_drawn, boxes, scores, labels = displayPredictions(frame[0], results[0], nms_th, score_threshold)

    return annotated_frame, box_drawn, boxes, scores, labels

# Open the video file
video_path = "test_clip_1_1.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(1)

frame_count = 0
skip_frames = 10  # Number of frames to skip before running inference

# Store the boxes, scores, and labels after every inference
last_boxes, last_scores, last_labels = [], [], []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    frame_count += 1

    if success:
        # Run inference only every `skip_frames` frames
        if frame_count % skip_frames == 0:
            # Visualize the results on the frame
            annotated_frame, box_drawn, last_boxes, last_scores, last_labels = process_frame(frame, model)

            # Display the annotated frame
            cv2.imshow("Inference", annotated_frame)

            # Freeze the frame if a box was drawn
            # if box_drawn:
            #     cv2.waitKey(0)
        else:
            # Draw the boxes on the frame
            annotated_frame = drawBoxes(frame, last_boxes, last_scores, last_labels)
            # Display the original frame
            cv2.imshow("Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
