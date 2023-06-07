import base64
import json
from io import BytesIO
import numpy as np
import time
import cv2
import os
import boto3

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
database = boto3.resource('dynamodb')

confidence_threshold = 0.3
nms_threshold = 0.1

labels_path = "coco.names"
cfg_path = "yolov3-tiny.cfg"
weights_path = "yolov3-tiny.weights"

def lambda_handler(event, context):
    """
    Lambda function handler to perform object detection on images and retrieve matching URLs from the database.
    """
    # Parse the event data
    event_data = event
    image_code = event_data['image']

    # Get the labels, configuration, and weights data
    labels = get_labels(labels_path)
    cfg_data = get_config(cfg_path)
    weights_data = get_weights(weights_path)

    # Perform object detection
    query_tags = start_detection(image_code, cfg_data, weights_data, labels)

    # Scan the database table
    table = database.Table('image_url_tag')
    response = table.scan()
    db_data = response['Items']
    image_urls = []

    # Iterate through the items in the database
    for element in db_data:
        url = element['s3-url']
        tags = element['tags']
        image_in_database = set(tags)
        query_from_user = set(query_tags)

        # If the user's query matches the image's tags, add the image URL to the list
        # If the user's query input is an empty list, return all URLs from the database
        if query_from_user == image_in_database:
            image_urls.append(url)

    result = {}
    if len(image_urls) > 0:
        result["links"] = image_urls
        json_result = json.dumps(result)
        return json_result
    else:
        return "No matching images."


def get_labels(labels_path):
    """
    Load the labels from a file.
    """
    with open(labels_path) as f:
        labels = f.read().strip().split("\n")
    return labels


def get_weights(weights_path):
    """
    Retrieve the weights data from an S3 bucket.
    """
    bucket = "detect123"
    key = weights_path

    response = s3_client.get_object(Bucket=bucket, Key=key)
    weights_data = response['Body'].read()
    return weights_data


def get_config(config_path):
    """
    Retrieve the configuration data from an S3 bucket.
    """
    bucket = "detect123"
    key = config_path

    response = s3_client.get_object(Bucket=bucket, Key=key)
    config_data = response['Body'].read()
    return config_data


def load_model(config_path, weights_path):
    """
    Load the YOLO model from disk.
    """
    print("[INFO] Loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net


def perform_prediction(image, net, labels):
    """
    Perform object detection on the image using the loaded YOLO model.
    """
    (H, W) = image.shape[:2]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Create a blob of the image and perform forward pass for YOLO object detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(output_layers)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    class_ids = []

    # Iterate through each output layer
    for output in layer_outputs:
        # Iterate through each detection
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    detected_objects = []

    if len(indices) > 0:
        for i in indices.flatten():
            detected_objects.append(labels[class_ids[i]])

    return detected_objects


def start_detection(image_code, cfg_data, weights_data, labels):
    """
    Perform the image detection process.
    """
    try:
        image_bytes = base64.b64decode(image_code)
        image_file = BytesIO(image_bytes)
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

        np_image = np.array(img)
        image = np_image.copy()

        net = load_model(cfg_data, weights_data)

        results = perform_prediction(image, net, labels)

        return results

    except Exception as e:
        print("Exception: {}".format(e))
