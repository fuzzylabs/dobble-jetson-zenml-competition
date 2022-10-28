import jetson.inference
import jetson.utils
import argparse
from card_detection import detect_cards
from card import *
from fetch_model import fetch_onnx_from_zenml
import cv2

NET_DIR="model/"

def are_overlapping(a, b):
    ax0, ax1, ay0, ay1 = a.Left, a.Right, a.Top, a.Bottom
    bx0, bx1, by0, by1 = b.Left, b.Right, b.Top, b.Bottom

    return not (bx0 > ax1 or ax0 > bx1 or by0 > ay1 or ay0 > by1)

def remove_overlaps(detections):
    _detections = []
    for i, a in enumerate(detections):
        overlaps = False
        for b in detections[i+1:]:
            if a.ClassID == b.ClassID and are_overlapping(a, b):
                overlaps = True
                break
        if not overlaps:
            _detections += [a]

    return _detections

def detect_dobble():
    net = jetson.inference.detectNet(
        argv=[f"--model={NET_DIR}/dobble_model.onnx", f"--labels={NET_DIR}/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"],
        threshold=0.5
    )

    _input = jetson.utils.videoSource(args.source)
    if args.output is not None:
        _output = jetson.utils.videoOutput(args.output)

    while True:
        img = _input.Capture()
        numpyImg = jetson.utils.cudaToNumpy(img)
        cards = [Card.from_numpy(*x) for x in detect_cards(numpyImg)]

        for i, card in enumerate(cards):
            detections = net.Detect(card.cudaImg)
            filtered_detections = remove_overlaps(detections)
            # for x in filtered_detections:
            #     print(x)
            for x in filtered_detections:
                card.add_object(net.GetClassDesc(x.ClassID), x)
            # card.guesses = [Guess() for x in filtered_detections]
            

        # print([x.detectedObjects for x in cards])
        match_name, confidence, detections = find_match(cards)
        print("Match found: ", (match_name, confidence))
        # for x in detections:
        #     print(x)

        if _output is not None:
            for detection in detections:
                x, y, w, h = int(detection.Left), int(detection.Top), int(detection.Width), int(detection.Height)
                numpyImg = cv2.rectangle(numpyImg,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(numpyImg,match_name,(x+w+10,y+h),0,1.5,(0,255,0))
            img = jetson.utils.cudaFromNumpy(numpyImg)
            _output.Render(img)
        if not _input.IsStreaming():
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Dobble images")

    parser.add_argument("source", type=str, help="Source to detect Dobble images on. Can be an image or a video stream (either a file or a device)")
    parser.add_argument("output", type=str, nargs='?', help='Filename to output to')

    args = parser.parse_args()

    # TODO configuration to connect to the remote ZenML server using credentials
    # fetch_onnx_from_zenml() # TODO check if we already have the latest model
    detect_dobble()






