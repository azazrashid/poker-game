
import re
import subprocess

def extract_detected_cards(output_string):
    pattern = r'\d+ (10|[2-9JQKA]{1,2})([HCDS])'
    matches = re.findall(pattern, output_string)

    # Process the matches to create a single concatenated string
    detected_cards = ''.join([''.join(match) for match in matches])

    return detected_cards


def detect_cards(image_path: str):
    # Command to run the YOLO model prediction
    command = f'yolo task=detect mode=predict model="./yolov8s_playing_cards.pt" source="{image_path}"'

    # Run the command and capture its output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Convert the output from bytes to string
    output_string = error.decode()

    return extract_detected_cards(output_string)

