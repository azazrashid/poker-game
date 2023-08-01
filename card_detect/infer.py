import re
import subprocess

def extract_detected_cards(output_string):
    pattern = r'\b(?!1 )(\d+ )?(10|[2-9JQKA]{1,2})([HCDS])'
    matches = re.findall(pattern, output_string)

    # Process the matches to create a single concatenated string
    cards = []
    for each in matches:
        cards.append(each[1])
        cards.append(each[2])

    cards = ''.join(card for card in cards)
    cards = cards.replace("10", "t")
    return cards[:10]


def detect_cards(image_path: str):
    # Command to run the YOLO model prediction
    command = f'yolo task=detect mode=predict model="card_detect/yolov8s_playing_cards.pt" source="{image_path}"'

    # Run the command and capture its output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Convert the output from bytes to string
    output_string = error.decode()
    print(output_string)
    return extract_detected_cards(output_string)

