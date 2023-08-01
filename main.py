import base64
import os
import shutil
import traceback

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from poker_analyzer.hands_analyzer import HandAnalyzer
from card_detection.detect_suits import find_suits
from card_detection.detect_cards import extract_cards


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def card_name(card_str):
    card_values = {
        "J": "jack",
        "Q": "queen",
        "K": "king",
        "A": "ace",
    }

    card_suits = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}

    def get_card_name(card):
        value_str = card[:-1]
        suit_str = card[-1]

        card_value = card_values.get(value_str, value_str)
        card_suit = card_suits.get(suit_str, suit_str)

        return f"{card_value}_of_{card_suit}"

    # Split the input string into two characters each
    card_str_list = [card_str[i : i + 2] for i in range(0, len(card_str), 2)]

    # Process each card and combine their names
    card_names = [get_card_name(card) for card in card_str_list]

    return card_names

suits = ['H', 'C', 'D', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

@app.post("/poker-analyze/")
async def poker_analyze(image: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image
        allowed_formats = [".jpg", ".jpeg", ".png"]
        file_ext = os.path.splitext(image.filename)[1]
        if file_ext.lower() not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only image files "
                "(jpg, jpeg, png) are allowed.",
            )

        # Create a temporary directory to store the uploaded image
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded image to the temporary directory
        image_path = os.path.join(temp_dir, image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # suits = detect(image_path)
        all_cards = extract_cards(image=image_path)
        all_suits = find_suits(all_images=all_cards)
        if len(all_suits) < 10:
            all_suits = all_suits.upper()
            num_padding = 10 - len(all_suits)
            padding_suits = suits[:num_padding // 2]
            padding_ranks = ranks[:num_padding - len(padding_suits)]
            all_suits += ''.join(rank + suit for rank, suit in
                                zip(padding_ranks, padding_suits))
        hand = HandAnalyzer(all_suits).analyze(
            return_full_analysis=False, return_bestdisc_cnts=True
        )
        best_hand = list(hand.keys())[0]
        best_hand = best_hand.replace("X", "").replace("x", "")
        cards = card_name(best_hand)
        # Remove the image file after processing
        os.remove(image_path)

        image_responses = []
        for card in cards:
            image_path = os.path.join("Deck", f"{card}.png")
            if os.path.exists(image_path):
                # convert to binary
                with open(image_path, "rb") as out_file:
                    byte_arr = out_file.read()
                    byte_arr = base64.b64encode(byte_arr)
                image_responses.append(byte_arr)
        return jsonable_encoder(image_responses)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Create a traceback message
        trace = traceback.format_exc()
        # Append the traceback to the exception message
        error_msg = f"{str(e)}\n{trace}"
        print(error_msg)
        # Remove the image file in case of an error
        if os.path.exists(image_path):
            os.remove(image_path)
        # Return an error response if something goes wrong
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the image."
        )
