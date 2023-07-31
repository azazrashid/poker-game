import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        "10": "Ten",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }

    card_suits = {"h": "Hearts", "d": "Diamonds", "c": "Clubs", "s": "Spades"}

    def get_card_name(card):
        value_str = card[:-1]
        suit_str = card[-1]

        card_value = card_values.get(value_str, value_str)
        card_suit = card_suits.get(suit_str, suit_str)

        return f"{card_value} of {card_suit}"

    # Split the input string into two characters each
    card_str_list = [card_str[i : i + 2] for i in range(0, len(card_str), 2)]

    # Process each card and combine their names
    card_names = [get_card_name(card) for card in card_str_list]

    return ", ".join(card_names)


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
        hand = HandAnalyzer(all_suits).analyze(
            return_full_analysis=False, return_bestdisc_cnts=True
        )
        best_hand = list(hand.keys())[0]
        best_hand = best_hand.replace("X", "").replace("x", "")
        hand = card_name(best_hand)

        # Remove the image file after processing
        os.remove(image_path)

        return JSONResponse(content={"Optimal Hand to play": hand}, status_code=200)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        # Remove the image file in case of an error
        if os.path.exists(image_path):
            os.remove(image_path)
        # Return an error response if something goes wrong
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the image."
        )
