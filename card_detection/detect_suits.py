from card_detection.current import process
from card_detection.helper import constants
from card_detection.models import model_wrapper
from card_detection.pre import preprocess


flatten_card_set = []

# get the model corresponding to ranks and suits
modelRanks, modelSuits = (
    model_wrapper.model_wrapper(
        "card_detection/imgs/ranks",
        constants.NUM_RANKS,
        "card_detection/weights/rankWeights.h5",
    ),
    model_wrapper.model_wrapper(
        "card_detection/imgs/suits",
        constants.NUM_SUITS,
        "card_detection/weights/suitWeights.h5",
    ),
)


def predict_suit(img):

    imgResult = img.copy()
    imgResult2 = img.copy()

    # preprocess the image
    thresh = preprocess.preprocess_img(img)
    # find the set of corners that represent the cards
    four_corners_set = process.findContours(thresh, imgResult, draw=True)
    # warp the corners to form an image of the cards
    flatten_card_set = process.flatten_card(imgResult2, four_corners_set)
    # get a crop of the borders for each of the cards
    cropped_images = process.get_corner_snip(flatten_card_set)
    # isolate the rank and suits from the cards
    rank_suit_mapping = process.split_rank_and_suit(cropped_images)
    # figure out what the suits and ranks might be for the cards
    pred = process.eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits)

    return pred


def find_suits(all_images):
    suits = []
    for image in all_images:
        pred = predict_suit(img=image)
        suits.append(pred)

    return "".join(suits)
