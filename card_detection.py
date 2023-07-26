from imutils import perspective
import cv2
import math
import sys
import numpy as np

file = 'cards_02.png'

card_symbols = {
    0: "H", 1: "D", 2: "C", 3: "S"
}

if len(sys.argv) == 2:
    file = sys.argv[1]

# generate a digit set
valueSize, baseline = cv2.getTextSize('123456789VDRA',
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

# create an image for the values (0 is useless because if we find 1 it is 10)
values_img = np.zeros((valueSize[1] + 7, valueSize[0]), np.uint8)
cv2.putText(values_img, '123456789VDRA', (0, valueSize[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

symbols_img = cv2.imread('symbols.png', cv2.IMREAD_GRAYSCALE)
ret, symbols_img = cv2.threshold(symbols_img, 170, 255, cv2.THRESH_BINARY_INV)

# find contours of all values and symbols
cnts_val, hierarchy = cv2.findContours(values_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
cnts_sym, hierarchy = cv2.findContours(symbols_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# sort symbols from left to right
cnts_val = sorted(cnts_val, key=lambda c: cv2.boundingRect(c)[0])
cnts_sym = sorted(cnts_sym, key=lambda c: cv2.boundingRect(c)[0])

values = []
for c in cnts_val:
    x, y, w, h = cv2.boundingRect(c)
    values.append(values_img[y:y + h, x:x + w])

symbols = []
for c in cnts_sym:
    x, y, w, h = cv2.boundingRect(c)
    symbols.append(symbols_img[y:y + h, x:x + w])

# Load an image and detect contours to find cards
ocard = cv2.imread(file, cv2.IMREAD_COLOR)
card = ocard.copy()
gcard = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

# canny
canny = cv2.Canny(gcard, 100, 200)
# find contours of all values and symbols
cnts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

output = []

for c in cnts:
    if cv2.contourArea(c) > 500:
        # find the 4 points of the card
        epsilon = 0.15 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) == 4:
            pts1 = []
            for point in approx:
                (x, y) = point[0]
                pts1.append([x, y])

            # order the points in clockwise order and calculate width & height
            pts1_cw = perspective.order_points(np.array(pts1))
            w_top = math.sqrt((pts1_cw[0][0] - pts1_cw[1][0]) ** 2 + (
                    pts1_cw[0][1] - pts1_cw[1][1]) ** 2)
            w_bottom = math.sqrt((pts1_cw[3][0] - pts1_cw[2][0]) ** 2 + (
                    pts1_cw[3][1] - pts1_cw[2][1]) ** 2)
            width = int(max(w_top, w_bottom))
            h_left = math.sqrt((pts1_cw[0][0] - pts1_cw[3][0]) ** 2 + (
                    pts1_cw[0][1] - pts1_cw[3][1]) ** 2)
            h_right = math.sqrt((pts1_cw[1][0] - pts1_cw[2][0]) ** 2 + (
                    pts1_cw[1][1] - pts1_cw[2][1]) ** 2)
            height = int(max(h_left, h_right))

            # create a image from the corner points
            pts1 = np.array([pts1_cw[0], pts1_cw[1], pts1_cw[3], pts1_cw[2]],
                            dtype="float32")
            pts2 = np.array([[0, 0], [width, 0], [0, height], [width, height]],
                            dtype="float32")
            # fix the image perspective
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            flat_card = cv2.warpPerspective(gcard, matrix, (width, height))
            if width > height:  # landscape
                flat_card = cv2.rotate(flat_card,
                                       cv2.ROTATE_90_COUNTERCLOCKWISE)
            # thresh the extracted card
            ret, thresh = cv2.threshold(flat_card, 140, 255,
                                        cv2.THRESH_BINARY_INV)
            # extract the top left corner to keep value and symbol
            h, w = thresh.shape[:2]
            thresh = thresh[0:int(h * 0.3), 0:int(w * 0.25)]
            # find contours on the card to extract the value and symbol
            cur_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            # remove fake detection
            cur_cnts = [cc for cc in cur_cnts if cv2.contourArea(cc) > 50]
            # keep only the contour of the value and the symbol
            cur_cnts.sort(key=lambda c: cv2.boundingRect(c)[0])
            cur_cnts = cur_cnts[:2]
            # be sure to always have the value before the symbol
            cur_cnts.sort(key=lambda c: cv2.boundingRect(c)[1])

            # compare value
            brect = cv2.boundingRect(cur_cnts[0])
            x, y, w, h = brect
            roi = thresh[y:y + h, x:x + w]
            percent_white_pix = 0
            value = '?'
            for i, v in enumerate(values):
                scaled_roi = cv2.resize(roi, v.shape[:2][::-1])
                bitwise = cv2.bitwise_and(v, cv2.bitwise_xor(scaled_roi, v))
                before = np.sum(v == 255)
                matching = 100 - (np.sum(bitwise == 255) / before * 100)
                if percent_white_pix < matching:
                    percent_white_pix = matching
                    value = '123456789VDRA'[i]

            # compare symbol
            brect = cv2.boundingRect(cur_cnts[1])
            x, y, w, h = brect
            roi = thresh[y:y + h, x:x + w]
            percent_white_pix = 0
            symbol = -1
            for i, s in enumerate(symbols):
                scaled_roi = cv2.resize(roi, s.shape[:2][::-1])
                bitwise = cv2.bitwise_and(s, cv2.bitwise_xor(scaled_roi, s))
                before = np.sum(v == 255)
                matching = 100 - (np.sum(bitwise == 255) / before * 100)
                if percent_white_pix < matching:
                    percent_white_pix = matching
                    symbol = i

            card_info = {
                'value': '10' if value == '1' else value,
                'symbol': symbol
            }
            output.append(card_info)

def card_to_string(card_info, card_symbols):
    return f"{card_info['value']}{card_symbols[card_info['symbol']]}"

# Print the output
output_string = "".join(card_to_string(card_info, card_symbols) for card_info in output)
print(output_string)
