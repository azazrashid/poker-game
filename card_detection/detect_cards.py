import os
import string
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


cardW = 57
cardH = 87
cornerXmin = 2
cornerXmax = 10.5
cornerYmin = 2.5
cornerYmax = 23

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
# You shouldn't need to change this
zoom = 4
cardW *= zoom
cardH *= zoom
cornerXmin = int(cornerXmin * zoom)
cornerXmax = int(cornerXmax * zoom)
cornerYmin = int(cornerYmin * zoom)
cornerYmax = int(cornerYmax * zoom)

card_suits = ["s", "h", "d", "c"]
card_values = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

# imgW,imgH: dimensions of the generated dataset images
imgW = 720
imgH = 720


refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array(
    [[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32
)
refCornerHL = np.array(
    [
        [cornerXmin, cornerYmin],
        [cornerXmax, cornerYmin],
        [cornerXmax, cornerYmax],
        [cornerXmin, cornerYmax],
    ],
    dtype=np.float32,
)
refCornerLR = np.array(
    [
        [cardW - cornerXmax, cardH - cornerYmax],
        [cardW - cornerXmin, cardH - cornerYmax],
        [cardW - cornerXmin, cardH - cornerYmin],
        [cardW - cornerXmax, cardH - cornerYmin],
    ],
    dtype=np.float32,
)
refCorners = np.array([refCornerHL, refCornerLR])
alphamask = np.ones((cardH, cardW), dtype=np.uint8) * 255


def varianceOfLaplacian(img):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    Source: A.Rosebrock, https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


def extract_card(img, output_fn=None, min_focus=20, debug=False):
    """ """

    imgwarp = None

    # Check the image is not too blurry
    focus = varianceOfLaplacian(img)
    if focus < min_focus:
        if debug:
            print("Focus too low :", focus)
        return False, None

    # Convert in gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise-reducing and edge-preserving filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge extraction
    edge = cv2.Canny(gray, 30, 200)

    # Find the contours in the edged image
    cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We suppose that the contour with largest area corresponds to the contour delimiting the card
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # We want to check that 'cnt' is the contour of a rectangular shape
    # First, determine 'box', the minimum area bounding rectangle of 'cnt'
    # Then compare area of 'cnt' and area of 'box'
    # Both areas sould be very close
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    areaCnt = cv2.contourArea(cnt)
    areaBox = cv2.contourArea(box)
    valid = areaCnt / areaBox > 0.90

    if valid:
        # We want transform the zone inside the contour into the reference rectangle of dimensions (cardW,cardH)
        ((xr, yr), (wr, hr), thetar) = rect
        # Determine 'Mp' the transformation that transforms 'box' into the reference rectangle
        if wr > hr:
            Mp = cv2.getPerspectiveTransform(np.float32(box), refCard)
        else:
            Mp = cv2.getPerspectiveTransform(np.float32(box), refCardRot)
        # Determine the warped image by applying the transformation to the image
        imgwarp = cv2.warpPerspective(img, Mp, (cardW, cardH))
        # Add alpha layer
        imgwarp = cv2.cvtColor(imgwarp, cv2.COLOR_BGR2BGRA)

        # Shape of 'cnt' is (n,1,2), type=int with n = number of points
        # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
        cnta = cnt.reshape(1, -1, 2).astype(np.float32)
        # Apply the transformation 'Mp' to the contour
        cntwarp = cv2.perspectiveTransform(cnta, Mp)
        cntwarp = cntwarp.astype(np.int64)

        # We build the alpha channel so that we have transparency on the
        # external border of the card
        # First, initialize alpha channel fully transparent
        alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
        # Then fill in the contour to make opaque this zone of the card
        cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)

        # Apply the alphamask onto the alpha channel to clean it
        alphachannel = cv2.bitwise_and(alphachannel, alphamask)

        # Add the alphachannel to the warped image
        imgwarp[:, :, 3] = alphachannel

        # Save the image to file
        if output_fn is not None:
            cv2.imwrite(output_fn, imgwarp)

    if debug:
        cv2.imshow("Gray", gray)
        cv2.imshow("Canny", edge)
        edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_bgr, [box], 0, (0, 0, 255), 3)
        cv2.drawContours(edge_bgr, [cnt], 0, (0, 255, 0), -1)
        cv2.imshow("Contour with biggest area", edge_bgr)
        if valid:
            cv2.imshow("Alphachannel", alphachannel)
            cv2.imshow("Extracted card", imgwarp)

    return valid, imgwarp


def display_img(img, polygons=[], channels="bgr", size=9):
    """
    Function to display an inline image, and draw optional polygons (bounding boxes, convex hulls) on it.
    Use the param 'channels' to specify the order of the channels ("bgr" for an image coming from OpenCV world)
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    if channels == "bgr":  # bgr (cv2 image)
        nb_channels = img.shape[2]
        if nb_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_facecolor((0, 0, 0))
    ax.imshow(img)
    for polygon in polygons:
        # An polygon has either shape (n,2),
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape) == 3:
            polygon = polygon.reshape(-1, 2)
        patch = patches.Polygon(polygon, linewidth=1, edgecolor="g", facecolor="none")
        ax.add_patch(patch)


def give_me_filename(output_dir, ext):
    """
    Generate a random filename for the extracted card image.
    """
    return os.path.join(
        output_dir, "".join(random.choices(string.ascii_letters, k=10)) + "." + ext
    )


def extract_cards(image, min_focus=15):
    """
    Extract cards from an input image.
    If 'output_dir' is specified, the cards are saved in 'output_dir'.
    One file per card with a random file name.

    Returns list of extracted images.
    """
    # if output_dir is not None and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    imgs_list = []
    image = cv2.imread(image)
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to find contours of potential cards
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours of potential cards
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and extract cards
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # If the contour area is within a reasonable range for a card, extract it
        if 15000 < area < 1000000:
            # Create a mask for the current card contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Extract the card using the mask
            card_img = cv2.bitwise_and(image, image, mask=mask)
            focus = cv2.Laplacian(card_img, cv2.CV_64F).var()
            # Check if the extracted card has sufficient focus
            if focus > min_focus:
                imgs_list.append(card_img)

                # # Save the card image if output directory is specified
                # if output_dir is not None:
                #     output_fn = give_me_filename(output_dir, "png")
                #     cv2.imwrite(output_fn, card_img)

    return imgs_list
