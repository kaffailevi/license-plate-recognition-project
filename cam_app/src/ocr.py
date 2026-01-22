import pytesseract
import cv2
import re

PLATE_REGEX = re.compile(r"[A-Z0-9]{4,10}")

def read_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = text.strip().replace(" ", "")
    match = PLATE_REGEX.search(text)
    return match.group(0) if match else None
