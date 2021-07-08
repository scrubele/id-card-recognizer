import cv2

from src import TEST_FOLDER


def doNothing(x):
    pass


def createTrackbar():
    cv2.namedWindow("Track Bars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("min_blue", "Track Bars", 0, 255, doNothing)
    cv2.createTrackbar("min_green", "Track Bars", 0, 255, doNothing)
    cv2.createTrackbar("min_red", "Track Bars", 0, 255, doNothing)
    cv2.createTrackbar("max_blue", "Track Bars", 0, 255, doNothing)
    cv2.createTrackbar("max_green", "Track Bars", 0, 255, doNothing)
    cv2.createTrackbar("max_red", "Track Bars", 0, 255, doNothing)


def get_properties():
    min_blue = cv2.getTrackbarPos("min_blue", "Track Bars")
    min_green = cv2.getTrackbarPos("min_green", "Track Bars")
    min_red = cv2.getTrackbarPos("min_red", "Track Bars")

    max_blue = cv2.getTrackbarPos("max_blue", "Track Bars")
    max_green = cv2.getTrackbarPos("max_green", "Track Bars")
    max_red = cv2.getTrackbarPos("max_red", "Track Bars")
    return min_blue, min_green, min_red, max_blue, max_green, max_red


def process_loop(img, hsv_image):
    createTrackbar()
    while True:

        min_blue, min_green, min_red, max_blue, max_green, max_red = get_properties()

        mask = cv2.inRange(
            hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red)
        )

        img = cv2.bitwise_not(img, img, mask=mask)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("mask", mask)
        cv2.imshow("gray", img_gray)
        cv2.imshow("result", img)

        key = cv2.waitKey(25)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return min_blue, min_green, min_red, max_blue, max_green, max_red


def find_hsv_mask_levels(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min_blue, min_green, min_red, max_blue, max_green, max_red = process_loop(
        img, hsv_image
    )
    print(f"min_blue {min_blue}  min_green {min_green} min_red {min_red}")
    print(f"max_blue {max_blue}  max_green {max_green} max_red {max_red}")


if __name__ == "__main__":
    image_path = TEST_FOLDER + "1.jpg"
    find_hsv_mask_levels(image_path)
