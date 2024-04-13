import cv2
from PIL import Image, ImageGrab
import numpy as np


def get_image_from_clipboard():
    im = ImageGrab.grabclipboard()
    if isinstance(im, Image.Image):
        image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        return image
    else:
        return None


def get_two_image_from_image(image):
    # 只支持2560x1440的分辨率
    # 图片大小：874x875
    # 图片1位置：301 312
    # 图片2位置：1376 312
    image1 = image[312:1187, 301:1175]
    image2 = image[312:1187, 1376:2250]
    return image1, image2


def get_diff(image1, image2):
    diff = cv2.absdiff(image1, image2)
    diff = np.max(diff, axis=2) != 0
    diff = diff.astype(np.uint8)
    loc = diff == 1
    diff[loc] = 255
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for _ in range(5):
        diff = cv2.erode(diff, kernel)
    # 膨胀
    for _ in range(4):
        diff = cv2.dilate(diff, kernel)
    # 找出连通域的大小
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8, ltype=cv2.CV_32S
    )
    # 找出最大的3个连通域
    max_3 = []
    for i in range(3):
        max_label = np.argmax(stats[1:, 4]) + 1
        max_3.append(max_label)
        stats[max_label, 4] = 0
    # 将不是最大的连通域置为0
    for i in range(1, num_labels):
        if i not in max_3:
            loc = labels == i
            diff[loc] = 0
    return diff


def mark(image2, diff):
    # 轮廓
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        img = cv2.drawContours(image2, [contour], -1, (0, 0, 255), 2)
    return img


if __name__ == "__main__":
    image = get_image_from_clipboard()
    (image1, image2) = get_two_image_from_image(image)
    diff = get_diff(image1, image2)
    img = mark(image2, diff)
    cv2.imwrite("mark.png", img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
