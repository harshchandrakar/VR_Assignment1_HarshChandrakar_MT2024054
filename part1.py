import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def detect_and_count_coins(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    smooth = cv.GaussianBlur(gray, (3, 3), 1)
    edges = cv.Canny(smooth, 50, 150)
    thresh = cv.adaptiveThreshold(edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 10)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    min_area = 4400
    valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    coins_outlined = image.copy()
    segmentation_mask = np.zeros_like(gray)
    individual_segments = []

    for contour in valid_contours:
        cv.drawContours(coins_outlined, [contour], -1, (0, 255, 0), 5)
        individual = np.zeros_like(gray)
        cv.drawContours(individual, [contour], -1, (255), -1)
        segmentation_mask = cv.add(segmentation_mask, individual)
        coin_segment = cv.bitwise_and(image, image, mask=individual)
        individual_segments.append(coin_segment)

    coin_count = len(valid_contours)
    print(f"Total number of coins detected: {coin_count}")

    rows = int(np.ceil(coin_count / 3))
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle("ALL INDIVIDUAL COINS")
    for idx, segment in enumerate(individual_segments):
        plt.subplot(rows, 3, idx + 1)
        plt.imshow(segment)
        plt.title(f'Coin {idx + 1}', pad=10, size=12)
        plt.axis('off')

    plt.figure(figsize=(15, 10))
    plt.suptitle("ALL COINS TOGETHER")
    plt.subplot(231)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(thresh, cmap='gray')
    plt.title('Adaptive Thresholding')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(coins_outlined)
    plt.title('Coins Outlined')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return coin_count, coins_outlined, segmentation_mask


if __name__ == "__main__":
    image_path = "image/part1/coins.jpg"
    count, outlined_image, segmentation = detect_and_count_coins(image_path)