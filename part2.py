import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def stitch(left_img, right_img):
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoints(left_img, right_img)
    match_points, good_matches = match_keypoints(key_points1, key_points2, descriptor1, descriptor2)
    final_H = ransac(match_points)

    vis = visualize_matches(left_img, key_points1, right_img, key_points2, good_matches)

    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    points2 = cv2.perspectiveTransform(points, final_H)
    list_of_points = np.concatenate((points1, points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
    output_img[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img

    return output_img,vis

def get_keypoints(left_img, right_img):
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(left_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(right_img, None)
    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoints(key_points1, key_points2, descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    match_points = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            match_points.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

    return match_points, good_matches


def homography(points):
    A = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)
    return H / H[2, 2]

def ransac(good_matches, iterations=5000, threshold=5):
    best_inliers = []
    final_H = None
    for _ in range(iterations):
        random_pts = random.sample(good_matches, 4)
        H = homography(random_pts)
        inliers = []
        for pt in good_matches:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp /= Hp[2]
            dist = np.linalg.norm(p_1 - Hp)
            if dist < threshold:
                inliers.append(pt)
        if len(inliers) > len(best_inliers):
            best_inliers, final_H = inliers, H
    return final_H


def visualize_matches(img1, kp1, img2, kp2, matches):
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    return img_matches

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

if __name__ == "__main__":
    images = load_images_from_folder("./image/part2/")
    result = images[0]
    counter = 1
    rows = int(np.ceil(len(images) / 3))
    plt.figure(figsize=(15, 3*rows))

    for img in images[1:]:
        result, vis1 = stitch(result, img)
        plt.subplot(rows,3,counter)
        plt.imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
        plt.title(f'Matches with image {counter} ')
        counter+=1
        plt.axis('off')

    plt.figure(figsize=(20, 15))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Final Stitched Image')
    plt.axis('off')
    plt.show()
