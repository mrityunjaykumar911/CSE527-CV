import requests
import cv2
import numpy as np

import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt

im1 = cv2.imdecode(
    np.frombuffer(requests.get("https://drive.google.com/uc?id=1y8eKxsxxQDDxnwZex_qNi_1QtEmr7xai").content, np.uint8),
    cv2.IMREAD_GRAYSCALE)
im2 = cv2.imdecode(
    np.frombuffer(requests.get("https://drive.google.com/uc?id=1ZRNAyo9SUeL0BcTJKFzKuEku2-YTkvA9").content, np.uint8),
    cv2.IMREAD_GRAYSCALE)
im3 = cv2.imdecode(
    np.frombuffer(requests.get("https://drive.google.com/uc?id=1DPGLB1NtZiPEhSVHnVq_1yc5d5XSCEjf").content, np.uint8),
    cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(20, 10))
plt.subplot(131), plt.imshow(im1, cmap='gray'), plt.title('image 1')
plt.subplot(132), plt.imshow(im2, cmap='gray'), plt.title('image 2')
plt.subplot(133), plt.imshow(im3, cmap='gray'), plt.title('image 3')


def get_matched_points(img1_, img2_):
    sift = cv2.xfeatures2d.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1_, None)
    kpts2, desc2 = sift.detectAndCompute(img2_, None)

    bfmatcher = cv2.BFMatcher_create()
    matches = bfmatcher.knnMatch(desc1, desc2, k=2)
    # Apply ratio test with 0.75 difference factor
    good = []
    # TODO: your code here
    for m, n in matches:
        if float(m.distance) / float(n.distance) < 0.75:
            good.append([m])  # making it subscriptable, to supress TypeError: 'cv2.DMatch' object is not subscriptable

    # for future use in the next question, align the "good" points in two arrays
    pts1, pts2 = np.array([[0, 0]] * len(good)), np.array([[0, 0]] * len(good))
    for i, match in enumerate(good):
        pts1[i] = kpts1[match[0].queryIdx].pt
        pts2[i] = kpts2[match[0].trainIdx].pt

    return pts1, pts2


im1wide = np.hstack([im1, np.zeros_like(im1)])


# a utility for warping and blending images
def warpAndblendImages_2(image1, image2, H):
    h_, w_ = image1.shape
    im2warp = cv2.warpPerspective(image2, H, (w_, h_), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    im2mask = cv2.warpPerspective(np.ones_like(image2), H, (w_, h_), flags=cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT)
    return image1 * (1.0 - im2mask) + im2warp * im2mask


# utility function to get the cylindrical warp (optimized method without for loops)
def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_, w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    # warp the image according to cylindrical coords
    return cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                     borderMode=cv2.BORDER_CONSTANT)


# get the cylindrical warping for the three inputs
h, w = im1.shape[:2]
K = np.array([[440, 0, w / 2], [0, 440, h / 2], [0, 0, 1]])  # mock intrinsics
im_cyl = [cylindricalWarp(im_, K) for im_ in [im1, im2, im3]]
im_cyl_mask = cylindricalWarp(np.ones_like(im1), K)  # mask of the warp, is needed for stitching


# find tx,ty the translation from image 1 to image 2
# TODO: complete this function
def findTranslation(im1_, im2_):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # TODO: your code
    kp1, desc1 = sift.detectAndCompute(im1_, None)
    kp2, desc2 = sift.detectAndCompute(im2_, None)

    # BFMatcher with default params
    # TODO: your code
    matcher = cv2.BFMatcher_create()
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good = []
    # TODO: your code
    for m, n in matches:
        if float(m.distance) / float(n.distance) < 0.75:
            good.append([m])

    plt.imshow(cv2.drawMatchesKnn(im1_, kp1, im2_, kp2, good, im1_.copy(), flags=2))

    # get aligned point lists
    pts1, pts2 = np.array([[0, 0]] * len(good)), np.array([[0, 0]] * len(good))
    for i, match in enumerate(good):
        pts1[i] = kp1[match[0].queryIdx].pt
        pts2[i] = kp2[match[0].trainIdx].pt

    # return the translation
    # think about the outliers!
    return pts1, pts2  # TODO: your code

# tx01,ty01 = findTranslation(im_cyl[0],im_cyl[1])
# H1,mask1 = cv2.findHomography(tx01,ty01,method=cv2.RANSAC)

r = im_cyl[0]+im_cyl[1]

plt.imshow(r)
plt.show()