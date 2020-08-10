import cv2
import numpy as np
import tensorflow as tf


def to_grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def comp(list_0):
    index = 0
    in_av = []
    for i in range(len(list_0) - 25):
        temp = list_0[i:i + 25]
        av = np.mean(temp)
        in_av += [av]
    in_av = np.array(in_av)
    max_value = np.min(in_av)
    max_index = np.argmin(in_av)
    return max_index, max_value


def find_point(vid_1_features, vid_2_features, index_1, index_2):
    F1_list = []
    for i in range(index_2, len(vid_2_features)):
        F1_list += [feature_distance(vid_1_features[index_1], vid_2_features[i])]

    F1_lowest = np.min(F1_list)
    F1_lowest_index = np.argmin(F1_list)

    F2_list = []
    for i in range(index_1, len(vid_1_features)):
        F2_list += [feature_distance(vid_2_features[index_2], vid_1_features[i])]

    F2_lowest = np.min(F2_list)
    F2_lowest_index = np.argmin(F2_list)

    if F1_lowest >= F2_lowest:
        in1 = int(F2_lowest_index) + index_1
        in2 = index_2
    else:
        in2 = int(F1_lowest_index) + index_2
        in1 = index_1

    return in1, in2


def differ(im1, im2, scale=True):
    h = np.shape(im1)[0]
    w = np.shape(im2)[1]
    diff = np.abs(im1 - im2)
    full = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            diff = np.mean(np.abs(im1[i, j] - im2[i, j]))
            full[i, j] = diff

    if scale:
        mx = np.max(full)
        full = full * (255 / mx)

    full = np.uint8(full)
    return full


def feature_distance(F1, F2):
    diff = F1 - F2
    ans = np.dot(diff, diff)
    return ans


size = (112, 112)

model = tf.keras.models.Sequential(
    [tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
     tf.keras.layers.Flatten()])

vid_source_2 = "C:/Users/Bruno/Downloads/bw.webm"
vid_source_1 = "C:/Users/Bruno/Downloads/colorized.webm"

vid_frames_1 = []
vid_frames_2 = []

large_frame_1 = []
large_frame_2 = []

count = 0
cap = cv2.VideoCapture(vid_source_1)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    vid_frames_1 += [to_grey(cv2.resize(frame, (224, 224)))]
    large_frame_1 += [cv2.resize(frame, (448, 448))]
    # Our operations on the frame come here
del cap
cap = cv2.VideoCapture(vid_source_2)
count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    vid_frames_2 += [to_grey(cv2.resize(frame, (224, 224)))]
    large_frame_2 += [cv2.resize(frame, (448, 448))]

vid_1_features = model.predict(tf.keras.applications.mobilenet_v2.preprocess_input(np.array(vid_frames_1)),
                               batch_size=64)
vid_2_features = model.predict(tf.keras.applications.mobilenet_v2.preprocess_input(np.array(vid_frames_2)),
                               batch_size=64)

index_1 = 0
index_2 = 0
count0 = 0
images = []

img_1 = []
img_2 = []

index_1, index_2 = find_point(vid_1_features, vid_2_features, index_1, index_2)

cv2.imwrite("00.jpg", large_frame_1[index_1])
cv2.imwrite("01.jpg", large_frame_2[index_2])

index_1 = 0
index_2 = 0

L1 = []
L2 = []

while True:
    count0 += 1
    if index_1 >= len(vid_1_features) - 1 or index_2 >= len(vid_2_features) - 1:
        break
    index_1, index_2 = find_point(vid_1_features, vid_2_features, index_1, index_2)
    if index_1 >= len(vid_1_features) - 1 or index_2 >= len(vid_2_features) - 1:
        break

    img1 = vid_frames_1[index_1]
    img2 = vid_frames_2[index_2]

    L1 += [large_frame_1[index_1]]
    L2 += [large_frame_2[index_2]]

    # h = cv2.hconcat([img1, img2])
    # images += [h]

    img_1 += [img1]
    img_2 += [img2]

    index_1 += 1
    index_2 += 1
    if count0 % 25 == 0:
        print(str(count0), index_1, index_2)
    if index_1 >= len(vid_1_features) - 1 or index_2 >= len(vid_2_features) - 1:
        break

del vid_frames_1
del vid_frames_2
while True:
    h = cv2.hconcat([L1[0], L2[0]])
    cv2.imshow('window', h)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

count = 0
limit = len(L1)
while True:
    h = cv2.hconcat([L1[count], L2[count]])
    count += 1
    cv2.imshow('window', h)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    if count >= limit:
        break
