import cv2
import mediapipe as mp
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


def landmark2np(hand_landmarks):
    li = []
    for j in (hand_landmarks.landmark):
        li.append([j.x, j.y, j.z])

    return np.array(li) - li[0]

def calc_score(A,B):
    x_score = manual_cos(A, B)

    A_v = np.diff(np.array(A), axis=0)
    B_v = np.diff(np.array(B), axis=0)
    v_score = manual_cos(A_v, B_v)

    A_a = np.diff(A_v, axis=0)
    B_a = np.diff(B_v, axis=0)
    a_score = manual_cos(A_a, B_a)

    print(round(x_score, 2), round(v_score, 2), round(a_score, 2))

    return [x_score, v_score, a_score]

def manual_cos(A, B):
    dot = np.sum(np.array(A)*np.array(B), axis=-1)
    A_norm = np.linalg.norm(A, axis=-1)
    B_norm = np.linalg.norm(B, axis=-1)
    cos = dot / (A_norm*B_norm+1e-7)

    return cos[1:].mean()

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    saved_array = None
    saved_landmark_array = None
    start = -100
    score = [0, 0, 0]
    now_array = []
    pose_time = 2
    counter = 0

    while True:
        _, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    saved_array = [landmark2np(hand_landmarks)]
                    saved_landmark_array = [hand_landmarks]
                    start = time.time()
                    score = [0, 0, 0]
                
                elif time.time()-start < pose_time:
                    saved_array.append(landmark2np(hand_landmarks))
                    saved_landmark_array.append(hand_landmarks)

                # cos類似度でチェック
                if saved_array is not None and time.time()-start > pose_time:
                    now_array.append(landmark2np(hand_landmarks))
                    
                    if len(now_array) > len(saved_array):
                        now_array.pop(0)
                        score = calc_score(saved_array, now_array)


        # 3s 表示
        if time.time() - start < pose_time:
            cv2.putText(img, 'now saving...', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), thickness=2)

        elif score[0]>0.91 and score[1]>0.1 and score[2]>0.1:
            saved_array = None
            now_array = []
            cv2.putText(img, 'pose!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), thickness=2)
        
        # 左上で保存したポーズを再生する
        if saved_array is not None and time.time()-start > pose_time:
            # Canvas
            p = 440
            size = (440, 440, 3)
            plot = np.zeros(size, dtype=np.uint8)

            # 再生フレームを決定
            i = counter % len(saved_landmark_array)
            plot_hand_landmark = saved_landmark_array[i]

            for i, lm in enumerate(plot_hand_landmark.landmark):
                height, width, channel = plot.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(plot, plot_hand_landmark, mp_hands.HAND_CONNECTIONS)

            img[0:height, 0:width] = plot


        cv2.imshow("Image", img)
        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()