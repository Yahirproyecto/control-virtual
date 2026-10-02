import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print('mediapipe', mp.__version__)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

img = cv2.imread('reconocimiento.py')
print('img', img)

try:
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    print('result type', type(result))
    print('result dir', [a for a in dir(result) if not a.startswith('_')])
    print('result repr', result)
    if hasattr(result, 'hand_landmarks'):
        print('hand_landmarks len', len(result.hand_landmarks))
    if hasattr(result, 'handedness'):
        print('handedness len', len(result.handedness))
    if hasattr(result, 'hand_world_landmarks'):
        print('hand_world_landmarks len', len(result.hand_world_landmarks))
    if hasattr(result, 'handednesses'):
        print('handednesses len', len(result.handednesses))
    if hasattr(result, 'handedness_l'):
        print('handedness_l', result.handedness_l)
except Exception as e:
    print('error', type(e).__name__, e)
finally:
    detector.close()
