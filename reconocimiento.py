import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.PAUSE = 0.0 # Desactivar la función de seguridad de PyAutoGUI
#variables globales
# Guardar la posición previa por cada mano (índice 0 = mano 1, índice 1 = mano 2)
previous_positions = [None, None]
Umbral_movimiento = 35 #sensibilidad

def detectar_movimiento_mano(hand_landmarks, width, height, prev_pos):
    """Devuelve (direccion, (curr_x, curr_y)). prev_pos puede ser None o (x,y)."""
    nudillo_9 = hand_landmarks[9]
    curr_x = int(nudillo_9.x * width)
    curr_y = int(nudillo_9.y * height)

    direccion = "parado"

    if prev_pos is not None:
        anterior_x, anterior_y = prev_pos
        delta_x = curr_x - anterior_x
        delta_y = curr_y - anterior_y

        if abs(delta_x) > abs(delta_y):
            if delta_x > Umbral_movimiento:
                direccion = "derecha"
            elif delta_x < -Umbral_movimiento:
                direccion = "izquierda"
        else:
            if delta_y > Umbral_movimiento:
                direccion = "abajo"
            elif delta_y < -Umbral_movimiento:
                direccion = "arriba"

    return direccion, (curr_x, curr_y)

     
def detectar_gestos_de_manos(hand_landmarks):
    puntas_dedos =[8, 12, 16, 20]
    nudillos_dedos = [6, 10, 14, 18]
    dedos_doblados = 0
    for punta, nudillo in zip(puntas_dedos, nudillos_dedos):
        if hand_landmarks[punta].y > hand_landmarks[nudillo].y:
            dedos_doblados +=1
    if dedos_doblados == 4:
        return "Puño cerrado"
    elif dedos_doblados == 0:
        return "Mano abierta"
    return "Desconocido"


def draw_hand_boxes(frame, hand_landmarks_list):
    height, width = frame.shape[:2]
    boxes = []
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        xs = [int(min(max(lm.x, 0.0), 1.0) * width) for lm in hand_landmarks]
        ys = [int(min(max(lm.y, 0.0), 1.0) * height) for lm in hand_landmarks]
        if not xs or not ys:
            continue

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        boxes.append((x_min, y_min, x_max, y_max))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Mano {idx + 1}",
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for x, y in zip(xs, ys):
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return boxes


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# Conexión con la webcam local (0 = cámara predeterminada)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam. Comprueba que esté conectada y disponible.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        boxes = draw_hand_boxes(frame, detection_result.hand_landmarks)
        
        # Para cada mano detectada, calcular y mostrar su estado
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            estado_mano = detectar_gestos_de_manos(hand_landmarks)
            
            # 1. Dirección y Lógica del Teclado
            # Solo la mano derecha (asumida como idx == 0) activa las flechas.
            height, width = frame.shape[:2]
            if idx == 0:
                prev = previous_positions[0] if 0 < len(previous_positions) and previous_positions[0] is not None else None
                direccion_mano, curr_pos = detectar_movimiento_mano(hand_landmarks, width, height, prev)

                # Asegurar que la lista tenga el tamaño necesario
                if 0 >= len(previous_positions):
                    previous_positions.extend([None] * (0 - len(previous_positions) + 1))
                previous_positions[0] = curr_pos

                # -- LÓGICA DEL TECLADO SOLO PARA MANO DERECHA --
                if estado_mano == "Puño cerrado":
                    if direccion_mano == "derecha":
                        pyautogui.press('right')
                        print(f"Mano {idx+1} Presionando: Derecha")
                    elif direccion_mano == "izquierda":
                        pyautogui.press('left')
                        print(f"Mano {idx+1} Presionando: Izquierda")
                    elif direccion_mano == "arriba":
                        pyautogui.press('up')
                        print(f"Mano {idx+1} Presionando: Arriba")
                    elif direccion_mano == "abajo":
                        pyautogui.press('down')
                        print(f"Mano {idx+1} Presionando: Abajo")
            else:
                # Mano izquierda: solo detectada, no se determina dirección
                direccion_mano = "N/A"
                # Registrar posición para evitar errores en índices y para mostrar en pantalla
                nudillo_9 = hand_landmarks[9]
                curr_pos = (int(nudillo_9.x * width), int(nudillo_9.y * height))
                if idx >= len(previous_positions):
                    previous_positions.extend([None] * (idx - len(previous_positions) + 1))
                previous_positions[idx] = curr_pos

            # 2. Calcular dónde se pondrá el texto (Esto faltaba en tu código)
            if idx < len(boxes):
                x_min, y_min, x_max, y_max = boxes[idx]
                text_x = x_min
                text_y = max(y_min - 25, 10)
            else:
                text_x, text_y = 50, 50 + idx * 30

            # 3. Dibujar el texto en pantalla
            cv2.putText(
                frame,
                f"Estado Mano {idx+1}: {estado_mano} | {direccion_mano}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    # 4. Limpiar la memoria SOLO si no hay ninguna mano en pantalla
    else:
        # resetear posiciones previas para todas las manos
        previous_positions = [None] * len(previous_positions)

    cv2.imshow("Deteccion de manos - MediaPipe Moderno", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()