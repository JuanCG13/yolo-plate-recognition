import yolov5
import cv2
import os
import pytesseract

# Función para guardar los recortes en una carpeta sin subcarpetas
def save_crops(img, boxes, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2]
        cv2.imwrite(f"{save_dir}/crop_{i}.jpg", crop_img)

# Carga el modelo
model = yolov5.load('keremberke/yolov5m-license-plate')

# Configura los parámetros del modelo
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
custom_config = r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Abre el archivo de video
video_path = 'IMG_0719.mp4'
cap = cv2.VideoCapture(video_path)

# Configura el códec y crea el objeto VideoWriter para guardar el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la inferencia en el fotograma
    results = model(frame, size=480)

    # Analiza los resultados
    predictions = results.pred[0]
    boxes = predictions[:, :4].numpy()  # Conversión a numpy array

    # Guarda los recortes de las placas detectadas
    save_dir = 'cropped_images'
    save_crops(frame, boxes, save_dir)

    # Realiza el reconocimiento de caracteres en los recortes
    for i in range(len(boxes)):
        crop_path = f"{save_dir}/crop_{i}.jpg"
        crop_img = cv2.imread(crop_path)
        if crop_img is not None:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (3, 3))
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config=custom_config)
            cv2.putText(frame, text, (int(boxes[i][0]), int(boxes[i][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Muestra el fotograma con las placas reconocidas
    cv2.imshow('Reconocimiento de placas', frame)
    
    # Guarda el fotograma procesado en el archivo de video de salida
    out.write(frame)

    # Presiona 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
out.release()
cv2.destroyAllWindows()