import cv2

# Inicializar el clasificador para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma desde la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar el fotograma con los rostros detectados
    cv2.imshow('Facial Recognition', frame)

    # Salir del bucle si se presiona la tecla 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
