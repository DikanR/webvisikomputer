from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import io as BytesIo # btw ternyata ini beda ama yang di skimage, apalah hitam
from PIL import Image
import base64

def index(request):
    gray_image = io.imread('static/images/maju-lo-rasa-malas.jpg', as_gray=True)
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)

    # Bagian ini buat ngerender imagenya sebagai base64 karena klo imagenya disimpan itu buang buang resourcec
    pil_img = Image.fromarray(gray_image_uint8, mode='L')
    buffer = BytesIo.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    base64_data_uri = f"data:image/png;base64,{base64_string}"  

    distances = [1]
    angles = [0, 90]
    if request.method == "POST":
        angles = list(map(int, request.POST.getlist('degrees')))


    # glcm = graycomatrix(gray_image_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    features = []
    properties = ['contrast', 'homogeneity', 'correlation', 'energy']
    if angles:
        for angle in angles:
            temp_array = []
            glcm = graycomatrix(gray_image_uint8, distances=distances, angles=[angle]) # ada hal redudansi kek ginian karena susunan dari graycoprops berbeda dari yang diinginkan dan pada tampilan buat django itu dia gak bisa array get by dynamic index
            count = 0
            for prop in properties:
                if count == 0:
                    temp_array.append(angle)
                    count += 1
                feature = graycoprops(glcm, prop)
                temp_array.append(feature[0][0])
            features.append(temp_array)

        properties.insert(0, 'angle (°)')
    
    print(angles)
    context = {
        'grayscaled_image': base64_data_uri,
        'properties': properties,
        'features': features,
        'angles': angles
    }
    return render(request, 'tugas.html', context)

def tugas2(request):
    # https://www.kaggle.com/datasets/uciml/iris
    df = pd.read_csv("static/other/Iris.csv")

    properties = df.columns[1:6].values
    x = df.iloc[:, 1:5].values
    y = df.iloc[:, -1].values

    # print(properties)
    # print(x)
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    classification_result = []
    if request.method == "POST":
        sample = [request.POST.getlist('features[]')]
        if sample[0][0] != '' and sample[0][-1] != '':
            # sample = [[5.1, 3.5, 1.4, 0.2]]
            sample_scaled = scaler.transform(sample)
            prediction = knn.predict(sample_scaled)

            classification_result = sample[0]
            classification_result.append(prediction[0])
            print("Classification result:", classification_result)

    features_and_classification = np.array([np.append(row, val) for row, val in zip(x, y)]) # ini buat ngegabungin fitur sama classnya buat ditampilin di template jadi [0, 0, 0, 0, 'ini-class']

    context = {
        'input_fields': properties[:-1],
        'properties': properties,
        'features_and_classification': features_and_classification,
        'classification_result': classification_result
    }
    return render(request, 'tugas2.html', context)