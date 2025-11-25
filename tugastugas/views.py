# from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import io as BytesIo # btw ternyata ini beda ama yang di skimage, apalah hitam
from PIL import Image
import base64
from sklearn.naive_bayes import CategoricalNB
import copy
# from sklearn.preprocessing import LabelEncoder

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

def tugas3(request):
    input_data_count = 0

    # Dataset
    properties = {
        "int": "Usia",
        "float": "Pendapatan", # float ama int sementara ini dianggap sama int adja dulu soalnya kebutuhannya masih belum kompleks
        "bool": "Pelajar",
        "classification": "Beli Laptop" # gibberish banget, tapi ini supaya pengecualian adja
        } # index akhir itu klasifikasinya, dan pakai key ini buat if di template nya nanti
    
    input_fields = copy.deepcopy(properties)
    input_fields.popitem() # ngehapus klasifikasi index terakhir

    # classification = ["Ya", "Tidak"] # klo yang ini dibuat dinamis nanti lumayan berat runtimenya soalnya array data itu di-loop, terus dilist setiap string yang berbeda di index terakhirnya
    data = [
        ["Muda", "Rendah", "Ya", "Tidak"],
        ["Muda", "Rendah", "Tidak", "Tidak"],
        ["Muda", "Sedang", "Tidak", "Ya"],
        ["Tengah", "Rendah", "Ya", "Ya"],
        ["Tengah", "Sedang", "Tidak", "Ya"],
        ["Tengah", "Tinggi", "Ya", "Ya"],
        ["Tua", "Sedang", "Tidak", "Tidak"],
        ["Tua", "Tinggi", "Tidak", "Ya"],
        ["Muda", "Tinggi", "Ya", "Ya"],
        ["Tua", "Rendah", "Tidak", "Tidak"],
        ["Tengah", "Tinggi", "Tidak", "Ya"],
        ["Muda", "Sedang", "Ya", "Ya"],
    ]

    cols = list(zip(*data))
    features_cols = cols[:-1]
    categories = [sorted(set(col)) for col in features_cols]

    # Pisahkan fitur dan target
    X = [row[:-1] for row in data]  # Semua kolom kecuali terakhir
    y = [row[-1] for row in data]   # Kolom terakhir

    # Encode fitur kategori menjadi angka
    encoders = [LabelEncoder() for _ in range(len(X[0]))]

    X_encoded = np.column_stack([
        encoders[i].fit_transform([row[i] for row in X])
        for i in range(len(X[0]))
    ])

    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)

    # Model
    model = CategoricalNB()
    model.fit(X_encoded, y_encoded)

    # Data untuk prediksi
    result = []
    if request.method == "POST":
        if request.POST.get('input_data_count') is not None and request.POST.get('input_data_count') != '':
            input_data_count = int(request.POST.get('input_data_count'))
            
        # test = [["Muda", "Sedang", "Ya"], ["Tua", "Rendah", "Ya"]]
        test = []
        if request.POST.getlist('items[]') is not None:
            test = request.POST.getlist('items[]')
            test = [test[i:i+3] for i in range(0, len(test), len(input_fields))]
            print(test)
            if len(test) > 0:
                test_encoded = np.column_stack([
                    encoders[i].transform([row[i] for row in test])
                    for i in range(len(test[0]))
                ])

                print(test_encoded)

                pred = model.predict(test_encoded)
                pred_proba = model.predict_proba(test_encoded)
                class_labels = y_encoder.inverse_transform(model.classes_)

                # print(pred_proba)
                # print(pred)

                # Tampilkan hasil prediksi
                index = 0
                for t, p in zip(test, pred):
                    # result.append([
                    #     t,
                    #     [pred_proba[index], y_encoder.inverse_transform([p])[0]]
                    # ])
                    result.append(t)
                    result[index].append([
                        y_encoder.inverse_transform([p])[0],
                        dict(zip(class_labels, pred_proba[index]))
                    ])
                    index += 1
                    # print("Input:", t, "=> Prediksi:", y_encoder.inverse_transform([p])[0])
    
    print(result)
    # print(properties.popitem())
    
    # print(range(input_data_count))

    # for t, p, predicted in zip(test, pred_proba, pred):
    #     print("Input:", t)
    #     print("  Prediksi:", y_encoder.inverse_transform([predicted])[0])
    #     # tampilkan probabilitas untuk setiap kelas
    #     for cls_label, prob in zip(class_labels, p):
    #         print(f"  P({cls_label} | x) = {prob:.4f}")
    #     print()

    context = {
        'test_inputs': 'test',
        'properties': properties,
        'input_fields': input_fields,
        'categories': categories,
        'input_data_count': input_data_count,
        'input_data_count_range': range(input_data_count),
        'features_and_classification': data,
        'result': result
    }
    return render(request, 'tugas3.html', context)