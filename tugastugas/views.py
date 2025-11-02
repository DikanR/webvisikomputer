from django.http import HttpResponse
from django.shortcuts import render
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


    # glcm = graycomatrix(gray_image_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    features = []
    for angle in angles:
        temp_array = []
        glcm = graycomatrix(gray_image_uint8, distances=distances, angles=[angle]) # ada hal redudansi kek ginian karena susunan dari graycoprops berbeda dari yang diinginkan dan pada tampilan buat django itu dia gak bisa array get by dynamic index
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']
        count = 0
        for prop in properties:
            if count == 0:
                temp_array.append(angle)
                count += 1
            feature = graycoprops(glcm, prop)
            temp_array.append(feature[0][0])
        features.append(temp_array)

    properties.insert(0, 'angle (°)')
    context = {
        'grayscaled_image': base64_data_uri,
        'properties': properties,
        'features': features
    }
    return render(request, 'tugas.html', context)