from commons import get_tensor, get_model
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# load model
model = get_model()

# load json file
with open('cat_to_name.json') as f:
    cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
    class_to_idx = json.load(f)


def get_cat_name(image_location):

    # prediction
    model.eval()
    with torch.no_grad():
        tensor = get_tensor(image_location)
        outputs = model.forward(tensor)

    prediction = outputs.argmax(1)  # ambil nilai terbesar
    category = prediction.item()  # ambil itemnya

    topk = 3  # rank prediksi
    probabilities = torch.exp(outputs).data.to(
        torch.float32)  # convert log >> exp
    probabilities = (probabilities * 100)

    prob = torch.topk(probabilities, topk)[0].tolist()[0]  # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0]  # index

    ind = []
    for i in range(len(class_to_idx.items())):
        ind.append(list(class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(3):
        label.append(ind[index[i]])

    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    labels = cat_to_name[category]
    max_probability = "{: .2f}".format(max_probability)

    # ambil nilai array list
    classes0 = label[0]
    classes1 = label[1]
    classes2 = label[2]

    prob0 = "{: .2f}".format(prob[0])
    prob1 = "{: .2f}".format(prob[1])
    prob2 = "{: .2f}".format(prob[2])

    total = "{: .1f}".format(float(prob0)+float(prob1)+float(prob2))

    return labels, max_probability, classes0, classes1, classes2, prob0, prob1, prob2, total, image_location
