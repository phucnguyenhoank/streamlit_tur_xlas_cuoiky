import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==== Định nghĩa mô hình ====
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==== Load model ====
@st.cache_resource
def load_model():
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r"./ndsvt_models/mnist_cnn_model.pt", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ==== Hàm dự đoán và hiển thị ====
def load_and_predict():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=64, shuffle=True)
    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    return images, predicted.cpu()

# ==== Streamlit UI ====
st.title("🧠 Dự đoán chữ số MNIST bằng CNN")
st.write("Mô hình CNN đã được huấn luyện trên tập dữ liệu MNIST.")

if st.button("🎲 Tải ảnh mới"):
    st.session_state["reload"] = True

if "reload" not in st.session_state:
    st.session_state["reload"] = True

if st.session_state["reload"]:
    images, predicted = load_and_predict()
    grid_img = make_grid(images.cpu(), nrow=8, padding=2, normalize=True)
    npimg = grid_img.numpy().transpose((1, 2, 0))

    st.image(npimg, caption="Lưới ảnh MNIST 8x8", use_container_width=True)

    st.write("### 🔢 Kết quả dự đoán:")
    for i in range(8):
        row_preds = predicted[i*8:(i+1)*8].numpy()
        st.write("## " + " ".join(str(num) for num in row_preds))

    st.session_state["reload"] = False
