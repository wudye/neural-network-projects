import os

from debugpy.server.cli import options

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import  unet
import matplotlib.pyplot as plt
from tqdm import  tqdm
from sklearn.model_selection import train_test_split
import  warnings
warnings.filterwarnings("ignore")

batch_size = 20
epochs = 10

device = 'cpu'
model = unet.Unet()
model = model.to(device)
model = torch.compile(model, backend="aot_eager")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

x_train = np.load('./data/mnist/x_train.npy')


y_train_label = np.load("./data/mnist/y_train_label.npy")


x_train_10,_, y_train_label_10, _ = train_test_split(x_train, y_train_label, test_size=0.99, random_state=42)

x_train = x_train_10
y_train_label = y_train_label_10


#mask = y_train_label <= 2
#x_train = x_train[mask]
x_tain = np.reshape(x_train, (-1, 1, 28, 28))
x_train /= 255.0

image = x_train[4]
print("Input min:", image.min().item())
print("Input max:", image.max().item())
print("Input mean:", image.mean().item())
image = np.reshape(image, (28, 28))
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.savefig('original_image.png')
plt.close()
state_dict = torch.load('./1unet.pt', map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()
image = image.reshape(1, 1, 28, 28)
image = torch.from_numpy(image).float().to(device)
with torch.no_grad():
    output = model(image)
    print("Output min:", output.min().item())
    print("Output max:", output.max().item())
    print("Output mean:", output.mean().item())
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(output, image) * 100
    print(loss)
    output = output.cpu().numpy().reshape(28, 28)
    plt.imshow(output)
    plt.title('Predicted Image')
    plt.axis('off')
    plt.savefig('predicted_image2.png')
    plt.close()

# why the output is not good? because the model is not trained well, only 10% of the data is used for training, and the model is trained for only 30 epochs, which is not enough for the model to learn the features of the data.