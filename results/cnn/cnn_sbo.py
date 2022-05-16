import GPy
import numpy as np
import pandas as pd
import safeopt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)
# np.random.seed(42)
batch_size = 16
num_epochs = 8

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(
    root="./dataset", train=True, download=False, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=False, transform=transform
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# targets = np.array(train_dataset.targets)
# subset_indices = []
# for i in range(len(classes)):
#     subset_indices.append(np.random.choice(np.where(targets == i)[0], 1000))

# subset_indices = np.concatenate(subset_indices)
# test_set_10k = torch.utils.data.Subset(dataset=train_dataset, indices=subset_indices)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # -> n, 120
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.dropout(x)
        x = self.fc3(x)  # -> n, 10
        return x


class Net(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 512)
        self.fc2 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(0.25)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# def cnn_fit(learning_rate, momentum, batch_size, num_epochs):
def cnn_fit(learning_rate, momentum):
    global train_loader, test_loader, batch_size, num_epochs
    # model = ConvNet().to(device)
    model = Net().to(device)

    print(
        "Fitting CNN for learning_rate=%f, momentum=%f, batch_size=%d, num_epochs=%d"
        % (learning_rate, momentum, batch_size, num_epochs)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    n_total_steps = len(train_loader)
    print_at = n_total_steps // 5
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_at == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )

    print("Finished Training")
    # PATH = "./cnn.pth"
    # torch.save(model.state_dict(), PATH)

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        # n_class_correct = [0 for i in range(10)]
        # n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # for i in range(batch_size):
            #     label = labels[i]
            #     pred = predicted[i]
            #     if label == pred:
            #         n_class_correct[label] += 1
            #     n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network: {acc} %")

        # for i in range(10):
        #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        #     print(f"Accuracy of {classes[i]}: {acc} %")

        return acc


X = []


def objective_function_wrapper(X_: list):
    global X
    X_ = np.atleast_2d(X_)
    Y_ = []
    print(X_)
    # for [learning_rate, momentum, batch_size, num_epochs] in X_:
    for [learning_rate, momentum] in X_:
        learning_rate = 10 ** int(learning_rate)
        momentum = max(min(round(momentum, 2), 0.95), 0.05)
        # batch_size = int(2 ** int(batch_size))
        # num_epochs = int(num_epochs) - int(num_epochs) % 5

        # fits the network and returns the test accuracy
        # y = cnn_fit(learning_rate, momentum, batch_size, num_epochs)
        y = cnn_fit(learning_rate, momentum)
        y = np.array(y).reshape((-1, 1))
        print(y)
        Y_.append(y)
        X.append(np.array([learning_rate, momentum]))
    return np.reshape(Y_, (-1, 1))


# Measurement noise
noise_var = 0.05 ** 2

# Bounds on the inputs variable
# learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# momentum = (0.1, 0.9)
# batch_size = [4, 8, 16, 32, 64]
# num_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# bounds = [(-5, -1), (0, 1), (2, 6), (10, 50)]
bounds = [(-5, -1), (0, 1)]
# fixing batch_size=16 and num_epochs=20, less machines available
safe_threshold = 50.0
parameter_set = safeopt.linearly_spaced_combinations(bounds, 100)

# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2.0, lengthscale=1.0, ARD=True)

# Initial safe set
# x0 = np.array([[-3, 0.7, 3, 10]])
x0 = np.array([[-3, 0.7]])


# noisy true function
objective_function = objective_function_wrapper

# The statistical model of our objective function
gp = GPy.models.GPRegression(x0, objective_function(x0), kernel, noise_var=noise_var)

start_time = time.time()
# The optimization routine
# opt = safeopt.SafeOptSwarm(gp, safe_threshold, bounds=bounds, threshold=-5.0)
opt = safeopt.SafeOpt(gp, parameter_set, safe_threshold, lipschitz=None, threshold=0)


for i in range(20):
    print("iteration", i)
    # Obtain next query point
    x_next = opt.optimize()
    # print(x_next)
    # Get a measurement from the real system
    y_meas = objective_function(x_next)
    # Add this to the GP model
    opt.add_new_data_point(x_next, y_meas)
    with open("res.npy", "wb") as f:
        np.save(f, np.array(X))
        np.save(f, opt.y)

total_run_time = time.time() - start_time
# X = opt.x
X = np.array(X)
Y = opt.y
max_val = Y.max()
idx = Y.argmax()
max_point = X[idx]

objective_function_name = "cnn"

arr = []
for x, y in zip(X, Y):
    arr.append(np.append(x, y))

arr = np.array(arr)
# dimension = X.shape[1]
# label = ["learning_rate", "momentum", "batch_size", "num_epochs", "y"]
label = ["learning_rate", "momentum", "y"]
# label.extend(["x" + str(i) for i in range(dimension)])
# label.append("y")

points_df = pd.DataFrame(arr, columns=label)

points_df.to_csv(
    "./results/"
    + objective_function_name
    + "_sbo"
    + time.strftime("_%d_%b_%Y_%H_%M_%S", time.localtime())
    + ".csv",
    index=False,
)
report = "Total run time : %f\n" % total_run_time
no_points_evaluated = points_df.shape[0]
report += "Number of points evaluated : %d\n" % no_points_evaluated
no_unsafe_evaluation = points_df.y[points_df.y < safe_threshold].count()
report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
optimum_value = points_df["y"].max()
optimum_value_at = points_df.iloc[points_df["y"].idxmax()][0 : points_df.shape[1]]
report += "Optimization results\ny = %f\nat\n%s" % (
    optimum_value,
    optimum_value_at.to_string(),
)
with open(
    "./results/"
    + objective_function_name
    + "_sbo"
    + time.strftime("_%d_%b_%Y_%H_%M_%S", time.localtime())
    + ".txt",
    "w",
) as res_file:
    res_file.write(report)
