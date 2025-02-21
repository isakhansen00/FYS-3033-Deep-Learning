import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]) * np.sqrt(1 / sizes[i-1]))
            self.biases.append(np.zeros((sizes[i], 1)))

    def forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = self.relu(z)  # Tanh activation for hidden layers
            else:
                a = self.softmax(z)  # Softmax for output layer
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]  # Number of training examples
        dZ = self.activations[-1] - y  # Output error
        gradients = []

        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))

            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self.gradient_relu(self.z[i-1])

        return gradients[::-1]

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def relu(self, Z):
        return np.maximum(0, Z)

    def gradient_relu(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability trick
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=0))

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=0)

if __name__ == "__main__":
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

    # One-hot encode labels
    y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transpose for correct shape (features, samples)
    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

    # Define the MLP model
    input_size = X_train.shape[0]
    hidden_sizes = [16, 16]  # Hidden layers
    output_size = y_train.shape[0]
    mlp = MLP(input_size, hidden_sizes, output_size)

    # Training parameters
    num_epochs = 10000
    learning_rate = 0.01
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        outputs = mlp.forward(X_train)
        gradients = mlp.backward(X_train, y_train)
        mlp.update_parameters(gradients, learning_rate)
        loss = mlp.cross_entropy_loss(y_train, outputs)
        losses.append(loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

    # Testing
    y_pred = mlp.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=0)
    accuracy = np.mean(y_pred == y_test_labels)
    y_train_pred = mlp.predict(X_train)

    y_train_labels = np.argmax(y_train, axis=0)
    train_accuracy = np.mean(y_train_pred == y_train_labels)
    print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {accuracy:.4f}")

    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = mlp.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[0, :], X_train[1, :], c=np.argmax(y_train, axis=0), cmap=plt.cm.coolwarm, edgecolors="k", label="Train")
    plt.scatter(X_test[0, :], X_test[1, :], c=y_test_labels, cmap=plt.cm.coolwarm, marker="s", edgecolors="k", label="Test")
    plt.legend()
    plt.title(f"Decision Boundary - Test Accuracy: {accuracy:.4f}")
    plt.show()

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), losses, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()