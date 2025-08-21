import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.input = input
        return np.maximum(0, input)
        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (input > 0).astype(input.dtype)
        # return super().compute_grad_input(input, grad_output)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output
        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * self.output * (1 - self.output)
        # return super().compute_grad_input(input, grad_output)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        exp_shifted = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.output
        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        batch_size, num_classes = input.shape
        grad_input = np.zeros_like(input)
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        return grad_input
        # return super().compute_grad_input(input, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        max_input = np.max(input, axis=1, keepdims=True)  # Р”Р»СЏ С‡РёСЃР»РµРЅРЅРѕР№ СЃС‚Р°Р±РёР»СЊРЅРѕСЃС‚Рё
        log_sum_exp = np.log(np.sum(np.exp(input - max_input), axis=1, keepdims=True))
        self.output = input - max_input - log_sum_exp
        return self.output
        # return super().compute_ou/tput(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        batch_size, num_features = input.shape
        grad_input = grad_output.copy()
        sum_grad_output = np.sum(grad_output, axis=1, keepdims=True)
        grad_input -= np.exp(self.output) * sum_grad_output
        return grad_input
        # return super().compute_grad_input(input, grad_output)
