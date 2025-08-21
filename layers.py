import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = np.dot(input, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output
        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        return np.dot(grad_output, self.weight)
        # return super().compute_grad_input(input, grad_output)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += np.dot(grad_output.T, input)
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)
        # super().update_grad_parameters(input, grad_output)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None 
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:

        if self.training:
            self.mean = np.mean(input, axis=0)
            self.var = np.var(input, axis=0)

            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var

            self.norm_input = (input - self.mean) / np.sqrt(self.var + self.eps)
        else:
            self.norm_input = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        


        if self.affine:
            return self.norm_input * self.weight + self.bias
        return self.norm_input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # print("compute_grad_input called")
        if self.training:
            batch_size = input.shape[0]
            std_inv = 1.0 / np.sqrt(self.var + self.eps)

            grad_norm_input = grad_output * self.weight if self.affine else grad_output
            grad_var = np.sum(grad_norm_input * (input - self.mean) * -0.5 * std_inv**3, axis=0)
            

            grad_mean = np.sum(grad_norm_input * -std_inv, axis=0) + grad_var * np.mean(-2.0 * (input - self.mean), axis=0)

            grad_input = grad_norm_input * std_inv + grad_var * 2.0 * (input - self.mean) / batch_size + grad_mean / batch_size
            return grad_input
        else:

        
            batch_size = input.shape[0]
            
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            
            grad_norm_input = grad_output * self.weight if self.affine else grad_output

            grad_input = grad_norm_input * std_inv
            
            return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        if self.affine:
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        РџСЂСЏРјРѕР№ РїСЂРѕС…РѕРґ
        :param input: Р’С…РѕРґРЅРѕР№ РјР°СЃСЃРёРІ РїСЂРѕРёР·РІРѕР»СЊРЅРѕР№ С„РѕСЂРјС‹
        :return: Р’С‹С…РѕРґРЅРѕР№ РјР°СЃСЃРёРІ С‚РѕР№ Р¶Рµ С„РѕСЂРјС‹
        """
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape).astype(input.dtype)
            return input * self.mask / (1 - self.p)
        else:
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        РћР±СЂР°С‚РЅС‹Р№ РїСЂРѕС…РѕРґ
        :param input: Р’С…РѕРґРЅРѕР№ РјР°СЃСЃРёРІ
        :param grad_output: Р“СЂР°РґРёРµРЅС‚ РїРѕ РІС‹С…РѕРґСѓ
        :return: Р“СЂР°РґРёРµРЅС‚ РїРѕ РІС…РѕРґСѓ
        """
        if self.training:
        # РћР±СЂР°С‚РЅС‹Р№ РїСЂРѕС…РѕРґ: РїСЂРёРјРµРЅСЏРµРј РјР°СЃРєСѓ
            return grad_output * self.mask / (1 - self.p)
        else:
            return grad_output
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

    


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        self.inputs = [input]  # РЎРѕС…СЂР°РЅСЏРµРј РІС…РѕРґС‹ РґР»СЏ РєР°Р¶РґРѕРіРѕ РјРѕРґСѓР»СЏ
        for module in self.modules:
            input = module.forward(input)
            self.inputs.append(input)  # РЎРѕС…СЂР°РЅСЏРµРј РІС‹С…РѕРґ С‚РµРєСѓС‰РµРіРѕ РјРѕРґСѓР»СЏ РєР°Рє РІС…РѕРґ РґР»СЏ СЃР»РµРґСѓСЋС‰РµРіРѕ
        return input

        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        for i, module in enumerate(reversed(self.modules)):
            grad_output = module.backward(self.inputs[-(i+2)], grad_output)
        return grad_output
        # return super().compute_grad_input(input, grad_output)

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
