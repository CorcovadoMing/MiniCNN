import copy
import numpy as np

class Maxpooling:
    def __init__(self, kw, kh):
        self.kernel_width = kw
        self.kernel_height = kh
        self.mask = []
    
    def _op_maxpooling(self, in_map):
        output_shape = np.array(in_map.shape) / np.array([self.kernel_width, self.kernel_height])
        output = np.zeros(output_shape)
        mask = np.zeros(in_map.shape)
        for i in xrange(output_shape[0]):
            for j in xrange(output_shape[1]):
                interest = in_map[i * self.kernel_width  : i * 2 + self.kernel_width
                                ,j * self.kernel_height : j * 2 + self.kernel_height]
                output[i][j] = interest.max()
                idx = interest.argmax()
                mask[i * self.kernel_width + idx / self.kernel_width ] \
                    [j * self.kernel_height + idx % self.kernel_height] = 1
        self.mask.append(mask)
        return output

    def _forward(self, x):
        # Cache the input for backward use
        self.input = copy.deepcopy(x)
        output = []
        for i in x:
            imm_result = []
            for j in i:
                imm_result.append(self._op_maxpooling(j))
            output.append(imm_result)
        return np.array(output)
            

