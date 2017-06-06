class Relu:
    def __init__(self):
        pass
    
    def _forward(self, x):
        # Be careful the shape changes here, so [0] used to unwrap
        x = x * [x > 0][0]
        # Fix the -0.0 issue
        x += 0.
        return x