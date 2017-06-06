class Net:
    def __init__(self):
        self.layers = []
    
    def push(self, layer):
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()

    def debug(self):
        print self.layers