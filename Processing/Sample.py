import numpy as np
class Sample:
    def __init__(self, start_index, end_index, layer_ends, x_size, y_size):
        self.start_index = start_index
        self.end_index = end_index
        self.layer_ends = layer_ends
        self.x_size = x_size
        self.y_size = y_size
        self.num_layers = len(layer_ends)
        self.layer_range = self.calculate_layer_ranges()

    def calculate_layer_ranges(self):
        layer_starts = np.append([self.start_index], np.int64(np.array(self.layer_ends[:-1]) + 1))
        # print(layer_starts)
        # print(self.layer_ends)
        layer_ranges = []
        for i, layer_start in enumerate(layer_starts):
            layer_ranges.append(range(layer_start, self.layer_ends[i]))
        return layer_ranges

    def calculate_layer_view(self, layer_number, coords, weight):
        layer = self.get_background()
        xs = coords[0][self.layer_range[layer_number]] - 1
        ys = coords[1][self.layer_range[layer_number]] - 1
        layer_coords = np.vstack([xs, ys])
        for i, x in enumerate(xs):
            layer[x, ys[i]]=1*weight
        return layer
                
    def get_background(self):
        return np.zeros([self.x_size, self.y_size])

    def calculate_full_view(self, coords):
        total_view = self.get_background()
        for layer in range(0,self.num_layers):
            layer_view = self.calculate_layer_view(layer, coords, layer+1)
            total_view += layer_view
        return total_view

    def calculate_layer_data_view(self, layer_number, coords, data, channel):
        layer = self.get_background()
        xs = coords[0][self.layer_range[layer_number]] - 1
        ys = coords[1][self.layer_range[layer_number]] - 1
        layer_coords = np.vstack([xs, ys])
        for i, x in enumerate(xs):
            # print(layer_number)
            # print(self.layer_range[layer_number][i])
            layer[x, ys[i]]=1*data[channel, self.layer_range[layer_number][i]]
        return layer

    def calculate_full_data_view(self, coords, data, channel):
        total_view = self.get_background()
        for layer in range(0,self.num_layers):
            layer_view = self.calculate_layer_data_view(layer, coords, data, channel)
            total_view += layer_view
        return total_view

    def calculate_tests(self, data):
        temp = np.transpose(np.array(data[:, self.start_index:self.end_index]))
        print(np.shape(temp))
        return temp

    def calculate_answers(self, categories):
        return np.array(categories[self.start_index:self.end_index])
    
    # def calculate_layer_param_view(self, layer_number, coords, data):
    
    def calculate_view_from_params(self, coords, data, channel):
        total_view = np.zeros([max(coords[:,0]), max(coords[:,1])])
        # print(np.shape(data))
        for (i, coords) in enumerate(coords):
            # print(i, coords)
            if channel != -1:
                total_view[coords[0]-1, coords[1]-1] = data[i][channel]
            else:
                total_view[coords[0]-1, coords[1]-1] = data[i] +1
        return total_view