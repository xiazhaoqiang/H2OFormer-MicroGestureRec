import sys

sys.path.extend(['../'])
from graph import tools


class Graph:
    def __init__(self, labeling_mode='spatial', num_point=25, index_mask=None):
        self.num_point = num_point
        if num_point == 25:
            self.inward_ori_index = [(3, 2), (2, 20), (20, 1), (1, 0),
                                     (0, 12), (12, 13), (13, 14), (14, 15),
                                     (0, 16), (16, 17), (17, 18), (18, 19),
                                     (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
                                     (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24)]

        elif num_point == 18:
            pass

        if index_mask is not None:
            self.inward_ori_index = [tup for tup in self.inward_ori_index if
                                     not any(item in index_mask for item in tup)]

        self.self_link = [(i, i) for i in range(num_point)]
        self.inward = [(i, j) for (i, j) in self.inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_inward(self):
        return self.inward

    def get_limb_label(self, groups):
        limb_label = []
        for tup in self.inward_ori_index:
            _, end = tup
            for label_id, group in enumerate(groups):
                if end in group:
                    limb_label.append(label_id)
        return limb_label

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_point, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A
