import torch
import numpy as np 
import scipy 
import os

class ChangeDetection:

    def __init__(self) -> None:
        # load display features from files
        pass

    def __len__(self) -> int:
        # return number of trials
        pass

    def get_trial(self, index):
        # return a dictionary
        # :features: A pair of numpy arrays of shape feature_shape
        # :displays: A pair of Tensors of shape (3, 224, 224)
        # :changed: boolean
        pass

    def feature_to_display(self, features):
        # return a 3 * 224 * 224 image from given features
        pass

    def feature_to_changed_displays(self, features):
        # return n images that have exactly one feature changed
        pass

def get_experiment(exp_name):
    if exp_name == 'RedBlue':
        return RedBlueCircles()
    elif exp_name == 'BlackWhite':
        return BlackWhitePatches()
    elif exp_name == 'ColoredSquares':
        return ColoredSquares()
    elif exp_name == 'ColoredSquares_SetSize':
        return ColoredSquares(load_paths=None)
    else:
        raise NotImplementedError(f'{exp_name} not implemented')

class RedBlueCircles(ChangeDetection):

    def __init__(self, load_path='E1/1A_Data_RedBlue') -> None:
        self.trials = scipy.io.loadmat(os.path.join(load_path, 'displays.mat'))['displays'][0]
        self.behaviors = scipy.io.loadmat(os.path.join(load_path, 'subjectData.mat'))['subjectData'][0]
        assert len(self.trials) == len(self.behaviors)

        self.feature_shape = (5, 5)
        self.image_size = 224

        patch_size = 44
        radius = 20

        # each patch is 44 * 44, so there is a 2 pixel margin
        centers = [(patch_size * (i + 0.5) + 1.5, patch_size * (j + 0.5) + 1.5) for i in range(5) for j in range(5)]
        self.circles = []
        for center in centers:
            circle = np.zeros((224, 224), dtype=bool)
            for i in range(224):
                for j in range(224):
                    if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                        circle[i, j] = 1
            self.circles.append(circle)

    def __len__(self) -> int:
        return len(self.trials)
    
    def get_trial(self, index):
        trial = self.trials[index]
        features = trial[0], trial[1]
        displays = self.feature_to_display(trial[0]), self.feature_to_display(trial[1])
        changed = trial[2]
        return {'features': features, 'displays': displays, 'changed': changed}
    
    def feature_to_display(self, features):
        # features: 5 * 5, 1 for red, 0 for blue
        # return a 3 * 224 * 224 image from given features
        red = torch.Tensor((246, 32, 30)) / 255
        blue = torch.Tensor((27, 37, 252)) / 255
        image = torch.zeros(3, self.image_size, self.image_size)
        features = features.reshape(-1)
        for i in range(25):
            image += (red if features[i] else blue).reshape(3, 1, 1) * self.circles[i]
        return image
    
    def feature_to_changed_displays(self, features):
        # features: 5 * 5, 1 for red, 0 for blue
        # return n images that have exactly one feature changed
        red = torch.Tensor((246, 32, 30)) / 255
        blue = torch.Tensor((27, 37, 252)) / 255
        features = features.reshape(-1)
        images = []
        for i in range(25):
            image = torch.zeros(3, self.image_size, self.image_size)
            features[i] = 1 - features[i]
            for j in range(25):
                image += (red if features[j] else blue).reshape(3, 1, 1) * self.circles[j]
            features[i] = 1 - features[i]
            images.append(image)
        images = torch.stack(images)
        return images
    
class BlackWhitePatches(ChangeDetection):

    def __init__(self, load_path='E1/1B_Data_BlackWhite') -> None:
        self.trials = scipy.io.loadmat(os.path.join(load_path, 'displays.mat'))['displays'][0]
        self.behaviors = scipy.io.loadmat(os.path.join(load_path, 'subjectData.mat'))['subjectData'][0]
        assert len(self.trials) == len(self.behaviors)

        self.feature_shape = (5, 5)
        self.image_size = 150
        self.patch_size = 30

    def __len__(self) -> int:
        return len(self.trials)
    
    def get_trial(self, index):
        trial = self.trials[index]
        features = trial[0], trial[1]
        displays = self.feature_to_display(trial[0]), self.feature_to_display(trial[1])
        changed = trial[2]
        return {'features': features, 'displays': displays, 'changed': changed}
    
    def feature_to_display(self, features):
        # features: 5 * 5, 1 for white, 0 for black
        # return a 3 * 150 * 150 image from given features
        image = torch.from_numpy(features.astype(np.float32))
        image = image.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1).unsqueeze(0).repeat(3, 1, 1)
        return image
    
    def feature_to_changed_displays(self, features):
        # features: 5 * 5, 1 for white, 0 for black
        # return n images that have exactly one feature changed
        images = []
        for i in range(25):
            image = torch.from_numpy(features.astype(np.float32))
            image[i // 5, i % 5] = 1 - image[i // 5, i % 5]
            image = image.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1).unsqueeze(0).repeat(3, 1, 1)
            images.append(image)
        images = torch.stack(images)
        return images

colors = torch.tensor([
        (128, 128, 128),
        (246, 32, 30),
        (27, 37, 252),
        (148, 61, 201),
        (77, 201, 61),
        (254, 253, 50),
        (0, 0, 0),
        (255, 255, 255),
    ]) / 255

class ColoredSquares(ChangeDetection):

    def __init__(self, load_paths=['E2/2A_Data', 'E2/2B_Data_Patterns']) -> None:
        self.trials = []
        self.behaviors = []

        if load_paths is not None:
            for load_path in load_paths:
                self.trials.append(scipy.io.loadmat(os.path.join(load_path, 'displays.mat'))['displays'][0])
                self.behaviors.append(scipy.io.loadmat(os.path.join(load_path, 'subjectData.mat'))['subjectData'][0])
            
            self.trials = np.concatenate(self.trials)
            self.behaviors = np.concatenate(self.behaviors)
            assert len(self.trials) == len(self.behaviors)
        else:
            self.trial = []
            for set_size in [1, 2, 3, 4, 8, 12]:
                for n in range(200):
                    positions = np.array([(i, j) for i in range(4) for j in range(5)])
                    square_positions = np.random.choice(len(positions), set_size, replace=False)
                    square_positions = positions[square_positions]
                    colors = np.random.choice(7, set_size, replace=True) + 1
                    features = np.zeros((4, 5), dtype=int)
                    for i in range(set_size):
                        features[square_positions[i][0], square_positions[i][1]] = colors[i]

                    changed_color = colors[0]
                    while changed_color == colors[0]:
                        changed_color = np.random.choice(7, 1, replace=True) + 1
                    changed_features = features.copy()
                    changed_features[square_positions[0][0], square_positions[0][1]] = changed_color

                    self.trials.append([features, features, False])
                    self.trials.append([features, changed_features, True])
                    self.behaviors.append(0)
                    self.behaviors.append(1)

        self.feature_shape = (4, 5)
        self.image_size = 100
        self.patch_size = 10

    def __len__(self) -> int:
        return len(self.trials)
    
    def get_trial(self, index):
        trial = self.trials[index]
        features = trial[0], trial[1]
        displays = self.feature_to_display(trial[0]), self.feature_to_display(trial[1])
        changed = trial[-1]
        return {'features': features, 'displays': displays, 'changed': changed}
    
    def feature_to_display(self, features):
        # features: 4 * 5, 0 - 7 for colors
        # return a 3 * 100 * 100 image from given features
        image = torch.empty(3, self.image_size, self.image_size)
        # color 0 is background
        image[:, :, :] = colors[0].reshape(3, 1, 1)
        for i in range(4):
            for j in range(5):
                image[:, i * 20 + 15: i * 20 + 25, j * 20 + 5: j * 20 + 15] = colors[features[i, j]].reshape(3, 1, 1)
        return image
    
    def feature_to_changed_displays(self, features):
        images = []
        for i in range(4):
            for j in range(5):
                if features[i, j] == 0:
                    continue
                for k in range(1, 8):
                    if k == features[i, j]:
                        continue
                    image = torch.empty(3, self.image_size, self.image_size)
                    image[:, :, :] = colors[0].reshape(3, 1, 1)
                    for i_ in range(4):
                        for j_ in range(5):
                            if i_ == i and j_ == j:
                                image[:, i_ * 20 + 15: i_ * 20 + 25, j_ * 20 + 5: j_ * 20 + 15] = colors[k].reshape(3, 1, 1)
                            else:
                                image[:, i_ * 20 + 15: i_ * 20 + 25, j_ * 20 + 5: j_ * 20 + 15] = colors[features[i_, j_]].reshape(3, 1, 1)

                    images.append(image)
        images = torch.stack(images)
        return images

if __name__ == '__main__':
    exp = get_experiment('BlackWhite')
    import matplotlib.pyplot as plt
    for i in [29, 37]:
        img = exp.feature_to_display(exp.get_trial(i)['features'][0])
        changed = exp.feature_to_display(exp.get_trial(i)['features'][1])
        
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

        plt.imshow(changed.permute(1, 2, 0))
        plt.show()
        plt.close()

    exit(0)

    imgs = exp.feature_to_changed_displays(exp.get_trial(1)['features'][1])

    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    plt.imshow(changed.permute(1, 2, 0))
    plt.show()

    for img in imgs[::5]:
        plt.imshow(img.permute(1, 2, 0))
        plt.show()