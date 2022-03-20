import os 
from PIL import Image 
import torchvision.transforms as tvtf 

class ChickenDataset:
    def __init__(self, 
                 root,
                 normal_path,
                 defect_path, 
                 is_train=True):
        
        normal_list = [os.path.join(root, normal_path, x) 
                        for x in os.listdir(os.path.join(root, normal_path))]
        
        defect_list = [os.path.join(root, defect_path, x) 
                        for x in os.listdir(os.path.join(root, defect_path))]
        
        self.data = normal_list + defect_list
        self.label = [0 for _ in range(len(normal_list))] + \
                        [1 for _ in range(len(defect_list))]

        if is_train:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])

        else:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, i):
        data = self.data[i]
        lbl = self.label[i]
        img = Image.open(data).convert('RGB')
        img = self.transforms(img)
        print(img.shape)
        return img, lbl

    def __len__(self):
        return len(self.data)


""" dataset = ChickenDataset(root='./data/sample_classification',
                         normal_path='normal',
                         defect_path='defect',
                         )

dataset[0] """
        