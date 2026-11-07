'''i used the dataset from lumiere for patients 001, 002, 003, 048 here, more info:
visits are in order: p1/w44,w56; p2/w37,w47; p3/w27,w38; p48/w23,w45,w49
(12; 10; 11; 22 then 4 weeks apart, respectively),
all scans and masks share the same mni atlas grid,
skull.nii.gz contains the skull-stripped post-contrast T1 MRI,
mask.nii.gz contains the automated DeepBraTumIA tumor labels, and
source (CC0) is https://doi.org/10.6084/m9.figshare.21249516'''

import torch, nibabel

class data(torch.utils.data.Dataset):
    def __init__(self, future_visits):
        self.pairs = future_visits
    def __len__(self):
        return len(self.future_visits)

patients = {
    'p1': [44, 56],
    'p2': [37, 47],
    'p3': [27, 38],
    'p48': [23, 45, 49],
}
future_visits = []

for patient, weeks in patients.items():
    for i in range(len(weeks) - 1):
        pairs.append((patient, weeks[i], weeks[i + 1]))

print(len(data(future_visits)))
