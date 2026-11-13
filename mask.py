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
        self.future_visits = future_visits
    
    def __len__(self):
        return len(self.future_visits)
    
    def __getitem__(self,index):
        patient,current,future=self.future_visits[index]
        currentPath=f'{patient}/w{current}'
        futurePath=f'{patient}/w{future}'
        scan=nibabel.load(f'{currentPath}/skull.nii.gz')
        scan2=nibabel.load(f'{currentPath}/mask.nii.gz')
        scan3=nibabel.load(f'{futurePath}/mask.nii.gz')
        mri=torch.from_numpy(scan.get_fdata(dtype='float32'))
        mask=torch.from_numpy(scan2.get_fdata(dtype='float16'))>0
        futureMask=torch.from_numpy(scan3.get_fdata(dtype='float16'))>0
        gap = float(future - current)
        return {
            'mri':mri.unsqueeze(0).float(),
            'mask':mask.unsqueeze(0).float(),
            'gap':torch.tensor(gap, dtype=torch.float32),
            'target':futureMask.unsqueeze(0).float(),
        }


patients = {
    'p1': [44, 56],
    'p2': [37, 47],
    'p3': [27, 38],
    'p48': [23, 45, 49],
}
future_visits = []

for patient, weeks in patients.items():
    for i in range(len(weeks) - 1):
        future_visits.append((patient, weeks[i], weeks[i + 1]))

dataset=data(future_visits)
example=dataset[0]
print(example['target'].unique(),example['mask'].unique(),example['gap'].item())
