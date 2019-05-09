import matplotlib.pyplot as plt
from PIL import Image

from torchreid.transforms import RandomCenterCrop, CenterCropN

rt = RandomCenterCrop(224)
ct = CenterCropN(224, 5, 5, 2)

img = Image.open('/data/VeRi/aic19/image_train/000007.jpg').convert('RGB')
plt.imshow(img)
plt.show()
plt.imshow(ct(img))
plt.show()
