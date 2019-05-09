import matplotlib.pyplot as plt
from PIL import Image

from torchreid.transforms import RandomCenterCrop

rt = RandomCenterCrop(224)

img = Image.open('/data/VeRi/aic19/image_train/000007.jpg').convert('RGB')
plt.imshow(img)
plt.show()
plt.imshow(rt(img))
plt.show()
