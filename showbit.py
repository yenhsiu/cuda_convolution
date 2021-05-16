import numpy as np
import matplotlib.pyplot as plt

img = np.fromfile('out.bin', dtype='uint8')
print(len(img))
wh = int(len(img) ** 0.5)# 圖片邊長
#img = np.delete(img,img[-1])
img = img.reshape([wh, wh])
plt.imshow(img,cmap="gray")
print(img.shape)
plt.show()

