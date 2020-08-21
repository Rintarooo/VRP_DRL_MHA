# https://own-search-and-study.xyz/2017/05/18/python%E3%81%AEmatplotlib%E3%81%A7gif%E3%82%A2%E3%83%8B%E3%83%A1%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B/
from glob import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
if __name__ == "__main__":
	folderName = "../../Downloads/"
	fileName = "newplot*.png"
	if not os.path.isdir(folderName):
		raise FileNotFoundError("check folderName")
	picList = glob(folderName + fileName)
	picList = sorted(picList)
	print(picList)

	fig = plt.figure(figsize = (8.0, 7.0)) 
	ax = plt.subplot(111)
	ax.axis("off")
	fig.tight_layout()
	 
	ims = []
	#画像ファイルを順々に読み込んでいく
	for i in range(len(picList)):	 
		#1枚1枚のグラフを描き、appendしていく
		tmp = Image.open(picList[i])
		#エイリアシングを防ぐため、線形補完
		ims.append([plt.imshow(tmp, interpolation="spline36")])     
	 
	#アニメーション作成    
	ani = animation.ArtistAnimation(fig, ims, interval = 150, repeat_delay = 1000)
	ani.save("./test.gif", writer = "imagemagick")