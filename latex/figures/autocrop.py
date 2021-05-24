

import glob
import os

img_files = []
img_files += glob.glob('*.png')
img_files += glob.glob('*.jpg')

for fn in img_files:
    txt = 'convert '+fn+' -trim +repage '+fn
    print(txt)
    os.system(txt)

    
