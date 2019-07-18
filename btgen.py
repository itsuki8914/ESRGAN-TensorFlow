import glob
import cv2
import numpy as np
import time
class BatchGenerator:
    def __init__(self, img_size, LRDir, HRDir, aug=False):
        self.LRPath = glob.glob(LRDir + "/*.png")
        self.HRPath = glob.glob(HRDir + "/*.png")
        print("read images")
        start = time.time()
        self.LRImages = [cv2.imread(img_path) for img_path in self.LRPath]
        self.HRImages = [cv2.imread(img_path) for img_path in self.HRPath]
        print("%.4f sec took reading"%(time.time()-start))
        #for i,j in enumerate(self.LRImages):
        #    y = self.LRImages[i].shape[0]
        #    x = self.LRImages[i].shape[1]
        #    Y = self.HRImages[i].shape[0]
        #    X = self.HRImages[i].shape[1]
        #    if x*4!=X or y*4!=Y:
        #        print(i)

        #self.orgSize = (218,173)
        self.LRSize = (img_size,img_size)
        self.HRSize = (img_size*4,img_size*4)
        self.datalen = len(self.LRPath)
        print("{} has {} files".format(LRDir, self.datalen))
        self.aug = aug
        assert len(self.LRPath) == len(self.HRPath)
        assert self.LRSize[0]==self.LRSize[1]

    def augment(self, img_x, img_y):
        #print(img_y.shape)
        rand = np.random.rand()
        if rand > .5:
            img_x = cv2.flip(img_x,0)
            img_y = cv2.flip(img_y,0)
        rand = np.random.rand()
        if rand > .5:
            img_x = cv2.flip(img_x,1)
            img_y = cv2.flip(img_y,1)
        """
        rand = np.random.rand()
        if rand > 2/3:
            angle = 90
        elif rand < 1/3:
            angle = -90
        else:
            angle = 0
        scale = 1.0
        center_x = (int(self.LRSize[1]/2), int(self.LRSize[0]/2))
        trans = cv2.getRotationMatrix2D(center_x, angle , scale)
        img_x = cv2.warpAffine(img_x, trans, (self.LRSize[1],self.LRSize[0]))
        center_y = (int(self.HRSize[1]/2), int(self.HRSize[0]/2))
        trans = cv2.getRotationMatrix2D(center_y, angle , scale)

        img_y = cv2.warpAffine(img_y, trans, (self.HRSize[1],self.HRSize[0]))
        """
        return img_x, img_y

    def crop(self, img_x, img_y):
        #print(img_y.shape)
        h, w = img_x.shape[:2]
        x = np.random.randint(0, w - self.LRSize[1]-1)
        y = np.random.randint(0, h - self.LRSize[0]-1)
        new_x = img_x[y:y+self.LRSize[0], x:x+self.LRSize[1]]
        new_y = img_y[y*4:y*4+self.HRSize[0], x*4:x*4+self.HRSize[1]]
        #print(new_y.shape)
        return new_x, new_y

    def getBatch(self, bs):
        id = np.random.choice(range(self.datalen),bs)
        x   = np.zeros( (bs,self.LRSize[0],self.LRSize[1],3), dtype=np.float32)
        y   = np.zeros( (bs,self.HRSize[0],self.HRSize[1],3), dtype=np.float32)
        for i,j in enumerate(id):

            img_lr = self.LRImages[j]
            img_hr = self.HRImages[j]
            #print(self.LRPath[j],self.HRPath[j])
            #cv2.imwrite("b.png",img_lr)
            #cv2.imwrite("a.png",img_hr)
            #print(img_hr.shape)
            img_x , img_y = self.crop(img_lr, img_hr)
            #cv2.imwrite("a.png",img_y)
            if self.aug :
                img_x, img_y = self.augment(img_x, img_y)
            img_x = cv2.resize(img_x,self.LRSize,interpolation = cv2.INTER_CUBIC)
            img_y = cv2.resize(img_y,self.HRSize,interpolation = cv2.INTER_CUBIC)
            x[i,:,:,:] = (img_x - 127.5) / 127.5
            y[i,:,:,:] = (img_y - 127.5) / 127.5


        return x, y

if __name__ == '__main__':
    import os
    import math
    def tileImage(imgs):
        d = int(math.sqrt(imgs.shape[0]-1))+1
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        r = np.zeros((h*d,w*d,3),dtype=np.float32)
        for idx,img in enumerate(imgs):
            idx_y = int(idx/d)
            idx_x = idx-idx_y*d
            r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
        return r
    def foloderLength(folder):
        dir = folder
        paths = os.listdir(dir)
        return len(paths)
    data_dir = "data"
    batchgen = BatchGenerator(96,"LR","HR",True)
    batch_images_x, batch_images_t = batchgen.getBatch(4)
    x = tileImage(batch_images_x)
    x = (x + 1)*127.5
    t = tileImage(batch_images_t)
    t = (t + 1)*127.5

    cv2.imwrite("test_x.png",x)
    cv2.imwrite("test_t.png",t)
