
#
# Pytorch Lightning BarlowModel (which encapsulates a TinyBarlowTwinsModel
#

from TinyBarlowTwinsModel import *
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import cv2

def write_image(fname, img) :
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname, img)

    
class BarlowModel(pl.LightningModule):
    def __init__(self, ds_train, args):
        super(BarlowModel, self).__init__()
        self.batch_size = args.batch_size
        self.lr = 0.0001
        self.betas = [0.5, 0.999]
        self.args = args
        self.c_matrix = None 
        self.model = TinyBarlowTwins(self.args) 
        self.trainset = ds_train 
        
    def forward(self, y1, y2):
        return self.model(y1, y2)
    
    def __shared_step(self, batch):
        y1, y2, _ = batch
        loss, diag, off_diag, c = self.forward(y1, y2)
        return loss, diag, off_diag, c
        
    def training_step(self, batch, batch_idx) :
        loss, diag, off_diag, c = self.__shared_step(batch)
        self.c_matrix = np.copy(c.detach().numpy())
        c_min = np.min(self.c_matrix)
        c_max = np.max(self.c_matrix)
        self.log("train_loss", loss, on_epoch=True) 
        self.log('diag', diag, on_epoch=True)
        self.log('off_diag', off_diag, on_epoch=True)
        self.log('c_max', c_max, on_epoch=True)
        self.log('c_min', c_min, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, 
                                      threshold=0.0001, threshold_mode='rel', cooldown=1, 
                                      min_lr=1.0e-7, eps=1e-08, verbose=True)        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}        
    
    # use pin_memory=True, ?
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, num_workers=6)
    
    def on_train_epoch_end(self):
        # make the C matrix image
        img = np.zeros((self.c_matrix.shape[0], self.c_matrix.shape[1], 3))
        self.c_matrix = np.abs(self.c_matrix)
        img = np.abs(self.c_matrix) * 255
        img = img.astype('uint8')
        fname = './lightning_logs/barlow/images/epoch_' + str(self.current_epoch) + '.jpg'
        write_image(fname, img)
        return
    
