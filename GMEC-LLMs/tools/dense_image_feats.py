import numpy as np
from pathlib import Path
from pdb import set_trace as stop
import os
class dense_feats:
    def __init__(self) -> None:
        self.image_feats_path = '../datasets/MELD-ECPE/Images_CLIPFeats'
        self.dense_feats_path = '../datasets/MELD-ECPE/MELDECPE/video_embedding_CLIPmean_4096.npy'
        self.video_id_mapping='../datasets/MELD-ECPE/MELDECPE/video_id_mapping.npy'
        self.img_cnnfeat_path='../datasets/MELD-ECPE/MELDECPE/video_embedding_4096.npy'


    def read_npy_data(self,data_path):
        file = np.load(data_path)
        return file
    def save_npy_data(self,data_path,clip_feats):
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_path,
            x=clip_feats,
        )
    def process(self,):
        vid2Id = eval(str(np.load(self.video_id_mapping, allow_pickle=True)))
        v_emb = np.load(self.img_cnnfeat_path, allow_pickle=True)
        new_emb = np.zeros_like(v_emb)
        for vid,Id in vid2Id.items():
            diaId, uttID = vid.replace('dia','').split('utt')
            img_feat_name = f'{vid}_m.npz'
            img_path = os.path.join(self.image_feats_path,img_feat_name)
            data = np.load(img_path)
           
            new_data= data['x'].reshape(16*16,-1).mean(0)
            new_emb[Id]=new_data

        np.save(self.dense_feats_path, new_emb, allow_pickle=True)





    def run(self,):
        self.process()




if __name__ == "__main__":
    dense_feats().run()