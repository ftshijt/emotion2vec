import fairseq
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import os


# Get the full path of the current script
script_path = os.path.abspath(__file__)

# Get the directory of the current script
script_dir = os.path.dirname(script_path)


@dataclass
class UserDirModule:
    user_dir: str


class EMO2VEC:
    def __init__(self, checkpoint_dir, use_gpu=True, label_predict=False, label_model=None):
        self.checkpoint_dir = checkpoint_dir
        model_path = UserDirModule(os.path.join(script_dir, "upstream"))
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
        self.model = model[0]
        self.model.eval()
        self.task = task
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.cuda()
        
        if label_predict:
            from emo2vec_versa.iemocap_downstream.model import BaseModel
            label_dict={'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
            self.idx2label = {v: k for k, v in label_dict.items()}
            self.label_model = BaseModel(input_dim=768, output_dim=len(label_dict))
            label_model_ckpt = torch.load(label_model)
            self.label_model.load_state_dict(label_model_ckpt)
            self.label_model.eval()
            self.label_model.cuda()
    
    def extract_feature(self, input_wav, fs, granularity='utterance'):
        input_wav= np.array(input_wav)
        if fs != 16000:
            raise ValueError("Sample rate should be 16kHz, but got {}".format(fs))
        if input_wav.ndim > 1:
            raise ValueError("Channel should be 1, but got {}".format(input_wav.shape[1]))
        
        with torch.no_grad():
            source = torch.from_numpy(input_wav).float()
            if self.use_gpu:
                source = source.cuda()
            if self.task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)
            try:
                feats = self.model.extract_features(source, padding_mask=None)
                feats = feats['x'].squeeze(0).cpu().numpy()
                if granularity == 'frame':
                    feats = feats
                elif granularity == 'utterance':
                    feats = np.mean(feats, axis=0)
                return feats
            except Exception as e:
                print("Error extracting features: ", e)
                return None
    
    # NOTE(jiatong): not verified
    def predict_label(self, input_wav, fs):
        input_wav= np.array(input_wav)
        if fs != 16000:
            raise ValueError("Sample rate should be 16kHz, but got {}".format(fs))
        if input_wav.ndim > 1:
            raise ValueError("Channel should be 1, but got {}".format(input_wav.shape[1]))
        
        with torch.no_grad():
            source = torch.from_numpy(input_wav).float()
            if self.use_gpu:
                source = source.cuda()
            if self.task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)
            try:
                feats = self.model.extract_features(source, padding_mask=None)
                padding_mask = torch.zeros(1, feats.size(1)).bool()
                outputs = self.label_model(feats, padding_mask)
                _, predict = torch.max(outputs.data, dim=1)
                return self.idx2label[predict.item()]
            except Exception as e:
                print("Error extracting features: ", e)
                return None


if __name__ == "__main__":
    ckpt = "/home/jiatong/projects/espnet/tools/versa/tools/emotion2vec/emotion2vec_base.pt"
    emo2vec = EMO2VEC(ckpt)
    a = np.random.randn(16000)
    print(emo2vec.extract_feature(a, 16000).shape)