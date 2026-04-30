class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512, f_min=20, f_max=16000, channels=1, norm="slaney", mel_scale="htk", target_size=(256, 256), top_db=80.0, delta_win=5,):
        super().__init__()
        self.channels = channels
        self.top_db = top_db

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            mel_scale=mel_scale,
            pad_mode="reflect",
            power=2.0,
            norm=norm,
            center=True,
        )

        print('channels', self.channels)
        self.resize = torchvision.transforms.Resize(size=target_size)

    def power_to_db(self, S):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(S))  # fixed ref = 1.0 effectively
        if self.top_db is not None:
            # per-sample top_db clipping: keep dims for broadcasting
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x, resize=True):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        mel_spec = self.mel_transform(x)          # (B, n_mels, time)
        mel_spec = self.power_to_db(mel_spec)

        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)

        if resize: mel_spec = self.resize(mel_spec)           # (B, C, H, W)

        
        B, C = mel_spec.shape[:2]
        flat = mel_spec.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)

        if squeeze:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec


import timm

class BirdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'scale':1,
            'backbone_pooling':'avg',
            'backbone':'tf_efficientnetv2_b0',
            'dropout':0.1,
            'pretrained':True,
            'channels':1,
            'num_labels':234,
        }
        if config: self.config.update(config)

        self.training = True

        self.backbone = timm.create_model(
            self.config['backbone'], 
            pretrained=self.config['pretrained'],  
            num_classes=self.config['num_labels'],  
            global_pool=self.config['backbone_pooling'],
            in_chans=1,
            drop_rate=self.config['dropout'],
        )
        feature_dim = self.backbone.num_features
        
    def forward(self, x):
        labels = self.backbone(x)
        return labels
