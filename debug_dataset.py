import numpy as np
import os

base_path = "./data/test"

jukebox = np.load(os.path.join(base_path, "jukebox_feats", "gBR_sBM_cAll_d04_mBR0_ch02_slice0.npy"))
beat     = np.load(os.path.join(base_path, "beat_feats/npy", "gBR_sBM_cAll_d04_mBR0_ch02_slice0_beat_encoding.npy"))
text     = np.load(os.path.join(base_path, "text_encodings/npy", "gBR_sBM_cAll_d04_mBR0_ch02_slice0_response_text_encoding.npy"))

print("jukebox shape:", jukebox.shape)
print("beat shape   :", beat.shape)
print("text shape   :", text.shape)