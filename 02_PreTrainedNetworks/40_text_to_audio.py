#%% packages
from transformers import pipeline
import scipy

#%% model selection
task = "text-to-audio"
model = "facebook/musicgen-small"

# %%
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True,"max_new_tokens": 50})

scipy.io.wavfile.write("musicgen_out1.wav", rate=music["sampling_rate"], data=music["audio"])
# %%
