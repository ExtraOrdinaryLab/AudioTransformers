# Audio Transformers

Inspired by [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), which revolutionise text-based embeddings for tasks like semantic similarity and clustering, Audio Transformers extends this concept to the audio domain. By leveraging pre-trained transformer models and advanced pooling techniques, this framework transforms audio data into meaningful, high-dimensional embeddings. These embeddings can be used for tasks such as audio classification, speaker verification, and content-based audio retrieval. Whether you're working with raw audio files or preprocessed waveforms, Audio Transformers provides an efficient and flexible pipeline for embedding generation, powered by the latest advancements in deep learning for audio.

# Getting Started

First download a pretrained model.

```python
from audio_transformers import AudioTransformer

audio_transformer = AudioTransformer(
    model_name_or_path='facebook/wav2vec2-base', 
    max_length_seconds=10, 
    return_attention_mask=True, 
    pooling_mode='mean', 
)
```

Then provide some audio files to the model.

```python
audio_filepaths = [
    '/mnt/confit/gtzan/genres/country/country.00092.au',
    '/mnt/confit/gtzan/genres/country/country.00029.au',
    '/mnt/confit/gtzan/genres/country/country.00026.au',
    '/mnt/confit/gtzan/genres/country/country.00072.au',
    '/mnt/confit/gtzan/genres/country/country.00090.au',
    '/mnt/confit/gtzan/genres/country/country.00094.au',
    '/mnt/confit/gtzan/genres/country/country.00088.au',
    '/mnt/confit/gtzan/genres/country/country.00073.au',
    '/mnt/confit/gtzan/genres/country/country.00019.au',
    '/mnt/confit/gtzan/genres/country/country.00069.au', 
]
embeddings = model.encode(audio_filepaths)
print(embeddings.shape)
# => (10, 768)
```