import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoModel
from transformers.image_utils import load_image
# Load Model

model = AutoModel.from_pretrained(
    '/media/data4/leizh/models/nemotron-colembed-vl-4b-v2',
    device_map='cuda',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).eval()

# Queries
queries = [
    'How is AI improving the intelligence and capabilities of robots?',
    'Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization.',
    'Generative AI can generate DNA sequences that can be translated into proteins for bioengineering.'
]

image_urls = [
    "https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg",
    "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/asr-nemo-canary-featured.jpg",
    "https://blogs.nvidia.com/wp-content/uploads/2023/02/genome-sequencing-helix.jpg"
]

# Load all images (load_image handles both local paths and URLs)
images = [load_image(img_path) for img_path in image_urls]

# Encoding
query_embeddings = model.forward_queries(queries, batch_size=8)
image_embeddings = model.forward_images(images, batch_size=8)

scores = model.get_scores(
    query_embeddings,
    image_embeddings
)
# Diagonal should have higher scores
print(scores)

# tensor([[21.5332, 21.1848, 20.9185],
#         [32.4948, 33.2485, 32.5982],
#         [26.0623, 26.1014, 26.5692]], device='cuda:0')
