# Minimal demo: image -> VAE encoder -> latent z -> (noise/quant) -> decoder -> reconstruction

import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image

# ---- Image variables ----
INPUT_PATH  = "dogTest.png"
OUTPUT_PATH = "reconstruction.png"
SIZE        = 512       # SD VAE expects 512x512 typically
SIGMA       = 0.10      # AWGN std in latent space
N_BITS      = 6         # uniform quantization bits (e.g., 3..8)
DROPOUT_P   = 0.00      # element-wise dropout prob on z
# -----------------------------

# Shouldn't take too long on CPU ******** check later for lab
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained SD VAE (encoder/decoder only)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

# Preprocess: resize and normalize to [-1,1]
pre = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load image
img = Image.open(INPUT_PATH).convert("RGB")
x = pre(img).unsqueeze(0).to(device)  # [1,3,H,W]

with torch.no_grad():
    # Encode: image -> latent z (respect scaling_factor used by SD VAE training)
    posterior = vae.encode(x)
    z = posterior.latent_dist.mode()
    sf = getattr(vae.config, "scaling_factor", 1.0)
    z = z * sf

    # ---- Channel: AWGN + uniform quant + optional dropout ----
    if SIGMA > 0:
        z = z + SIGMA * torch.randn_like(z)

    if N_BITS is not None:
        L = (2 ** int(N_BITS)) - 1
        zc = torch.clamp(z, -1, 1)
        z = torch.round((zc + 1) * 0.5 * L) / L
        z = z * 2 - 1

    if DROPOUT_P > 0:
        mask = (torch.rand_like(z) > DROPOUT_P).float()
        z = z * mask
    # ----------------------------------------------------------

    # Decode: latent -> image (invert scaling_factor)
    x_hat = vae.decode(z / sf).sample  # in [-1,1]

# Postprocess to uint8 and save
x_hat = (x_hat.clamp(-1,1) + 1) * 0.5
x_hat = (x_hat * 255).byte().cpu().squeeze(0).permute(1,2,0).numpy()
Image.fromarray(x_hat).save(OUTPUT_PATH)
print(f"Done -> {OUTPUT_PATH}")
