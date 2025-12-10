🌟 README — End-to-End CLIP + Stable Diffusion Pipeline with Fine-Tuning & Evaluation

Welcome to the Text-to-Image Generation & Fine-Tuning Project! 🚀
This repository implements a complete, production-grade workflow that combines COCO Dataset preprocessing, CLIP text embeddings, Stable Diffusion generation, domain-specific fine-tuning, and quantitative evaluation using FID & Inception Score.
It includes dataset extraction, training, inference, optimization, and deep integration between CLIP and the Stable Diffusion UNet.

🧩 Project Overview

This project walks through every major stage of building a high-performance text-to-image generator:

🔽 Download & explore COCO Image Captioning dataset

🖼️ Build image–caption pairs (400k+)

✂️ Extract domain-specific subsets (Animals / Vehicles / Food / Sports / Indoor)

🧹 Resize, clean, and prepare images for model input

🧠 Generate CLIP embeddings for all captions

🎨 Load Stable Diffusion v1-4 for text-to-image

🔗 Verify CLIP → UNet cross-attention integration

🎯 Fine-tune Stable Diffusion on animal domain

📊 Evaluate model quality with proper FID and IS metrics

⚙️ Compare schedulers & CFG scales to pick the best combination

🦁 Generate beautiful images using your fine-tuned model

Every stage is automated, logged, visualized, and stored in structured folders for reproducibility. 💾

📦 1. Dataset Acquisition & Preprocessing

The project begins by pulling the COCO Dataset via KaggleHub:

Over 400,000 caption–image pairs

Multiple annotations directories

Images processed into 256×256 RGB

All invalid/missing images removed

Captions stored in captions_subset.json

A domain-filtering system lets you choose categories like:

🐶 Animals
🚗 Vehicles
🍕 Food
🏀 Sports
🛋️ Indoor

For this project, the Animals domain was selected.
More than 69,000 captioned animal images were detected, and 4,000 samples were used for training-ready preprocessing.

🧠 2. CLIP Text Embedding Generation

The project loads OpenAI’s CLIP ViT-B/32:

Tokenizes captions

Implements mean pooling

Generates 512-dim embeddings per caption

Saves embeddings to:

data/processed/embeddings/text_embeddings.npy

data/processed/embeddings/text_index.json

We verify embedding meaningfulness via cosine similarity (e.g., closely related captions yield high similarity). 🔍

This embedding matrix becomes the conditioning signal for Stable Diffusion.

🎨 3. Stable Diffusion Integration

Stable Diffusion v1-4 is loaded with:

⚡ FP16 precision

🧩 Attention slicing

🛑 Safety checker disabled for faster inference

Then we run Milestone-1: generate sample images for prompts like:

“A golden retriever playing fetch on a sunny beach”

“A futuristic neon-lit city skyline at night”

“Fresh sushi on a wooden plate with chopsticks”

Generation time is around 8–9 seconds per image on Tesla P100 GPU. ⚡

🔗 4. Deep CLIP ↔ Diffusion Verification

We confirm:

CLIP generates 77 × 768 embeddings

These embeddings flow into 16 UNet cross-attention layers

Latents & embeddings match dtype and shape

UNet uses CLIP conditioning during denoising

This verification ensures the foundation for fine-tuning is correct, stable, and fully integrated. ✔️

🐾 5. Domain-Specific Fine-Tuning (Animals)

We fine-tune only the text-conditioning attention layers (attn2.to_v), keeping 850M+ parameters frozen.

Advantages:

🔥 Low GPU memory

🚀 Fast training

🛡️ Stable convergence

🐘 Strong domain specialization

Breakdown:

2,000 animal images

2 epochs (~4,000 steps)

SGD optimizer

FP16 + gradient checkpointing

Auto-casting for speed

Timesteps sampled from 0–999

Training logs include:

Loss curves

Memory usage

Speed (1.6–1.7 images/sec)

Checkpoints every 100 steps

Final avg loss ~0.23 📉

Everything is saved in:

models/fine_tuned_animals/


Including:

fine-tuned UNet

full pipeline

training metadata

🦊 6. Generating Images with the Fine-Tuned Model

After training, prompts like:

“A majestic lion in the savanna”

“A colorful parrot on a branch”

“A playful puppy in a garden”

produce much sharper, more domain-aware, and more consistent images than the base model. 🎉

A comparison plot includes:

📉 Training loss curve

🦁 Three sample generated images

📊 7. FID & IS Evaluation (Proper Implementation)

We implement correct versions of:

🔢 Fréchet Inception Distance (FID)

Using:

Mean of features

Covariance matrices

√ product covariance

Trace operations

🧪 Inception Score (IS)

Using KL divergence across splits.

🧬 8. Scheduler + CFG Experiment Grid

We test:

Schedulers

DDIM

PNDM

Euler

CFG scales

5.0

7.5

10.0

Settings

Steps: 40

Images per config: 40

Prompt: “A photorealistic orange tabby cat…”

A reference set (40 images) is generated using DPMSolver.
Then each scheduler–CFG combo is evaluated using:

FID score 🧮

Inception Score 🔥

Results help determine:

Best realism

Best diversity

Best prompt alignment

All metrics logged, saved, and reusable.

💡 9. Final Inference Mode

Users can type:

Enter your prompt:


and instantly generate images using the domain-specialized model.
On GPU, generation takes ~8 seconds per 512×512 image.

🏁 Project Highlights

✨ Complete pipeline from dataset → embeddings → diffusion → fine-tuning
🧠 Intelligent domain filtering
📦 Efficient data preprocessing pipeline
🧬 CLIP + Diffusion integration verified
🔥 Fine-tuning with only 16 parameters
📊 Full evaluation using proper FID & IS
🦁 Domain-optimized final model
🛠️ Best-in-class scheduler comparison
🚀 Fast inference & clean model deployment structure

🎉 Conclusion

This project delivers a fully functional, fine-tuned, and evaluated text-to-image system—capable of generating high-quality animal images conditioned on natural-language prompts.
It demonstrates mastery across:

Large-scale data processing

Embedding extraction

Latent diffusion modeling

GPU-optimized training

Advanced evaluation metrics

Domain specialization

You now have a pipeline that can be adapted to ANY domain and ANY dataset, including LoRA, DreamBooth, or multi-domain fine-tuning.
