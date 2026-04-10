"""
Face recognition inference script.

Given a trained embedding model and a gallery of known identities, this
script encodes a query image and finds the closest match using cosine
similarity.

Usage
-----
    # Build a gallery from a directory of identities then recognise a face:
    python -m src.recognize \
        --model_path models/face_embedding.pth \
        --gallery_dir data/gallery \
        --query image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import get_eval_transforms
from src.model import FaceEmbeddingNet

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognise a face from a gallery")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved .pth weights")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Gallery directory (one sub-folder per identity)")
    parser.add_argument("--query", type=str, required=True, help="Path to the query image")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum cosine similarity to accept a match")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to display")
    return parser.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_image(model: FaceEmbeddingNet, img_path: Path, device: torch.device) -> np.ndarray:
    transform = get_eval_transforms()
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    embedding = model(tensor).cpu().numpy().flatten()
    return embedding


def build_gallery(
    model: FaceEmbeddingNet,
    gallery_dir: Path,
    device: torch.device,
) -> list[tuple[str, Path, np.ndarray]]:
    """Return a list of (identity, image_path, embedding) for every gallery image."""
    gallery: list[tuple[str, Path, np.ndarray]] = []

    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        identity = person_dir.name
        for img_file in sorted(person_dir.iterdir()):
            if img_file.suffix.lower() in SUPPORTED_EXT:
                emb = embed_image(model, img_file, device)
                gallery.append((identity, img_file, emb))

    if not gallery:
        raise ValueError(f"No gallery images found in {gallery_dir}")

    print(f"Gallery built: {len(gallery)} images across "
          f"{len({g[0] for g in gallery})} identities")
    return gallery


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def recognize(
    query_embedding: np.ndarray,
    gallery: list[tuple[str, Path, np.ndarray]],
    top_k: int = 3,
) -> list[tuple[str, Path, float]]:
    """Return top-k matches sorted by descending cosine similarity."""
    scores = [
        (identity, path, cosine_similarity(query_embedding, emb))
        for identity, path, emb in gallery
    ]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    model = FaceEmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    gallery = build_gallery(model, Path(args.gallery_dir), device)
    query_emb = embed_image(model, Path(args.query), device)

    matches = recognize(query_emb, gallery, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print("-" * 50)
    for rank, (identity, path, score) in enumerate(matches, 1):
        status = "MATCH" if score >= args.threshold else "below threshold"
        print(f"  #{rank}  {identity:<20s}  similarity={score:.4f}  ({status})")

    best_identity, best_path, best_score = matches[0]
    if best_score >= args.threshold:
        print(f"\nRecognised as: {best_identity} (score={best_score:.4f})")
    else:
        print(f"\nNo confident match found (best score {best_score:.4f} "
              f"< threshold {args.threshold})")


if __name__ == "__main__":
    main()
