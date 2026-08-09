import os
from pathlib import Path
from PIL import Image, ImageOps


DATASET_ROOT = Path(".")                 
OUTPUT_ROOT = Path("cropped_dataset")     
OUTPUT_SIZE = 224                        
PADDING_RATIO = 0.08                      
MIN_CROP_SIZE = 20                       
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = [
    "demoman", "engineer", "heavy", "medic", "pyro",
    "scout", "sniper", "soldier", "spy",
]
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]



def letterbox_resize(img: Image.Image, size: int) -> Image.Image:
    """Aspect ratio'yu koruyarak resize edip kare padding ekler."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))  # gri padding (YOLO standardı)
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas


def find_image_for_label(images_dir: Path, label_stem: str) -> Path | None:
    for ext in IMG_EXTENSIONS:
        candidate = images_dir / f"{label_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_split(split: str):
    labels_dir = DATASET_ROOT / split / "labels"
    images_dir = DATASET_ROOT / split / "images"

    if not labels_dir.exists():
        print(f"[!] {labels_dir} bulunamadı, atlanıyor.")
        return

    total = 0
    skipped_small = 0
    skipped_missing = 0

    label_files = list(labels_dir.glob("*.txt"))
    for label_path in label_files:
        img_path = find_image_for_label(images_dir, label_path.stem)
        if img_path is None:
            skipped_missing += 1
            continue

        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)  # bazı jpg'lerde rotasyon meta datası olur
            img_w, img_h = im.size

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                # normalize -> pixel
                box_w = w * img_w
                box_h = h * img_h
                cx = xc * img_w
                cy = yc * img_h

                # padding ekle
                box_w *= (1 + PADDING_RATIO)
                box_h *= (1 + PADDING_RATIO)

                x1 = max(0, int(cx - box_w / 2))
                y1 = max(0, int(cy - box_h / 2))
                x2 = min(img_w, int(cx + box_w / 2))
                y2 = min(img_h, int(cy + box_h / 2))

                if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                    skipped_small += 1
                    continue

                crop = im.crop((x1, y1, x2, y2))
                crop = letterbox_resize(crop, OUTPUT_SIZE)

                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    continue
                class_name = CLASS_NAMES[class_id]

                out_dir = OUTPUT_ROOT / split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{label_path.stem}_{idx}.jpg"
                crop.save(out_path, quality=95)
                total += 1

    print(f"[{split}] {total} crop kaydedildi. "
          f"({skipped_small} küçük diye atlandı, {skipped_missing} resim bulunamadı)")


def main():
    for split in SPLITS:
        process_split(split)
    print("\nBitti. Çıktı klasörü:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
