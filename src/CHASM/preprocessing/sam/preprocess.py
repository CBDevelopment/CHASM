import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import cv2
import os
from pathlib import Path


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        print(m.shape)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    ax.imshow(img)


# TODO make this work with command args


def preprocess_sam(
    base_dir,
    checkpoint,  # TODO make this argument automated somehow?
    drawing_dir=None,
    save_dir=None,
    model_type="vit_h",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if base_dir is not None:
        base_path = Path(base_dir)
        drawing_dir = base_path / "drawings"
        save_dir = base_path / "sam_masks"

    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Initialize mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    # Process images
    drawing_dir_path = Path(drawing_dir)
    for image_file in tqdm(list(drawing_dir_path.iterdir())):
        if not image_file.is_file() or image_file.name == "README.md":
            continue

        save_path = Path(save_dir) / f"{image_file.stem}-masks.npz"
        if save_path.exists():  # Skip already processed images
            continue

        print("Processing: ", image_file.name)
        try:
            # Load and preprocess image
            image = cv2.imread(str(Path(DIR) / image_file.name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate masks
            masks = mask_generator.generate(image)

            # Save masks
            np.savez_compressed(save_path, *masks)
            print(f"Mask generated and saved for {image_path}")

            # Clear memory
            del masks, image
            ##if DEVICE.type == "cuda":
            #    torch.cuda.empty_cache()
            #    torch.cuda.ipc_collect()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            break

    # Free up model memory
    del sam
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    DIR = r"data_test_set\drawings\2017"
    SAVE_DIR = r"data_test_set\masks\2017"

    sam_checkpoint = r"src\pipeline\checkpoints\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)

    if DEVICE.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)

    # Initialize mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        # min_mask_region_area=100,  # Uncomment if OpenCV is available
    )

    # Process images
    dir_path = Path(DIR)
    for image_file in tqdm(list(dir_path.iterdir())):
        if not image_file.is_file() or image_file.name == "README.md":
            continue

        save_path = Path(SAVE_DIR) / f"{image_file.stem}-masks.npz"
        if save_path.exists():  # Skip already processed images
            continue

        print("Processing: ", image_file.name)
        try:
            # Load and preprocess image
            image = cv2.imread(str(dir_path / image_file.name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate masks
            masks = mask_generator.generate(image)

            # Save masks
            np.savez_compressed(save_path, *masks)
            print(f"Mask generated and saved for {image_path}")

            # Clear memory
            del masks, image
            ##if DEVICE.type == "cuda":
            #    torch.cuda.empty_cache()
            #    torch.cuda.ipc_collect()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            break

    # Free up model memory
    del sam
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
