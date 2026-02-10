import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import cv2
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SAMSegmentationMasksGenerator:
    def __init__(
        self,
        swpc_drawing_dir: Path,
        save_dir: Path,
        model_type="vit_h",
        sam_checkpoint_filepath: Path = None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.sam_checkpoint_filepath = sam_checkpoint_filepath
        self.swpc_drawing_dir = swpc_drawing_dir
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.device = device
        logger.info(f"Using device: {self.device}")

        if device.type == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def initialize_sam_model(
        self,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    ):
        logger.info("Initializing SAM model...")
        sam = sam_model_registry[self.model_type](
            checkpoint=self.sam_checkpoint_filepath
        )
        sam.to(device=self.device)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        )

        return mask_generator

    def generate_masks_for_image(
        self, sam: SamAutomaticMaskGenerator, image_filepath: Path
    ) -> list:
        image = cv2.imread(str(image_filepath))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            masks = sam.generate(image_rgb)

        return masks

    def save_masks(self, masks: list, save_filepath: Path):
        np.savez_compressed(save_filepath, *masks)

    def process_all_images(self):
        logger.info("Starting SAM segmentation mask generation...")
        mask_generator = self.initialize_sam_model()

        for image_path in tqdm(list(self.swpc_drawing_dir.iterdir())):
            if not image_path.is_file() or image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:
                continue

            print("Processing: ", image_path.name)
            drawing_name = (
                image_path.stem
            )  # Assuming filename without extension is the date string
            save_path = self.save_dir / f"{drawing_name}-SAM_masks.npz"

            if save_path.exists():
                print(f"Mask already exists for {image_path.name}, skipping")
                continue

            try:
                masks = self.generate_masks_for_image(mask_generator, image_path)

                self.save_masks(masks, save_path)
                del masks
                print(f"Mask generated and saved for {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")

        del mask_generator
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
