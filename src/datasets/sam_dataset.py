import os
import cv2
from pathlib import Path
from datasets.folder_utils import extract_date
import matplotlib.pyplot as plt
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from tqdm import tqdm
from datasets.drawings_dataset import DrawingsDataset

# TODO check if has any usage and clean up as needed in that case
class SAMDataset():

    def __init__(self, root):
        self.root = root

    # TODO make these all just build
        
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            print(m.shape)
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        
        ax.imshow(img)

    def filenames(self) -> list[str]:
        return sorted(
            (str(p) for p in Path(self.root).rglob("*") if p.is_file())
        )
        
    def build(self,
              checkpoint, # TODO make this argument automated somehow?
              drawings: DrawingsDataset,
              model_type = "vit_h",
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        if os.path.exists(self.root):
            print("Cannot build to preexisting path. Exiting.")
            exit(1)
        
        os.makedirs(self.root)

        # if base_dir!=None:
        #     drawing_dir = os.path.join(base_dir, "drawings")
        #     save_dir = os.path.join(base_dir, "sam_masks")
        
        if device.type == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Initialize SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
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
        for image_path in tqdm(drawings.filenames()):

            # save_path = os.path.join(self.root, image_path[:-4] + "-SAM_masks.npz")
            # if os.path.exists(save_path):  # Skip already processed images if present (shouldn't happen)
            #     continue

            print("Processing: ", image_path)
            try:
                # Load and preprocess image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Generate masks
                masks = mask_generator.generate(image)

                # Save masks
                date_str = extract_date(os.path.basename(image_path))
                save_path = os.path.join(self.root, f"{date_str}-SAM_masks.npz")

                np.savez_compressed(save_path, *masks)
                print(f"Mask generated and saved for {image_path}")

                # Clear memory
                del masks, image
                if device.type == "cuda":
                   torch.cuda.empty_cache()
                   torch.cuda.ipc_collect()

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                break

        # Free up model memory
        del sam
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()