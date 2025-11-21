import torch
import math

class TiledImageSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
                "feather_ratio": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TILE_INFO")
    RETURN_NAMES = ("tiles", "masks", "tile_info")
    FUNCTION = "split"
    CATEGORY = "TiledNodes"

    def split(self, image, tile_width, tile_height, overlap, feather_ratio):
        # image shape: [Batch, Height, Width, Channels]
        # We assume Batch=1 for simplicity or process all batch items similarly
        # For now, let's handle single image input or treat batch as individual tasks to tile
        
        results_tiles = []
        results_masks = []
        
        batch_size, img_h, img_w, channels = image.shape
        
        # Ensure tile size is not larger than image
        tile_width = min(tile_width, img_w)
        tile_height = min(tile_height, img_h)
        
        # Calculate number of tiles
        stride_w = tile_width - overlap
        stride_h = tile_height - overlap
        
        cols = math.ceil((img_w - overlap) / stride_w)
        rows = math.ceil((img_h - overlap) / stride_h)
        
        tile_info = {
            "original_height": img_h,
            "original_width": img_w,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "overlap": overlap,
            "feather_ratio": feather_ratio,
            "batch_size": batch_size,
            "positions": [] 
        }
        
        for b in range(batch_size):
            for r in range(rows):
                for c in range(cols):
                    # Calculate coordinates
                    y = r * stride_h
                    x = c * stride_w
                    
                    # Adjust last row/col to fit exactly at the end if needed
                    # Strategy: Shift start point back so tile ends at image edge
                    if x + tile_width > img_w:
                        x = img_w - tile_width
                    if y + tile_height > img_h:
                        y = img_h - tile_height
                    
                    # Crop image
                    # image is [B, H, W, C]
                    crop = image[b:b+1, y:y+tile_height, x:x+tile_width, :]
                    results_tiles.append(crop)
                    
                    # Generate Mask
                    # Mask shape: [H, W]
                    mask = torch.ones((tile_height, tile_width), dtype=torch.float32)
                    
                    # Apply feathering to mask based on adjacencies
                    # Feathering creates a gradient from 0 (edge) to 1 (inner)
                    # feather_pixel = int(min(tile_width, tile_height) * feather_ratio)
                    
                    feather_w = int(tile_width * feather_ratio)
                    feather_h = int(tile_height * feather_ratio)
                    
                    # Create gradient tensors
                    # Horizontal gradient (0 to 1)
                    grad_x = torch.linspace(0, 1, feather_w)
                    # Vertical gradient (0 to 1)
                    grad_y = torch.linspace(0, 1, feather_h)
                    
                    # Mask Left (if not first column)
                    if x > 0: 
                        mask[:, :feather_w] *= grad_x
                        
                    # Mask Right (if not last column)
                    if x + tile_width < img_w:
                        mask[:, -feather_w:] *= grad_x.flip(0)
                        
                    # Mask Top (if not first row)
                    if y > 0:
                        mask[:feather_h, :] *= grad_y.unsqueeze(1)
                        
                    # Mask Bottom (if not last row)
                    if y + tile_height < img_h:
                        mask[-feather_h:, :] *= grad_y.flip(0).unsqueeze(1)
                    
                    results_masks.append(mask)
                    
                    tile_info["positions"].append({
                        "batch_index": b,
                        "x": x,
                        "y": y,
                        "row": r,
                        "col": c
                    })

        # Stack results
        # tiles: [N, H, W, C]
        # masks: [N, H, W] -> ComfyUI expects [N, H, W] for masks usually
        
        if not results_tiles:
            return (image, torch.zeros((img_h, img_w)), tile_info)
            
        final_tiles = torch.cat(results_tiles, dim=0)
        final_masks = torch.stack(results_masks, dim=0)
        
        return (final_tiles, final_masks, tile_info)

class TiledImageMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_info": ("TILE_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "TiledNodes"

    def merge(self, images, tile_info):
        # images: [N, tile_h, tile_w, C]
        # tile_info: dict with original size and positions
        
        orig_h = tile_info["original_height"]
        orig_w = tile_info["original_width"]
        batch_size = tile_info.get("batch_size", 1)
        
        # Create output tensor [B, H, W, C]
        # We need an accumulator for color and an accumulator for weights (alpha)
        output = torch.zeros((batch_size, orig_h, orig_w, 3), dtype=torch.float32)
        weights = torch.zeros((batch_size, orig_h, orig_w, 1), dtype=torch.float32)
        
        tile_idx = 0
        
        # If the input images count doesn't match expected tiles, we might have an issue, 
        # but we proceed as far as we can.
        
        for pos in tile_info["positions"]:
            if tile_idx >= len(images):
                break
                
            tile = images[tile_idx] # [H, W, C]
            b_idx = pos["batch_index"]
            x = pos["x"]
            y = pos["y"]
            h, w, c = tile.shape
            
            # Re-create the feather mask for weighting
            # Ideally we should pass the mask through, but re-generating is cheap
            # and saves memory bandwidth if unchanged.
            # However, if the processing CHANGED the mask, we assume standard feathering for blending.
            
            feather_ratio = tile_info["feather_ratio"]
            feather_w = int(w * feather_ratio)
            feather_h = int(h * feather_ratio)
            
            weight_mask = torch.ones((h, w, 1), dtype=torch.float32)
            
            grad_x = torch.linspace(0, 1, feather_w)
            grad_y = torch.linspace(0, 1, feather_h)
            
            # Apply same logic as splitter to create weight mask
            if x > 0: 
                weight_mask[:, :feather_w, 0] *= grad_x
            if x + w < orig_w:
                weight_mask[:, -feather_w:, 0] *= grad_x.flip(0)
            if y > 0:
                weight_mask[:feather_h, :, 0] *= grad_y.unsqueeze(1)
            if y + h < orig_h:
                weight_mask[-feather_h:, :, 0] *= grad_y.flip(0).unsqueeze(1)
                
            # Add to accumulator
            output[b_idx, y:y+h, x:x+w, :] += tile * weight_mask
            weights[b_idx, y:y+h, x:x+w, :] += weight_mask
            
            tile_idx += 1
            
        # Normalize
        # Avoid division by zero
        weights[weights == 0] = 1.0
        output /= weights
        
        return (output,)

NODE_CLASS_MAPPINGS = {
    "TiledImageSplitter": TiledImageSplitter,
    "TiledImageMerger": TiledImageMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageSplitter": "Tiled Image Splitter",
    "TiledImageMerger": "Tiled Image Merger"
}

