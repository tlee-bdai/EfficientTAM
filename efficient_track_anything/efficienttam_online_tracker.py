# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy as dc
from efficient_track_anything.modeling.efficienttam_base import (
    EfficientTAMBase,
    NO_OBJ_SCORE,
)


class EfficientTAMOnlineBatchTracker(EfficientTAMBase):
    """
    Online multi-object tracker:
      - Initializes from a SINGLE frame (no full video preloading).
      - Tracks multiple objects in BATCH (no per-object loops).
      - Requires equal number of clicks per object (enforced via assert).
      - Inputs are numpy/list; this class converts to tensors internally.

    IMPORTANT parallels to EfficientTAMVideoPredictor:
      * Images are resized to model.image_size (no extra normalization).
      * Point coords are optionally normalized by (W,H) and then scaled by image_size.
      * Outputs are mask LOGITS (not probabilities). We do NOT call sigmoid here.
    """

    def __init__(self, fill_hole_area: int = 0, non_overlap_masks: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks

        # Online state
        self._initialized = False
        self._t = 0  # current frame index in the online stream
        self._video_H = None
        self._video_W = None
        self._B = 0  # number of objects
        self._storage_device = None

        # Batched storage (shared across all objects via batch dimension)
        # Exactly the same structure that EfficientTAMVideoPredictor expects per object,
        # but here we store batched tensors for ALL objects at once.
        self._output_dict = {
            "cond_frame_outputs": {},     # {frame_idx: out_batched}
            "non_cond_frame_outputs": {}, # {frame_idx: out_batched}
        }

        # Cache constants that are invariant across frames (maskmem_pos_enc slices)
        self._constants = {}

    # ------------- helpers copied-in-spirit from the video predictor -------------

    @property
    def _device(self):
        return next(self.parameters()).device

    def _resize_image_to_model(self, img_t: torch.Tensor) -> torch.Tensor:
        """
        img_t: [1, 3, H, W] float tensor (0..255 or 0..1; we do not normalize here)
        Resize to [1, 3, image_size, image_size] using bilinear+antialias.
        """
        H, W = img_t.shape[-2:]
        if (H, W) == (self.image_size, self.image_size):
            return img_t
        return F.interpolate(
            img_t, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False, antialias=True
        )
    
    def _normalize_image(self, img_t: torch.Tensor) -> torch.Tensor:
        """
        img_t: [1, 3, H, W] float tensor (0..255 or 0..1)
        Normalize to ImageNet stats.
        """
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=img_t.device).view(1, 3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=img_t.device).view(1, 3, 1, 1)
        return (img_t - mean) / std

    def _expand_backbone_to_batch(self, backbone_out, B):
        """
        Expand a single-image backbone feature dict to batch size B (like _get_image_feature).
        """
        expanded = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded["backbone_fpn"]):
            expanded["backbone_fpn"][i] = feat.expand(B, -1, -1, -1)
        for i, pos in enumerate(expanded["vision_pos_enc"]):
            expanded["vision_pos_enc"][i] = pos.expand(B, -1, -1, -1)
        return expanded

    def _to_video_res_and_constrain(self, any_res_masks: torch.Tensor) -> torch.Tensor:
        """
        Resize logits to original (video_H, video_W) and optionally apply non-overlap constraints.
        any_res_masks: [B, 1, h, w] logits on self._device
        Returns: [B, 1, video_H, video_W] on self._device
        """
        if any_res_masks.shape[-2:] == (self._video_H, self._video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(
                any_res_masks, size=(self._video_H, self._video_W),
                mode="bilinear", align_corners=False
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return video_res_masks

    def _maybe_cache_maskmem_pos_enc(self, out_batched):
        """
        Cache (once) the per-level maskmem_pos_enc slice so we can expand later.
        Mirrors EfficientTAMVideoPredictor._get_maskmem_pos_enc behavior.
        """
        out_maskmem_pos_enc = out_batched.get("maskmem_pos_enc", None)
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in self._constants:
                # store a single-object slice; it is identical across batch objs
                cached = [x[0:1].clone() for x in out_maskmem_pos_enc]
                self._constants["maskmem_pos_enc"] = cached

    @torch.inference_mode()
    def initialize(
        self,
        image,                    # HxWx3 (numpy)
        points_list,              # List[List[[x, y], ...]], length = B
        labels_list,              # List[List[int, ...]], length = B
        normalize_coords: bool = True,
    ):

        # --- basic setup ---
        img_np = np.asarray(image)
        assert img_np.ndim == 3 and img_np.shape[2] == 3, "image must be HxWx3"

        H, W = img_np.shape[:2]
        B = len(points_list)
        assert B > 0, "Need at least one object."
        # enforce equal clicks per object
        num_clicks = [len(p) for p in points_list]
        assert all(nc == num_clicks[0] for nc in num_clicks), f"Unequal clicks per object: {num_clicks}"
        P = num_clicks[0]
        assert P > 0, "At least one click per object is required."

        self._video_H, self._video_W = H, W
        self._storage_device = self._device
        self._B = B

        # --- image -> model size ---
        img_t = torch.as_tensor(img_np, device=self._device).float().permute(2, 0, 1).unsqueeze(0)
        img_t = self._resize_image_to_model(img_t)
        img_t = self._normalize_image(img_t)

        # --- shared backbone once ---
        backbone_out = self.forward_image(img_t)
        _, vfeats_base, vpos_base, fsizes = self._prepare_backbone_features(backbone_out)

        all_outs = []
        for b in range(B):
            # per-object prompts -> tensors
            pts = torch.tensor(points_list[b], dtype=torch.float32, device=self._device)   # [P,2]
            labs = torch.tensor(labels_list[b], dtype=torch.int32,  device=self._device)   # [P]
            if normalize_coords:
                pts = pts / torch.tensor([W, H], device=self._device, dtype=torch.float32)
            pts = pts * float(self.image_size)
            point_inputs = {
                "point_coords": pts.unsqueeze(0),   # [1,P,2]
                "point_labels": labs.unsqueeze(0),  # [1,P]
            }

            # ---- PASS 1: click -> low-res logits (no memory write) ----
            click_out = self.track_step(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=vfeats_base,
                current_vision_pos_embeds=vpos_base,
                feat_sizes=fsizes,
                point_inputs=point_inputs,     # keep the clicks
                mask_inputs=None,
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                track_in_reverse=False,
                run_mem_encoder=False,
            )
            prev_logits = torch.clamp(click_out["pred_masks"], -32.0, 32.0)

            # ---- PASS 2: re-run with SAME clicks + prev_sam_mask_logits, and encode memory ----
            current_out = self.track_step(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=vfeats_base,
                current_vision_pos_embeds=vpos_base,
                feat_sizes=fsizes,
                point_inputs=point_inputs,     # <- MUST be present with prev_sam_mask_logits
                mask_inputs=None,              # <- MUST be None
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                track_in_reverse=False,
                run_mem_encoder=True,          # write memory
                prev_sam_mask_logits=prev_logits,
            )
            all_outs.append(current_out)

        # ---- stack per-object outputs ----
        def _cat_if_tensor_list(x_list):
            if isinstance(x_list[0], torch.Tensor):
                return torch.cat(x_list, dim=0)
            # list of lists (maskmem_pos_enc): cat per level
            return [torch.cat([x[i] for x in x_list], dim=0) for i in range(len(x_list[0]))]

        current_out = {
            "maskmem_features": _cat_if_tensor_list([o["maskmem_features"] for o in all_outs]),
            "maskmem_pos_enc":  _cat_if_tensor_list([o["maskmem_pos_enc"]  for o in all_outs]),
            "pred_masks":       _cat_if_tensor_list([o["pred_masks"]       for o in all_outs]),
            "obj_ptr":          _cat_if_tensor_list([o["obj_ptr"]          for o in all_outs]),
            "object_score_logits": (
                torch.cat([o["object_score_logits"] for o in all_outs], dim=0)
                if all_outs[0].get("object_score_logits") is not None else None
            ),
        }

        # Optional hole fill on low-res logits (same timing as predictor)
        if self.fill_hole_area > 0:
            from efficient_track_anything.utils.misc import fill_holes_in_mask_scores
            current_out["pred_masks"] = fill_holes_in_mask_scores(current_out["pred_masks"], self.fill_hole_area)

        # cache pos-enc slice & store as conditioning memory (batched)
        self._maybe_cache_maskmem_pos_enc(current_out)
        self._output_dict["cond_frame_outputs"].clear()
        self._output_dict["non_cond_frame_outputs"].clear()
        self._output_dict["cond_frame_outputs"][0] = current_out

        # video-resize for return
        video_res = self._to_video_res_and_constrain(current_out["pred_masks"].to(self._device))
        self._initialized = True
        self._t = 0
        return {"frame_idx": 0, "pred_masks_video_res": video_res}




    @torch.inference_mode()
    def step(self, image):
        """
        Track on the NEXT frame (t -> t+1) for all objects in batch.
        Returns mask logits resized to original resolution.
        """
        assert self._initialized, "Call initialize() with the first frame before step()."

        # to tensor, resize to image_size
        import numpy as np
        img_np = np.asarray(image)
        assert img_np.ndim == 3 and img_np.shape[2] == 3, "image must be HxWx3"
        # (We keep original H,W for output resizing; for online streams they may vary; if they do,
        # we follow the first frame's H,W to keep consistent scaling like the video predictor.)
        img_t = torch.as_tensor(img_np, device=self._device)
        if img_t.dtype != torch.float32:
            img_t = img_t.float()
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        img_t = self._resize_image_to_model(img_t)
        img_t = self._normalize_image(img_t)

        # single-image backbone -> expand to batch B
        backbone_out = self.forward_image(img_t)
        backbone_exp = self._expand_backbone_to_batch(backbone_out, self._B)
        _, vfeats, vpos, fsizes = self._prepare_backbone_features(backbone_exp)

        # No points/masks for tracking frames
        current_out = self.track_step(
                        frame_idx=self._t + 1,
                        is_init_cond_frame=False,
                        current_vision_feats=vfeats,
                        current_vision_pos_embeds=vpos,
                        feat_sizes=fsizes,
                        point_inputs=None,
                        mask_inputs=None,
                        output_dict=self._output_dict,  # batched memories
                        num_frames=max(self._t + 1, self.num_maskmem+1),         # <-- correct
                        track_in_reverse=False,
                        run_mem_encoder=True,
                    )


        # Fill holes if requested (same spot as initialize)
        pred_masks = current_out["pred_masks"]  # [B,1,h,w] logits
        if self.fill_hole_area > 0:
            from efficient_track_anything.utils.misc import fill_holes_in_mask_scores
            pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)
            current_out["pred_masks"] = pred_masks

        # Update bank: add this frame as non-conditioning memory (batched)
        non_cond_out = {
            "maskmem_features": current_out["maskmem_features"],
            "maskmem_pos_enc": current_out["maskmem_pos_enc"]
                if current_out["maskmem_pos_enc"] is not None
                else self._expand_cached_maskmem_pos_enc(self._B),
            "pred_masks": current_out["pred_masks"],
            "obj_ptr": current_out["obj_ptr"],
            "object_score_logits": current_out.get("object_score_logits", None),
        }
        self._output_dict["non_cond_frame_outputs"][self._t + 1] = non_cond_out

        # output video-res logits
        video_res = self._to_video_res_and_constrain(pred_masks.to(self._device))
        self._t += 1

        if self._t > self.num_maskmem:
            self._output_dict["non_cond_frame_outputs"].pop(self._t - self.num_maskmem)
        print(self._t, self._output_dict["non_cond_frame_outputs"].keys())
        return {
            "frame_idx": self._t,
            "pred_masks_video_res": video_res,  # [B,1,H,W] logits
        }

    # ------------- tiny utilities -------------

    def _expand_cached_maskmem_pos_enc(self, B: int):
        """
        Expand cached per-level pos enc to batch B (if we ever stored compact slices).
        """
        cached = self._constants.get("maskmem_pos_enc", None)
        if cached is None:
            return None
        return [x.expand(B, -1, -1, -1) for x in cached]



class EfficientTAMOnlineBatchTrackerVOS(EfficientTAMOnlineBatchTracker):
    """Optimized for the VOS setting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_all_components()

    def _compile_all_components(self):
        print("Compiling all components for VOS setting. First time may be very slow.")
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,  # Num. of memories varies
        )

        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

    def forward_image(self, img_batch: torch.Tensor):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the backbone features and pos encoding to enable compilation.
        """
        backbone_out = self.image_encoder(img_batch)
        # Clone to help torch.compile
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
            backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][
                i
            ].clone()
        return backbone_out

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the outputs of prompt_encoder and mask_decoder to enable compilation.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the memories and their pos enc to enable compilation.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = maskmem_out["vision_features"].clone()
        maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc
