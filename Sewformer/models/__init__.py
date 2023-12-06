from .garment_detr_2d import build as build_former
from .garment_backbone import build as build_backbone

def build_model(args):
    if args["NN"]["model"] == "GarmentBackbone":
        return build_backbone(args)
    else:
        return build_former(args)