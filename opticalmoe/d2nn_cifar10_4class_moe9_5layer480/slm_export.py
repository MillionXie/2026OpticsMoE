import sys
from pathlib import Path

import torch

from utils import BASE_DIR, save_json

OPTICALMOE_ROOT=BASE_DIR.parent
if str(OPTICALMOE_ROOT) not in sys.path: sys.path.insert(0,str(OPTICALMOE_ROOT))
from slm_bmp import export_plane_bmp


@torch.no_grad()
def export_best_slm_package(model,loader,checkpoint_path,output_dir,config,device,class_names):
    cfg=config.get("slm_export",{})
    if not bool(cfg.get("enabled",True)):return None
    payload=torch.load(checkpoint_path,map_location=device);model.load_state_dict(payload["model_state_dict"]);model.eval()
    selected=None;fallback=None
    for images,labels in loader:
        images=images.to(device);labels=labels.to(device);logits=model(images);preds=logits.argmax(1)
        for i in range(len(images)):
            score=float(logits[i,labels[i]].item());candidate=(score,images[i:i+1].detach(),int(labels[i]),int(preds[i]))
            if fallback is None or score>fallback[0]:fallback=candidate
            if preds[i]==labels[i] and (selected is None or score>selected[0]):selected=candidate
    selected=selected or fallback
    if selected is None:raise RuntimeError("Cannot export from empty loader.")
    score,image,true_label,pred_label=selected
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scale=int(cfg.get("scale_factor",2));width=int(cfg.get("slm_width",1920));height=int(cfg.get("slm_height",1200))
    active=model.layout.active_aperture
    input_active=model.prepare_canvas_input(image).abs()[0,active.y0:active.y1,active.x0:active.x1]
    files=[export_plane_bmp(input_active,out/"input_amplitude_active450.bmp","amplitude",scale,width,height)]
    _,items=model(image,return_intermediates=True)
    prompt_amplitude=items["prompt_amplitude"][0,active.y0:active.y1,active.x0:active.x1]
    prompt_phase=items["prompt_phase"][0,active.y0:active.y1,active.x0:active.x1]
    files.append(export_plane_bmp(prompt_amplitude,out/"prompt_amplitude_active450.bmp","amplitude",scale,width,height))
    files.append(export_plane_bmp(prompt_phase,out/"prompt_phase_active450.bmp","phase",scale,width,height))
    phase_stack=model.phase_stack().detach().cpu()
    mosaics=model.expert_phase_mosaics().detach().cpu()
    for layer_index,mosaic in enumerate(mosaics,start=1):
        files.append(export_plane_bmp(mosaic,out/f"expert_layer_{layer_index:02d}_mosaic_active450.bmp","phase",scale,width,height))
    global_fc_phase=items["global_fc_phase"].detach().cpu()
    files.append(export_plane_bmp(global_fc_phase,out/"global_fc_phase_active450.bmp","phase",scale,width,height))
    torch.save({"input_active450":input_active.cpu(),"prompt_amplitude":prompt_amplitude.cpu(),"prompt_phase":prompt_phase.cpu(),"routing_weights":items["routing_weights"][0].cpu(),"routing_selected_indices":items["routing_selected_indices"][0].cpu(),"expert_phase_stack":phase_stack,"expert_phase_mosaics":mosaics,"global_fc_phase":global_fc_phase},out/"raw_optical_planes.pt")
    metadata={
        "checkpoint":str(checkpoint_path),"checkpoint_epoch":int(payload.get("epoch",-1)),
        "true_label":true_label,"true_name":class_names[true_label],"pred_label":pred_label,"pred_name":class_names[pred_label],"selection_score":score,
        "source_pixel_size_um":float(cfg.get("source_pixel_size_um",16.0)),"slm_pixel_size_um":float(cfg.get("slm_pixel_size_um",8.0)),
        "active_source_shape":[450,450],"active_slm_shape":[900,900],"slm_size_wh":[width,height],
        "slm_center_padding":{"left":(width-900)//2,"right":width-900-(width-900)//2,"top":(height-900)//2,"bottom":height-900-(height-900)//2},
        "phase_encoding":"uint8 round((phase mod 2pi) * 255 / 2pi)","amplitude_encoding":"uint8 round(clamp(amplitude,0,1)*255)","files":files,
        "routing_top_k":int(items["routing_selected_indices"].shape[1]),"routing_selected_indices":[int(v) for v in items["routing_selected_indices"][0].cpu()],"routing_weights":[float(v) for v in items["routing_weights"][0].cpu()],
    }
    save_json(metadata,out/"manifest.json");return metadata
