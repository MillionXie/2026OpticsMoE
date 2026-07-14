"""Validate optical expert routing before training the sparse MoE.

This script bypasses the electronic router and forces known one-hot/multi-hot
expert selections.  It verifies where the optical energy actually lands and
whether the input image is reproduced at the requested expert apertures.
"""

import argparse
from pathlib import Path

import torch

from data import create_loaders
from model import OpticalMoEClassifier
from utils import BASE_DIR, choose_device, load_yaml, save_json, set_seed, write_rows
from visualization import _save_map, save_expert_entrance


def parse_args():
    parser=argparse.ArgumentParser(description="Validate one/multi-expert optical routing without training.")
    parser.add_argument("--config",default="configs/config.yaml")
    parser.add_argument("--device",default=None)
    parser.add_argument("--output-dir",default="routing_validation")
    parser.add_argument("--smoke-test",action="store_true")
    return parser.parse_args()


def _correlation(left,right):
    left=left.float().reshape(-1);right=right.float().reshape(-1);left=left-left.mean();right=right-right.mean()
    return float((left*right).sum()/(left.norm()*right.norm()+1e-12))


def _route_sets():
    routes=[[index] for index in range(9)]
    routes.extend([list(range(count)) for count in range(2,10)])
    routes.extend([[0,4,8],[2,4,6],[0,3,6],[1,3,4,5,7]])
    unique=[]
    for route in routes:
        if route not in unique:unique.append(route)
    return unique


def _weights(route,device):
    value=torch.zeros(1,9,device=device);value[0,route]=1.0/len(route);return value


def _local_intensity_centroid(field,aperture):
    intensity=field[0,aperture.y0:aperture.y1,aperture.x0:aperture.x1].abs().square().float()
    y,x=torch.meshgrid(
        torch.arange(intensity.shape[0],device=intensity.device,dtype=intensity.dtype),
        torch.arange(intensity.shape[1],device=intensity.device,dtype=intensity.dtype),
        indexing="ij",
    )
    total=intensity.sum()+1e-12
    return torch.stack(((intensity*y).sum()/total,(intensity*x).sum()/total))


def _stage_diagnostics(model,field,route):
    rows=[];stage0_centroids={}
    for stage in range(6):
        ratios=model.expert_energy_ratios(field)[0]
        top_indices=torch.topk(ratios,k=len(route)).indices.sort().values.tolist()
        selected_total=float(ratios[route].sum());unselected=[i for i in range(9) if i not in route]
        centroids={index:_local_intensity_centroid(field,model.layout.expert_apertures[index]) for index in route}
        if stage==0:stage0_centroids={index:value.clone() for index,value in centroids.items()}
        drift={index:float(torch.linalg.vector_norm(value-stage0_centroids[index])) for index,value in centroids.items()}
        rows.append({
            "stage":stage,"selected_energy_ratio":selected_total,
            "unselected_energy_ratio":float(ratios[unselected].sum()) if unselected else 0.0,
            "topk_exact":top_indices==sorted(route),"energy_ratios":[float(value) for value in ratios],
            "selected_local_centroids_yx":{str(index):[float(v) for v in value] for index,value in centroids.items()},
            "selected_centroid_drift_from_stage0_px":{str(index):value for index,value in drift.items()},
            "max_selected_centroid_drift_from_stage0_px":max(drift.values()),
        })
        if stage<5:
            field=model.expert_layers[stage](field)
            if stage<4:field=model.inter_props[stage](field)
    return rows


@torch.no_grad()
def validate(config,device,output_dir,smoke_test=False):
    set_seed(int(config.get("seed",7)));config.setdefault("dataset",{})["num_workers"]=0
    _,test_loader,class_names=create_loaders(config,int(config.get("seed",7)),smoke_test)
    image,label=next(iter(test_loader));image=image[:1].to(device);label=int(label[0])
    model=OpticalMoEClassifier(config,4).to(device).eval();root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    _save_map(image[0,0],root/"input_120.png",f"Validation input: {class_names[label]}","gray","amplitude",normalize=False,vmin=0,vmax=1)
    input_canvas=model.prepare_canvas_input(image);expected_intensity=torch.flip(image[0,0].square(),dims=(-2,-1))
    summaries=[]
    for route in _route_sets():
        weights=_weights(route,device);selected=weights[0]>0;transmission=model.prompt.transmission(weights)
        entrance=model.global_fanout_convolution(input_canvas,transmission);ratios=model.expert_energy_ratios(entrance)[0]
        correlations=[];route_name="k"+str(len(route))+"_"+"_".join(str(value) for value in route);route_dir=root/route_name
        _save_map(model.prompt.amplitude_map(weights)[0],route_dir/"prompt_amplitude.png","Prompt amplitude: one uniform routing weight per 150x150 region","viridis","amplitude",normalize=False,vmin=0,vmax=1)
        _save_map(model.prompt.phase_map(),route_dir/"prompt_phase.png","Prompt phase: one continuous global lens + carrier map","twilight","phase (rad)",normalize=False,vmin=0,vmax=2*torch.pi)
        save_expert_entrance(entrance[0],ratios,selected,model.layout,route_dir/"expert_entrance_and_energy.png")
        for expert_index in route:
            aperture=model.layout.expert_apertures[expert_index];crop=entrance[0,aperture.y0:aperture.y1,aperture.x0:aperture.x1].abs().square()
            correlations.append(_correlation(crop,expected_intensity));_save_map(crop,route_dir/f"expert_{expert_index}_received_image.png",f"Received image at E{expert_index}")
        stages=_stage_diagnostics(model,entrance,route);unselected=[i for i in range(9) if i not in route]
        stage0_centroids=torch.tensor(list(stages[0]["selected_local_centroids_yx"].values()))
        summary={
            "route":route,"top_k":len(route),"stage0_selected_energy_ratio":float(ratios[route].sum()),
            "stage0_unselected_energy_ratio":float(ratios[unselected].sum()) if unselected else 0.0,
            "minimum_received_image_correlation":min(correlations),"received_image_correlations":correlations,
            "all_stages_topk_exact":all(stage["topk_exact"] for stage in stages),"stages":stages,
            "stage0_selected_centroid_range_y_px":float(stage0_centroids[:,0].max()-stage0_centroids[:,0].min()),
            "stage0_selected_centroid_range_x_px":float(stage0_centroids[:,1].max()-stage0_centroids[:,1].min()),
            "max_centroid_drift_through_expert_stack_px":max(stage["max_selected_centroid_drift_from_stage0_px"] for stage in stages),
        }
        # Routing correctness is an energy/localisation property.  Image
        # correlation is retained as an independent fidelity diagnostic; it
        # must not turn a correct equal-power route into a false routing fail.
        summary["routing_passed"]=summary["stage0_selected_energy_ratio"]>=0.95 and summary["all_stages_topk_exact"] and summary["max_centroid_drift_through_expert_stack_px"]<=1.0
        summary["image_fidelity_passed"]=summary["minimum_received_image_correlation"]>=0.50
        summary["passed"]=summary["routing_passed"] and summary["image_fidelity_passed"]
        save_json(summary,route_dir/"metrics.json");summaries.append(summary)
    equal_route=next(row for row in summaries if len(row["route"])==9)
    equal_ratios=equal_route["stages"][0]["energy_ratios"]
    equal_spread=max(equal_ratios)-min(equal_ratios)
    equal_alignment=max(equal_route["stage0_selected_centroid_range_y_px"],equal_route["stage0_selected_centroid_range_x_px"])
    maximum_stack_drift=max(row["max_centroid_drift_through_expert_stack_px"] for row in summaries)
    report={
        "passed":all(row["passed"] for row in summaries) and equal_spread<=0.01 and equal_alignment<=1.0 and maximum_stack_drift<=1.0,"class_name":class_names[label],"routes_tested":len(summaries),
        "geometry":{"architecture":"global_4f_convolution","convolution_distance_m":model.prompt.convolution_distance_m,"focal_length_m":float(config["optics"]["prompt_focal_length_m"]),"phase_coordinate_system":"single_global_grid","amplitude_cells":"3x3_uniform_regions"},
        "equal_amplitude_check":{"expert_energy_ratios":equal_ratios,"max_minus_min":equal_spread,"passed":equal_spread<=0.01},
        "spatial_alignment_check":{"equal_route_centroid_range_y_px":equal_route["stage0_selected_centroid_range_y_px"],"equal_route_centroid_range_x_px":equal_route["stage0_selected_centroid_range_x_px"],"maximum_copy_alignment_range_px":equal_alignment,"passed":equal_alignment<=1.0},
        "parallel_propagation_check":{"maximum_centroid_drift_through_expert_stack_px":maximum_stack_drift,"passed":maximum_stack_drift<=1.0},
        "thresholds":{"stage0_selected_energy_ratio":0.95,"received_image_correlation_diagnostic":0.50,"all_stages_topk_exact":True,"equal_amplitude_energy_max_minus_min":0.01,"copy_alignment_range_px":1.0,"expert_stack_centroid_drift_px":1.0},"routes":summaries,
    }
    save_json(report,root/"routing_validation.json")
    write_rows(root/"routing_validation.csv",[{
        "route":" ".join(map(str,row["route"])),"top_k":row["top_k"],"passed":row["passed"],
        "stage0_selected_energy_ratio":row["stage0_selected_energy_ratio"],"stage0_unselected_energy_ratio":row["stage0_unselected_energy_ratio"],
        "minimum_received_image_correlation":row["minimum_received_image_correlation"],"routing_passed":row["routing_passed"],
        "image_fidelity_passed":row["image_fidelity_passed"],"all_stages_topk_exact":row["all_stages_topk_exact"],
        "stage0_selected_centroid_range_y_px":row["stage0_selected_centroid_range_y_px"],"stage0_selected_centroid_range_x_px":row["stage0_selected_centroid_range_x_px"],
        "max_centroid_drift_through_expert_stack_px":row["max_centroid_drift_through_expert_stack_px"],
    } for row in summaries])
    return report


def main():
    args=parse_args();config_path=Path(args.config);config_path=config_path if config_path.is_absolute() else BASE_DIR/config_path
    report=validate(load_yaml(config_path),choose_device(args.device),BASE_DIR/args.output_dir,args.smoke_test)
    print(f"routing validation passed={report['passed']} routes={report['routes_tested']} output={BASE_DIR/args.output_dir}")
    return 0 if report["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
