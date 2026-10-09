"""Targeted checks for continuation branches and actual chi/square depth.

No entropy algorithms or older verification results are edited. This writes
revision-v6 results and a camera/scene manifest for composing four panels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import build_schematics as b
import depth_renderer as r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-assets",action="store_true")
    args=parser.parse_args()
    checks=[]
    def check(name,condition,**data):
        assert bool(condition),(name,data)
        checks.append(dict(name=name,passed=True,**data))

    first=b.step1();double=b.step2(first);traced=b.step3(double)
    moved=b.step4(traced,6.);flies=b.step5(moved);edge=b.step6(traced);replicas=b.step7(flies,6.)
    scenes={"01":first,"02":double,"02_bis":b.step2_bis(double),
        "03":traced,"03_bis":b.step3_bis(traced),"04":moved,
        "05":flies,"05_bis":b.step5_bis(flies),"06":edge,"06_bis":b.step6_bis(edge),
        "07":replicas,"07_bis":b.step7_bis(replicas)}
    counts={}
    for name in ("02_bis","03_bis","05_bis","06_bis","07_bis"):
        counts[name]=len({e["ellipsis_id"] for e in scenes[name] if e.get("role")=="ellipsis"})
    check("continuation_group_counts",counts==dict(zip(
        ("02_bis","03_bis","05_bis","06_bis","07_bis"),(6,6,15,3,10))),counts=counts)
    check("two_and_three_branches_preserve_their_distinct_physical_legs",
        not any(e.get("role") in ("physical_open","physical_trace") for e in scenes["02_bis"]) and
        sum(e.get("role")=="physical_trace" for e in scenes["03_bis"])==24 and
        sum(e.get("role")=="physical_open" for e in scenes["03_bis"])==48)
    for branch,base in (("02_bis","02"),("03_bis","03"),("05_bis","05"),("06_bis","06"),("07_bis","07")):
        retained={e["id"]:e for e in scenes[branch]}
        assert all(retained[e["id"]]==e for e in scenes[base]),branch
    check("continuation_branches_only_add_elements",True)
    new_context=[e for e in scenes["06_bis"] if e.get("ellipsis_id")=="ellipsis/context_left"]
    check("left_context_ellipsis_has_quarter_opacity_underlay",
        len(new_context)==3 and all(e.get("context_background") and np.isclose(e["alpha"],.25) for e in new_context))
    groups5={e["ellipsis_id"].split("/")[2] for e in scenes["05_bis"] if e.get("role")=="ellipsis"}
    groups7={e["ellipsis_id"].split("/")[2] for e in scenes["07_bis"] if e.get("role")=="ellipsis"}
    expected={"left_original","left_copy","right_middle","right_top","right_bottom"}
    check("five_regions_and_five_full_or_half_chains_all_extended",groups5==expected and groups7==expected)

    pr,pu,rr,ru,eye=r.camera_frame(b.PROJECTION,depth_scale=b.DEPTH_SCALE,depth_angle=b.DEPTH_ANGLE)
    projection=np.array([pr,pu])
    for name in counts:
        groups={}
        for e in scenes[name]:
            if e.get("role")=="ellipsis":groups.setdefault(e["ellipsis_id"],[]).append(e)
        for uid,group in groups.items():
            points=np.array([e["center"] for e in sorted(group,key=lambda e:e["id"])])@projection.T
            assert np.allclose(np.linalg.norm(np.diff(points,axis=0),axis=1),.14),(name,uid)
    check("all_continuation_symbols_retain_equal_screen_dot_spacing",True,screen_spacing=.14)
    check("chi_tubes_never_have_symbol_owner_masks",all(
        not e.get("symbol_owners") for scene in scenes.values() for e in scene if e.get("role")=="chi_chain"))
    check("thin_attached_wires_keep_symbol_masks",all(
        e.get("symbol_owners") for scene in (edge,replicas) for e in scene if e.get("role") in ("cut_half","flying")))

    camera=dict(projection=b.PROJECTION,depth_scale=b.DEPTH_SCALE,depth_angle=b.DEPTH_ANGLE)
    def probe(elements,point):
        image,meta=r._render_arrays(r._expand_elements(elements),401,401,1,30,-60,**camera)
        scale=meta["world_units_per_output_pixel"];bounds=meta["screen_bounds"]
        col=int(np.floor((point[0]-bounds["xmin"])/scale))
        row=int(np.floor((bounds["ymax"]-point[1])/scale))
        return image[row,col]

    face=[.2,.3,.4]
    square=dict(kind="square",center=[0.,0.,0.],side=.3,color=face,
        shade=False,id="face",symbol_id="face")
    tube=dict(kind="cylinder",p0=[0.,-1.,0.],p1=[0.,1.,0.],radius=b.BLUE_RADIUS,
        color=b.BLUE,shade=False,role="chi_chain")
    # These pixels lie inside the square. The same y-axis tube is in front
    # on the negative-y side and behind the square on the positive-y side.
    front=projection@np.array([0.,-.1,0.])
    rear=projection@np.array([0.,.1,0.])
    check("tube_in_front_of_square_is_visible_by_ray_depth",
        np.allclose(probe([square,tube],front),b.BLUE))
    check("square_in_front_of_tube_hides_it_by_ray_depth",
        np.allclose(probe([square,tube],rear),face))
    for ordering in ([square,tube],[tube,square]):
        assert np.allclose(probe(ordering,front),b.BLUE)
        assert np.allclose(probe(ordering,rear),face)
    check("mutual_tube_square_occlusion_is_draw_order_independent",True)
    for half in ("upper","lower"):
        half_tube=dict(tube,half=half)
        assert np.allclose(probe([square,half_tube],front),b.BLUE),half
        assert np.allclose(probe([square,half_tube],rear),face),half
    check("both_clipped_half_tubes_use_the_same_actual_ray_depth",True)
    red_wire=dict(kind="cylinder",p0=(.3*eye+[-.5,0.,0.]).tolist(),
        p1=(.3*eye+[.5,0.,0.]).tolist(),radius=.015,color=b.RED,
        shade=False,symbol_owners=["face"])
    check("attached_thin_wire_mask_is_retained_without_affecting_chi_tubes",
        np.allclose(probe([square,red_wire],[0.,0.]),face) and
        np.allclose(probe([square,dict(red_wire,symbol_owners=[])],[0.,0.]),b.RED))

    manifest=dict(projection_matrix=projection.tolist(),toward_eye=eye.tolist(),
        world_unit="honeycomb bond length 1",scene_order=list(scenes),
        four_panel_candidates=["02_bis","05_bis","06_bis","07_bis"],
        alternative_upper_left="03_bis",panels={})
    if not args.skip_assets:
        margins={}
        for name,scene in scenes.items():
            scene_file=ROOT/"coordinates"/f"step_{name}.json"
            camera_file=ROOT/"coordinates"/f"camera_{name}.json"
            figure=ROOT/"figures"/f"step_{name}.png"
            meta=json.loads(camera_file.read_text(encoding="utf-8"))
            retained=json.loads(scene_file.read_text(encoding="utf-8"))
            assert len(retained)==len(scene) and np.allclose(meta["screen_projection_matrix"],projection)
            pixels=np.array(Image.open(figure).convert("RGB"))
            rows,cols=np.nonzero(np.any(pixels<250,axis=2))
            margin=[int(cols.min()),int(pixels.shape[1]-1-cols.max()),
                    int(rows.min()),int(pixels.shape[0]-1-rows.max())]
            assert max(margin)<=5,(name,margin)
            margins[name]=margin
            manifest["panels"][name]=dict(scene_json=str(scene_file),camera_json=str(camera_file),
                figure=str(figure),width=meta["width"],height=meta["height"],
                world_units_per_output_pixel=meta["world_units_per_output_pixel"],
                screen_bounds=meta["screen_bounds"],world_bounds=meta["world_bounds"])
            if name in ("06","06_bis","07","07_bis"):
                assert meta["chi_square_occlusion_policy"]=="chi_tubes_unmasked_share_actual_foreground_ray_depth_with_square_faces"
                assert meta["chi_chain_true_depth_primitive_count"]==(10 if name.startswith("06") else 25)
        check("all_twelve_retained_scenes_match_current_camera_and_minimal_borders",True,margins=margins)
        check("render_metadata_discloses_true_chi_depth",True)
    report=dict(all_checks_passed=True,check_count=len(checks),checks=checks,
        asset_checks_skipped=args.skip_assets)
    output=Path(__file__).with_name("projection_verification_v6.json")
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest_file=Path(__file__).with_name("render_manifest_v6.json")
    manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"PASS: {len(checks)} targeted continuation and cylinder/square depth checks.")
    print(output)
    print(manifest_file)


if __name__=="__main__":main()
