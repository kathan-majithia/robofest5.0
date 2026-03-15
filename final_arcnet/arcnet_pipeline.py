"""
ARCNET Complete Minefield Detection & Mapping Pipeline
=========================================================
Single script combining:
  1. TFLite landmine detection on captured frames
  2. GPS-based orthomosaic generation (high quality)
  3. 2D minefield map with safe path (A* / Dijkstra)
  4. Interactive 3D Gaussian risk map (Plotly)

Usage:
    python arcnet_pipeline.py

Folder structure expected:
    frames/
    â”œâ”€â”€ frame_0000.jpg ...
    â””â”€â”€ gps_log.json
    best_float32.tflite

Outputs (all in result/):
    result/
    mine_detections.json  per-frame detection + GPS log
    frame_0000.jpg        annotated detection frames
    orthomosaic.jpg       blended satellite-style map
    mine_overlay.jpg      map with mine markers
    debug_positions.jpg   drone centres vs mine positions
    mines_export.json     clean mine GPS export
    minefield_map.png     2D safe path map
    mine_risk_map.html    interactive 3D risk map
"""

import cv2
import os
import json
import math
import threading
import numpy as np
import pyproj
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from ai_edge_litert.interpreter import Interpreter

MAP_WIDTH      = 6.0    # metres
MAP_LENGTH     = 10.0   # metres

INPUT_FOLDER   = "frames"
GPS_LOG_FILE   = "frames/gps_log.json"
OUTPUT_FOLDER  = "result"
MODEL_PATH     = "best_float32.tflite"

# Detection
CONF_THRESHOLD = 0.40
NMS_THRESHOLD  = 0.40
INPUT_SIZE     = 320

# Orthomosaic
OUTPUT_W       = 2400
OUTPUT_H       = 2400
DRONE_ALT_M    = 3.5
CAMERA_FOV_H   = 62.2
CAMERA_FOV_V   = 48.8
GPS_BAR_PX     = 90
EXCL_RADIUS_M  = 0.8

# Minefield map

print("=" * 60)
print("   ARCNET â€” Complete Minefield Detection & Mapping Pipeline")
print("=" * 60)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\n" + "â”€" * 60)
print("  STAGE 1 â€” Landmine Detection (TFLite)")
print("â”€" * 60)

def run_detection(interpreter, input_details, output_details, frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output[0])
    raw_boxes, confidences = [], []
    for i in range(predictions.shape[1]):
        confidence = predictions[4][i]
        if confidence < CONF_THRESHOLD:
            continue
        cx = predictions[0][i]; cy = predictions[1][i]
        bw = predictions[2][i]; bh = predictions[3][i]
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        raw_boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))
    results = []
    if raw_boxes:
        nms_boxes = [[x, y, x2-x, y2-y] for x, y, x2, y2 in raw_boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            results.append((raw_boxes[i], confidences[i]))
    return results

def overlay_info(frame, gps_entry, detections):
    h, w = frame.shape[:2]
    lat  = gps_entry.get("lat");  lon  = gps_entry.get("lon")
    alt  = gps_entry.get("alt");  fix  = gps_entry.get("fix", 0)
    sats = gps_entry.get("sats", 0); fid = gps_entry.get("frame_id", 0)
    ov = frame.copy()
    cv2.rectangle(ov, (0, h-90), (w, h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    fix_color = (0,255,80) if fix >= 3 else (0,80,255)
    fix_text  = "3D FIX" if fix >= 3 else f"NO FIX (type={fix})"
    coord_text = (f"LAT: {lat:.7f}   LON: {lon:.7f}"
                  if lat and lon else "LAT: ---.-------   LON: ---.-------")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, coord_text, (10,h-62), font, 0.58, (255,255,255), 1)
    cv2.putText(frame, f"ALT: {alt}m   MINES: {len(detections)}   SATS: {sats}   FRAME: {fid:04d}",
                (10,h-36), font, 0.52, (200,200,200), 1)
    cv2.putText(frame, fix_text, (10,h-10), font, 0.52, fix_color, 1)
    cx2, cy2 = w-30, 30
    cv2.line(frame,(cx2-12,cy2),(cx2+12,cy2),(0,255,80),1)
    cv2.line(frame,(cx2,cy2-12),(cx2,cy2+12),(0,255,80),1)
    cv2.circle(frame,(cx2,cy2),8,(0,255,80),1)
    for box, conf in detections:
        x1,y1,x2,y2 = box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,f"Landmine {conf:.2f}",(x1,max(y1-10,10)),font,0.6,(0,0,255),2)
    return frame

# Load GPS log
if not os.path.exists(GPS_LOG_FILE):
    print(f"[ERR] GPS log not found: {GPS_LOG_FILE}")
    exit()
with open(GPS_LOG_FILE) as f:
    gps_log = json.load(f)
gps_lookup = {e["frame_id"]: e for e in gps_log}
print(f"  GPS log loaded  : {len(gps_log)} entries")

# Collect frames
frame_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                      if f.startswith("frame_") and f.endswith(".jpg")])
if not frame_files:
    print(f"[ERR] No frames found in {INPUT_FOLDER}/")
    exit()
print(f"  Frames found    : {len(frame_files)}")

# Load model
print("  Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("  Model loaded âœ“\n")
print(f"  Processing {len(frame_files)} frames...")
print("  " + "â”€" * 40)

mine_log = []
for fname in frame_files:
    try:
        frame_id = int(fname.replace("frame_","").replace(".jpg",""))
    except ValueError:
        continue
    frame = cv2.imread(os.path.join(INPUT_FOLDER, fname))
    if frame is None:
        continue
    gps_entry = gps_lookup.get(frame_id, {
        "frame_id": frame_id, "lat": None, "lon": None,
        "alt": None, "fix": 0, "sats": 0, "timestamp": None
    })
    detections = run_detection(interpreter, input_details, output_details, frame)
    frame = overlay_info(frame, gps_entry, detections)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    if detections:
        print(f"  âš   {fname} â†’ {len(detections)} mine(s) detected!")
        entry = {
            "frame": os.path.join(OUTPUT_FOLDER, fname),
            "frame_id": frame_id,
            "timestamp": gps_entry.get("timestamp"),
            "gps": {k: gps_entry.get(k) for k in ["lat","lon","alt","fix","sats"]},
            "detections": []
        }
        for idx, (box, conf) in enumerate(detections):
            x1,y1,x2,y2 = box
            entry["detections"].append({
                "detection_id": idx, "confidence": round(conf,4),
                "bbox": {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            })
            print(f"     â†’ Mine {idx}: conf={conf:.2f}  bbox=[{x1},{y1},{x2},{y2}]")
        mine_log.append(entry)
    else:
        print(f"  âœ“  {fname} â†’ No mines detected.")

log_path = os.path.join(OUTPUT_FOLDER, "mine_detections.json")
with open(log_path, "w") as f:
    json.dump(mine_log, f, indent=2)

total_mines = sum(len(e["detections"]) for e in mine_log)
print(f"\n  {'â”€'*40}")
print(f"  Frames processed     : {len(frame_files)}")
print(f"  Frames with mines    : {len(mine_log)}")
print(f"  Total detections     : {total_mines}")
print(f"  Detection log saved  â†’ {log_path}")
print("\n" + "â”€" * 60)
print("  STAGE 2 â€” GPS-Based Orthomosaic")
print("â”€" * 60)

ORTHO_PATH   = "result/orthomosaic.jpg"
OVERLAY_PATH = "result/mine_overlay.jpg"

valid_frames = [g for g in gps_log if g["lat"] and g["lon"]]
fix3_frames  = [g for g in valid_frames if g.get("fix",0) >= 3]
use_frames   = fix3_frames if fix3_frames else valid_frames
print(f"  GPS frames available : {len(use_frames)}")

# Mine coords with bbox offset
mine_coords = []
for entry in mine_log:
    lat = entry["gps"].get("lat"); lon = entry["gps"].get("lon")
    if not lat or not lon: continue
    for det in entry["detections"]:
        bx1=det["bbox"]["x1"]; by1=det["bbox"]["y1"]
        bx2=det["bbox"]["x2"]; by2=det["bbox"]["y2"]
        mine_px_x = (bx1+bx2)/2.0; mine_px_y = (by1+by2)/2.0
        FRAME_W=1280; FRAME_H=720-GPS_BAR_PX
        mine_coords.append({
            "lat":lat,"lon":lon,"frame_id":entry["frame_id"],
            "confidence":det["confidence"],
            "norm_offset_x": (mine_px_x/FRAME_W)-0.5,
            "norm_offset_y": (mine_px_y/FRAME_H)-0.5,
        })
print(f"  Mine coords loaded   : {len(mine_coords)}")

# Projection
anchor_lat = np.mean([f["lat"] for f in use_frames])
anchor_lon = np.mean([f["lon"] for f in use_frames])
proj = pyproj.Proj(proj='aeqd', lat_0=anchor_lat, lon_0=anchor_lon, datum='WGS84')

ground_w = 2*DRONE_ALT_M*math.tan(math.radians(CAMERA_FOV_H/2))
ground_h = 2*DRONE_ALT_M*math.tan(math.radians(CAMERA_FOV_V/2))
print(f"  Frame coverage       : {ground_w:.2f}m Ã— {ground_h:.2f}m")

for f in use_frames:
    f["local_x"], f["local_y"] = proj(f["lon"], f["lat"])

for m in mine_coords:
    dx, dy = proj(m["lon"], m["lat"])
    m["local_x"] = dx + m["norm_offset_x"]*ground_w
    m["local_y"] = dy - m["norm_offset_y"]*ground_h

all_x = [f["local_x"] for f in use_frames]
all_y = [f["local_y"] for f in use_frames]
gps_spread_x = max(all_x)-min(all_x)
gps_spread_y = max(all_y)-min(all_y)
print(f"  GPS spread           : {gps_spread_x:.3f}m Ã— {gps_spread_y:.3f}m")

if gps_spread_x < ground_w*0.5 and gps_spread_y < ground_w*0.5:
    print("  [WARN] GPS spread small â€” using grid layout")
    n=len(use_frames)
    cols=math.ceil(math.sqrt(n*(OUTPUT_W/OUTPUT_H)))
    step_x=ground_w*0.80; step_y=ground_h*0.80
    for i,fr in enumerate(use_frames):
        fr["local_x"]=(i%cols)*step_x; fr["local_y"]=(i//cols)*step_y
    all_x=[f["local_x"] for f in use_frames]
    all_y=[f["local_y"] for f in use_frames]

min_x=min(all_x)-ground_w*0.6; max_x=max(all_x)+ground_w*0.6
min_y=min(all_y)-ground_h*0.6; max_y=max(all_y)+ground_h*0.6
map_w_m=max_x-min_x; map_h_m=max_y-min_y
px_per_m=min(OUTPUT_W/map_w_m, OUTPUT_H/map_h_m)
actual_w=int(map_w_m*px_per_m); actual_h=int(map_h_m*px_per_m)
print(f"  Canvas               : {actual_w}Ã—{actual_h} px  ({px_per_m:.1f} px/m)")

def world_to_px(lx, ly):
    return (int((lx-min_x)*px_per_m),
            actual_h - int((ly-min_y)*px_per_m))

# Quality helpers
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def enhance_frame(img):
    b,g,r = cv2.split(img.astype(np.float32))
    ma = (b.mean()+g.mean()+r.mean())/3
    for ch,mean in zip([b,g,r],[b.mean(),g.mean(),r.mean()]):
        if mean>0: ch[:] = np.clip(ch*(ma/mean),0,255)
    img = cv2.merge([b,g,r]).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b_ch = cv2.split(lab)
    img = cv2.cvtColor(cv2.merge([clahe.apply(l),a,b_ch]), cv2.COLOR_LAB2BGR)
    return img

def sharpness_score(img):
    return cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()

def sharpen(img):
    blur = cv2.GaussianBlur(img,(0,0),3)
    return cv2.addWeighted(img,1.5,blur,-0.5,0)

# Pre-load + global colour normalisation
print("  Pre-loading frames...")
preloaded={}; global_means=[]
for fd in use_frames:
    p=fd.get("frame", f"{INPUT_FOLDER}/frame_{fd['frame_id']:04d}.jpg")
    img=cv2.imread(p)
    if img is None: img=cv2.imread(p.replace(".jpg",".png"))
    if img is None: continue
    if img.shape[0]>GPS_BAR_PX+50: img=img[:img.shape[0]-GPS_BAR_PX,:]
    preloaded[fd["frame_id"]]=img
    global_means.append(img.mean(axis=(0,1)))

target_mean = np.median(global_means,axis=0) if global_means else np.array([128.,128.,128.])

canvas=np.zeros((actual_h,actual_w,3),dtype=np.float64)
weight_map=np.zeros((actual_h,actual_w),dtype=np.float64)
fw_px=max(1,int(ground_w*px_per_m)); fh_px=max(1,int(ground_h*px_per_m))
placed=0; sharpness_scores=[]

print("  Blending frames...")
for fd in use_frames:
    img=preloaded.get(fd["frame_id"])
    if img is None: continue
    fm=img.mean(axis=(0,1))
    for c in range(3):
        if fm[c]>0:
            img[:,:,c]=np.clip(img[:,:,c].astype(np.float32)*(target_mean[c]/fm[c]),0,255).astype(np.uint8)
    img=enhance_frame(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    med=np.median(gray)
    bm=(gray>min(255,med*2.2)).astype(np.uint8)*255
    if bm.any():
        bm=cv2.dilate(bm,np.ones((5,5),np.uint8))
        img=cv2.inpaint(img,bm,5,cv2.INPAINT_TELEA)
    sharp=sharpness_score(img); sharpness_scores.append(sharp)
    img_r=cv2.resize(img,(fw_px,fh_px),interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
    cx2,cy2=world_to_px(fd["local_x"],fd["local_y"])
    x1,y1=cx2-fw_px//2,cy2-fh_px//2
    ox1,oy1=max(0,x1),max(0,y1)
    ox2,oy2=min(actual_w,x1+fw_px),min(actual_h,y1+fh_px)
    ix1,iy1=ox1-x1,oy1-y1
    ix2,iy2=ix1+(ox2-ox1),iy1+(oy2-oy1)
    if ox2<=ox1 or oy2<=oy1: continue
    wx=np.hanning(fw_px)[ix1:ix2]**2
    wy=np.hanning(fh_px)[iy1:iy2]**2
    w=np.outer(wy,wx).astype(np.float64)*(sharp+1.0)
    canvas[oy1:oy2,ox1:ox2]    +=img_r[iy1:iy2,ix1:ix2]*w[:,:,np.newaxis]
    weight_map[oy1:oy2,ox1:ox2]+=w
    placed+=1

mask=weight_map>0
ortho=np.zeros((actual_h,actual_w,3),dtype=np.uint8)
ortho[mask]=np.clip(canvas[mask]/weight_map[mask,np.newaxis],0,255).astype(np.uint8)

# Crop to content
rows_c=np.where(mask.any(axis=1))[0]; cols_c=np.where(mask.any(axis=0))[0]
if len(rows_c) and len(cols_c):
    pad=10
    r1=max(0,rows_c[0]-pad);  r2=min(actual_h,rows_c[-1]+pad)
    c1=max(0,cols_c[0]-pad);  c2=min(actual_w,cols_c[-1]+pad)
    crop_x=c1; crop_y=r1
    actual_h_o=actual_h; actual_w_o=actual_w
    actual_h=r2-r1; actual_w=c2-c1
    ortho=ortho[r1:r2,c1:c2]; mask=mask[r1:r2,c1:c2]
    def world_to_px(lx,ly):
        return (int((lx-min_x)*px_per_m)-crop_x,
                actual_h_o-int((ly-min_y)*px_per_m)-crop_y)
    print(f"  Cropped to           : {actual_w}Ã—{actual_h} px")
else:
    crop_x=crop_y=0

ortho[~mask]=0
covered=mask.astype(np.uint8)*255
seam_zone=(cv2.dilate(covered,np.ones((7,7),np.uint8))-covered).astype(bool)
if seam_zone.any():
    blurred=cv2.GaussianBlur(ortho,(5,5),0)
    ortho[seam_zone]=blurred[seam_zone]
ortho=sharpen(ortho)

cv2.imwrite(ORTHO_PATH, ortho, [cv2.IMWRITE_JPEG_QUALITY,98])
print(f"  âœ“ Orthomosaic saved  â†’ {ORTHO_PATH}  ({placed} frames)")

# Mine overlay
img=ortho.copy(); ov=img.copy()
excl_r=max(12,int(EXCL_RADIUS_M*px_per_m))
for m in mine_coords:
    px2,py2=world_to_px(m["local_x"],m["local_y"])
    cv2.circle(ov,(px2,py2),excl_r,(0,30,200),-1)
cv2.addWeighted(ov,0.30,img,0.70,0,img)
for i,m in enumerate(mine_coords):
    px2,py2=world_to_px(m["local_x"],m["local_y"])
    cv2.circle(img,(px2,py2),excl_r,(0,60,255),2)
    cv2.circle(img,(px2,py2),excl_r+4,(0,0,0),1)
    cv2.rectangle(img,(px2-22,py2-22),(px2+22,py2+22),(0,0,255),2)
    cv2.circle(img,(px2,py2),5,(0,0,255),-1)
    cv2.circle(img,(px2,py2),5,(255,255,255),1)
    lbl=f"M{i+1}  {m['confidence']:.2f}"
    (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
    cv2.rectangle(img,(px2-2,py2-excl_r-th-8),(px2+tw+4,py2-excl_r),(0,0,0),-1)
    cv2.putText(img,lbl,(px2,py2-excl_r-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),1)

cr_cx,cr_cy,cr_r=actual_w-55,55,35
cv2.circle(img,(cr_cx,cr_cy),cr_r+2,(0,0,0),-1)
cv2.circle(img,(cr_cx,cr_cy),cr_r,(50,50,50),-1)
cv2.circle(img,(cr_cx,cr_cy),cr_r,(200,200,200),1)
cv2.arrowedLine(img,(cr_cx,cr_cy+cr_r-8),(cr_cx,cr_cy-cr_r+8),(255,255,255),2,tipLength=0.35)
cv2.putText(img,"N",(cr_cx-7,cr_cy-cr_r-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
bar_px=max(30,int(1.0*px_per_m)); bx,by=25,actual_h-30
cv2.rectangle(img,(bx-4,by-18),(bx+bar_px+4,by+8),(0,0,0),-1)
cv2.line(img,(bx,by),(bx+bar_px,by),(255,255,255),2)
cv2.line(img,(bx,by-5),(bx,by+5),(255,255,255),2)
cv2.line(img,(bx+bar_px,by-5),(bx+bar_px,by+5),(255,255,255),2)
cv2.putText(img,"1m",(bx,by-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
cv2.imwrite(OVERLAY_PATH,img,[cv2.IMWRITE_JPEG_QUALITY,98])
print(f"  âœ“ Mine overlay saved â†’ {OVERLAY_PATH}")

# Debug positions
dbg=ortho.copy()
for fr in use_frames:
    fx,fy=world_to_px(fr["local_x"],fr["local_y"])
    cv2.circle(dbg,(fx,fy),4,(0,255,0),-1)
for m in mine_coords:
    mx2,my2=world_to_px(m["local_x"],m["local_y"])
    cv2.circle(dbg,(mx2,my2),6,(0,0,255),-1)
cv2.imwrite("result/debug_positions.jpg",dbg,[cv2.IMWRITE_JPEG_QUALITY,95])
print(f"  âœ“ Debug map saved    â†’ result/debug_positions.jpg")

# Export JSON
export={
    "anchor":{"lat":anchor_lat,"lon":anchor_lon},
    "map_coverage_m":{"width":round(map_w_m,2),"height":round(map_h_m,2)},
    "total_mines":len(mine_coords),
    "mines":[{"id":i+1,"lat":m["lat"],"lon":m["lon"],
              "local_x_m":round(m["local_x"],3),"local_y_m":round(m["local_y"],3),
              "confidence":round(m["confidence"],4),"source_frame":m["frame_id"]}
             for i,m in enumerate(mine_coords)]
}
with open("result/mines_export.json","w") as f: json.dump(export,f,indent=2)
print(f"  âœ“ Mine coords saved  â†’ result/mines_export.json")

print("\n" + "â”€" * 60)
print("  STAGE 3 â€” 2D Minefield Map & Safe Path")
print("â”€" * 60)

mines_gps = [(m["lat"],m["lon"]) for m in mine_coords]
if not mines_gps:
    # Fall back to mine_log GPS if no bbox coords
    for entry in mine_log:
        lat=entry["gps"].get("lat"); lon=entry["gps"].get("lon")
        if lat and lon: mines_gps.append((lat,lon))

if mines_gps:
    anc_lat=np.mean([x[0] for x in mines_gps])
    anc_lon=np.mean([x[1] for x in mines_gps])
    p2=pyproj.Proj(proj="aeqd",lat_0=anc_lat,lon_0=anc_lon,datum="WGS84")
    raw_xy=[p2(lon,lat) for lat,lon in mines_gps]
    xs=[p[0] for p in raw_xy]; ys=[p[1] for p in raw_xy]
    min_x2,max_x2=min(xs),max(xs); min_y2,max_y2=min(ys),max(ys)
    sx=max_x2-min_x2 if max_x2!=min_x2 else 1.0
    sy=max_y2-min_y2 if max_y2!=min_y2 else 1.0
    sc_x=(MAP_WIDTH*0.8)/sx; sc_y=(MAP_LENGTH*0.7)/sy
    mines_local=[]
    for ox,oy in raw_xy:
        mx2=(ox-min_x2)*sc_x+MAP_WIDTH*0.1
        my2=(oy-min_y2)*sc_y+MAP_LENGTH*0.2
        mines_local.append((mx2,my2))
        print(f"  [MINE] Local XY â†’ ({mx2:.2f}m, {my2:.2f}m)")

    resolution=0.5; safe_radius=1.0
    gw=int(MAP_WIDTH/resolution); gh=int(MAP_LENGTH/resolution)
    G=nx.Graph()
    edges=[]
    for x in range(gw):
        for y in range(gh):
            if x<gw-1: edges.append(((x,y),(x+1,y),{'weight':1.0}))
            if y<gh-1: edges.append(((x,y),(x,y+1),{'weight':1.0}))
            if x<gw-1 and y<gh-1:
                edges.append(((x,y),(x+1,y+1),{'weight':1.414}))
                edges.append(((x+1,y),(x,y+1),{'weight':1.414}))
    G.add_edges_from(edges)
    for x in range(gw):
        for y in range(gh):
            rx,ry=x*resolution,y*resolution
            for mx2,my2 in mines_local:
                if np.sqrt((rx-mx2)**2+(ry-my2)**2)<=safe_radius:
                    if (x,y) in G: G.remove_node((x,y)); break

    ideal_col=int((MAP_WIDTH/2)/resolution); start_node=None
    for row in range(gh):
        for delta in range(gw):
            for col in [ideal_col-delta,ideal_col+delta]:
                if 0<=col<gw and (col,row) in G:
                    start_node=(col,row); break
            if start_node: break
        if start_node: break

    physical_path=[]; target_x=target_y=None
    if start_node:
        G.add_node("FINISH_LINE")
        top_nodes=[(x,gh-1) for x in range(gw) if (x,gh-1) in G]
        if not top_nodes:
            for row in range(gh-2,-1,-1):
                top_nodes=[(x,row) for x in range(gw) if (x,row) in G]
                if top_nodes: break
        if top_nodes:
            G.add_edges_from([(n,"FINISH_LINE",{'weight':0.0}) for n in top_nodes])
            try:
                path=nx.shortest_path(G,source=start_node,target="FINISH_LINE",weight="weight")
                path.pop()
                physical_path=[(p[0]*resolution,p[1]*resolution) for p in path]
                target_x,target_y=physical_path[-1]
                print(f"  [SUCCESS] Safe exit â†’ X={target_x:.1f}m, Y={target_y:.1f}m")
            except nx.NetworkXNoPath:
                print("  [WARN] No safe path found")

    fig,ax=plt.subplots(figsize=(6,10))
    ax.set_xlim(0,MAP_WIDTH); ax.set_ylim(0,MAP_LENGTH)
    ax.set_aspect('equal'); ax.grid(True,linestyle='--',alpha=0.4)
    for mx2,my2 in mines_local:
        ax.plot(mx2,my2,'kx',markersize=6)
        ax.add_patch(patches.Circle((mx2,my2),radius=safe_radius,color='red',alpha=0.3))
    if physical_path:
        px_,py_=zip(*physical_path)
        ax.plot(px_,py_,'g-',linewidth=2.5,label="Shortest Safe Route")
        ax.plot(target_x,target_y,'mo',markersize=8,label="Optimal Exit Point")
    ax.plot(MAP_WIDTH/2,0,'bo',markersize=8,label='Start Point')
    ax.axhline(y=MAP_LENGTH,color='purple',linestyle='--',alpha=0.5,label='Target Boundary')
    ax.set_xlabel('Width (metres)'); ax.set_ylabel('Length (metres)')
    ax.set_title(f'{MAP_LENGTH}m Ã— {MAP_WIDTH}m Drone-Surveyed Minefield')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha=1.0,
              facecolor='white',edgecolor='black')
    plt.savefig("result/minefield_map.png",dpi=300,bbox_inches='tight')
    plt.close()
    print(f"  âœ“ 2D map saved       â†’ result/minefield_map.png")
else:
    print("  [SKIP] No mine GPS data for 2D map")

print("\n" + "â”€" * 60)
print("  STAGE 4 â€” Interactive 3D Risk Map (Plotly)")
print("â”€" * 60)

if mines_gps:
    sigma=0.8; max_risk=1000.0; safe_threshold=100.0
    gw2=int(MAP_WIDTH/resolution); gh2=int(MAP_LENGTH/resolution)
    X_grid=np.linspace(0,MAP_WIDTH,gw2); Y_grid=np.linspace(0,MAP_LENGTH,gh2)
    X_mesh,Y_mesh=np.meshgrid(X_grid,Y_grid)
    Z_risk=np.zeros_like(X_mesh)
    for mx2,my2 in mines_local:
        Z_risk+=max_risk*np.exp(-((X_mesh-mx2)**2+(Y_mesh-my2)**2)/(2*sigma**2))

    G2=nx.Graph()
    for x in range(gw2):
        for y in range(gh2):
            if Z_risk[y,x]<safe_threshold: G2.add_node((x,y))
    for x in range(gw2):
        for y in range(gh2):
            if (x,y) not in G2: continue
            if x<gw2-1 and (x+1,y) in G2:
                G2.add_edge((x,y),(x+1,y),weight=1.0+Z_risk[y,x+1])
            if y<gh2-1 and (x,y+1) in G2:
                G2.add_edge((x,y),(x,y+1),weight=1.0+Z_risk[y+1,x])
            if x<gw2-1 and y<gh2-1 and (x+1,y+1) in G2:
                G2.add_edge((x,y),(x+1,y+1),weight=1.414+Z_risk[y+1,x+1])
            if x>0 and y<gh2-1 and (x-1,y+1) in G2:
                G2.add_edge((x,y),(x-1,y+1),weight=1.414+Z_risk[y+1,x-1])

    ideal2=int((MAP_WIDTH/2)/resolution); sn=None
    for row in range(gh2):
        for delta in range(gw2):
            for col in [ideal2-delta,ideal2+delta]:
                if 0<=col<gw2 and (col,row) in G2: sn=(col,row); break
            if sn: break
        if sn: break

    top2=[(x,gh2-1) for x in range(gw2) if (x,gh2-1) in G2]
    if not top2:
        for row in range(gh2-2,-1,-1):
            top2=[(x,row) for x in range(gw2) if (x,row) in G2]
            if top2: break

    def h_astar(a,b):
        if not isinstance(a,tuple) or not isinstance(b,tuple): return 0.0
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    px3,py3,pz3=[],[],[]
    finish_reachable=False
    if sn and top2:
        G2.add_edges_from([(n,"FINISH_LINE",{'weight':0.0}) for n in top2])
        reachable=nx.node_connected_component(G2,sn)
        finish_reachable="FINISH_LINE" in reachable
        print(f"  Start node           : ({sn[0]*resolution:.1f}m, {sn[1]*resolution:.1f}m)")
        print(f"  Safe path reachable  : {finish_reachable}")
        if finish_reachable:
            try:
                best=min(top2,key=lambda n:abs(n[0]-sn[0]))
                path3=nx.astar_path(G2,source=sn,target=best,heuristic=h_astar,weight='weight')
                px3=[p[0]*resolution for p in path3]
                py3=[p[1]*resolution for p in path3]
                pz3=[Z_risk[p[1],p[0]]+20 for p in path3]
                print(f"  [SUCCESS] A* path    : {len(path3)} steps â†’ exit ({px3[-1]:.1f}m, {py3[-1]:.1f}m)")
            except Exception as e:
                print(f"  [WARN] A* error: {e}")
        else:
            print("  [WARN] No safe path â€” computing lowest-risk fallback...")
            G3=nx.Graph()
            for x in range(gw2):
                for y in range(gh2): G3.add_node((x,y))
            for x in range(gw2):
                for y in range(gh2):
                    if x<gw2-1: G3.add_edge((x,y),(x+1,y),weight=1.0+Z_risk[y,x+1])
                    if y<gh2-1: G3.add_edge((x,y),(x,y+1),weight=1.0+Z_risk[y+1,x])
                    if x<gw2-1 and y<gh2-1: G3.add_edge((x,y),(x+1,y+1),weight=1.414+Z_risk[y+1,x+1])
                    if x>0 and y<gh2-1: G3.add_edge((x,y),(x-1,y+1),weight=1.414+Z_risk[y+1,x-1])
            bf=min([(x,gh2-1) for x in range(gw2)],key=lambda n:abs(n[0]-sn[0]))
            try:
                path3=nx.astar_path(G3,source=sn,target=bf,heuristic=h_astar,weight='weight')
                px3=[p[0]*resolution for p in path3]
                py3=[p[1]*resolution for p in path3]
                pz3=[Z_risk[p[1],p[0]]+20 for p in path3]
                total_r=sum(Z_risk[p[1],p[0]] for p in path3)
                print(f"  [FALLBACK] Lowest-risk path: {len(path3)} steps, risk={total_r:.1f}")
            except Exception as e:
                print(f"  [WARN] Fallback failed: {e}")

    fig3=go.Figure()
    fig3.add_trace(go.Surface(z=Z_risk,x=X_mesh,y=Y_mesh,colorscale='YlOrRd',
                              opacity=0.8,showscale=True,colorbar=dict(title="Risk Level")))
    if px3:
        col3='#00FF00' if finish_reachable else '#FF8800'
        lbl3='Optimal Safe Path (A*)' if finish_reachable else 'âš  Lowest-Risk Path'
        fig3.add_trace(go.Scatter3d(x=px3,y=py3,z=pz3,mode='lines',
                                    line=dict(color=col3,width=10),name=lbl3))
    fig3.add_trace(go.Scatter3d(
        x=[m[0] for m in mines_local],y=[m[1] for m in mines_local],
        z=[max_risk]*len(mines_local),mode='markers',
        marker=dict(size=5,color='black',symbol='diamond'),name='Detected Mines'))
    if sn:
        fig3.add_trace(go.Scatter3d(x=[sn[0]*resolution],y=[sn[1]*resolution],z=[20],
                                    mode='markers',marker=dict(size=7,color='blue',symbol='circle'),
                                    name='Start Point'))
    title3=("Interactive 3D Minefield Risk Map" +
            (" | âœ… Safe Path Found" if finish_reachable and px3
             else " | âš  Lowest Risk Route" if px3
             else " | âŒ No Path"))
    fig3.update_layout(title=title3,
                       scene=dict(xaxis_title='Width (m)',yaxis_title='Length (m)',
                                  zaxis_title='Danger Penalty',aspectmode='manual',
                                  aspectratio=dict(x=1,y=2,z=0.5),
                                  camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),
                       template='plotly_dark')
    fig3.write_html("result/mine_risk_map.html")
    print(f"  âœ“ 3D map saved       â†’ result/mine_risk_map.html")
else:
    print("  [SKIP] No mine GPS data for 3D map")

print("\n" + "â•" * 60)
print("  PIPELINE COMPLETE")
print("â•" * 60)
print(f"  Frames processed     : {len(frame_files)}")
print(f"  Mines detected       : {total_mines}")
print(f"  Orthomosaic          â†’ result/orthomosaic.jpg")
print(f"  Mine overlay         â†’ result/mine_overlay.jpg")
print(f"  2D safe path map     â†’ result/minefield_map.png")
print(f"  3D risk map          â†’ result/mine_risk_map.html")
print(f"  Mine GPS coords      â†’ result/mines_export.json")
print(f"\n  View all outputs at  â†’ http://<pi-ip>:8080/result/")
print("â•" * 60)