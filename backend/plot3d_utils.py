# backend/plot3d_utils.py
from __future__ import annotations
import base64
import gzip
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


def generate_plotly_from_mask_b64(mask_base64: str) -> dict:
    decoded = base64.b64decode(mask_base64)
    if decoded[:2] == b"\x1f\x8b":
        decoded = gzip.decompress(decoded)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(decoded)
        tmp_path = Path(tmp.name)

    img = nib.load(str(tmp_path))
    data = np.round(img.get_fdata()).astype(int)
    tmp_path.unlink(missing_ok=True)

    if np.sum(data > 0) < 10:
        raise ValueError("해마 데이터가 없습니다.")

    coords = np.argwhere(data > 0)
    center = np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])])

    traces = []
    label_positions = {}
    all_x, all_y, all_z = [], [], []

    def create_trace(label_id, color, name, text_label):
        if np.sum(data == label_id) < 10:
            return None

        m = (data == label_id).astype(float)
        m_smooth = gaussian_filter(m, sigma=0.5)

        try:
            verts, faces, _, _ = measure.marching_cubes(m_smooth, level=0.5, step_size=1)
            verts_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T
            verts_centered = verts_xyz - np.array([center[2], center[1], center[0]])

            all_x.extend(verts_centered[:, 0])
            all_y.extend(verts_centered[:, 1])
            all_z.extend(verts_centered[:, 2])

            label_positions[text_label] = {
                "x": float(np.mean(verts_centered[:, 0])),
                "y": float(np.mean(verts_centered[:, 1])),
                "z": float(np.mean(verts_centered[:, 2])),
                "color": color,
            }

            return go.Mesh3d(
                x=verts_centered[:, 0].tolist(),
                y=verts_centered[:, 1].tolist(),
                z=verts_centered[:, 2].tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
                color=color,
                opacity=1.0,
                name=name,
                flatshading=False,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    roughness=0.7,
                    specular=0.1,
                    fresnel=0.5,
                ),
                lightposition=dict(x=1000, y=1000, z=5000),
            )
        except Exception as e:
            print(f"메쉬 오류 ({name}): {e}")
            return None

    t1 = create_trace(1, "#27ae60", "Left Hippocampus", "L")
    if t1:
        traces.append(t1)
    t2 = create_trace(2, "#e74c3c", "Right Hippocampus", "R")
    if t2:
        traces.append(t2)

    if not traces:
        raise RuntimeError("3D 모델 생성 실패")

    fig = go.Figure(data=traces)

    annotations = []
    for txt, pos in label_positions.items():
        annotations.append(
            dict(
                showarrow=False,
                x=pos["x"],
                y=pos["y"],
                z=pos["z"] + 15,
                text=txt,
                font=dict(color=pos["color"], size=24, family="Arial Black"),
                xanchor="center",
                yanchor="bottom",
            )
        )

    padding = 20
    if all_x:
        max_x, max_y, max_z = max(all_x), max(all_y), max(all_z)
        annotations.append(dict(showarrow=False, x=max_x + padding, y=0, z=0, text="x", font=dict(size=14)))
        annotations.append(dict(showarrow=False, x=0, y=max_y + padding, z=0, text="y", font=dict(size=14)))
        annotations.append(dict(showarrow=False, x=0, y=0, z=max_z + padding, text="z", font=dict(size=14)))

    axis_style = dict(
        showgrid=False,
        zeroline=False,
        showbackground=False,
        showticklabels=False,
        visible=False,
        showline=False,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_style),
            yaxis=dict(**axis_style),
            zaxis=dict(**axis_style),
            aspectmode="data",
            bgcolor="white",
            annotations=annotations,
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return json.loads(fig.to_json())
