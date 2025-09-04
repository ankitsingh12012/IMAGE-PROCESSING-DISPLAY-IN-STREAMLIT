import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import math
import pandas as pd
from PIL import Image

# ---------- Helper Functions ----------

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_polygon_area(points):
    x, y = zip(*points)
    return 0.5 * abs(
        sum(
            x[i] * y[(i + 1) % len(points)]
            - x[(i + 1) % len(points)] * y[i]
            for i in range(len(points))
        )
    )

def calculate_internal_angles(points):
    angles = []
    for i in range(len(points)):
        p0 = np.array(points[i - 1])
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % len(points)])
        v1, v2 = p0 - p1, p2 - p1
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(math.degrees(math.acos(np.clip(cosang, -1, 1))))
    return angles

def best_fit_circle(points):
    A, B = [], []
    for x, y in points:
        A.append([x, y, 1])
        B.append(-(x * x + y * y))
    A = np.array(A)
    B = np.array(B).reshape(-1, 1)
    try:
        a, b, c = np.linalg.lstsq(A, B, rcond=None)[0].flatten()
        cx, cy = -0.5 * a, -0.5 * b
        r = math.sqrt((a * a + b * b) / 4 - c)
        return (int(cx), int(cy)), int(r)
    except:
        return None, None

def best_fit_ellipse(points):
    if len(points) < 5:
        return None
    arr = np.array(points, dtype=np.int32)
    try:
        (cx, cy), (w, h), ang = cv2.fitEllipse(arr)
        return (int(cx), int(cy)), (int(w/2), int(h/2)), ang
    except:
        return None

# ---------- Streamlit UI ----------

st.set_page_config(layout="wide")
st.title("📏 Interactive Image Measurement Tool")

# Persist background in session
if 'bg_img' not in st.session_state:
    st.session_state.bg_img = None

# Image upload
uploaded = st.file_uploader("📷 Upload an image", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

base = np.array(Image.open(uploaded).convert("RGB"))
h, w = base.shape[:2]
if st.session_state.bg_img is None or st.session_state.get('uploaded') != uploaded:
    st.session_state.bg_img = base.copy()
    st.session_state.uploaded = uploaded

# Sidebar controls
st.sidebar.header("Settings")
shape = st.sidebar.selectbox("Shape to measure:", ["Line","Polygon","Best-Fit Circle","Best-Fit Ellipse"])
ref_val = st.sidebar.number_input(
    "Reference length:", min_value=0.000001, value=1.0,
    format="%.9f", step=0.000001
)
unit = st.sidebar.selectbox("Unit:", ["mm","cm","inches","pixels"])
if st.sidebar.button("Clear All"):
    st.session_state.bg_img = base.copy()

# Interactive canvas
canvas_res = st_canvas(
    background_image=Image.fromarray(st.session_state.bg_img),
    height=h, width=w,
    drawing_mode="point",
    stroke_color="#FF0000",
    stroke_width=5,
    update_streamlit=True,
    key="canvas",
)

# Process clicks and annotate
if canvas_res.json_data and canvas_res.json_data.get("objects"):
    pts = [(int(o['left']), int(o['top'])) for o in canvas_res.json_data['objects']]
    if len(pts) >= 2:
        ref_pts = pts[:2]
        shape_pts = pts[2:]

        # calculate scale
        pix = calculate_distance(ref_pts[0], ref_pts[1])
        if pix < 1e-6:
            st.warning("⚠️ Reference points too close. Re-select two distinct points.")
            st.session_state.bg_img = base.copy()
            st.experimental_rerun()
        scale = ref_val / pix

        # annotate on fresh copy
        img = base.copy()
        # draw reference
        cv2.line(img, ref_pts[0], ref_pts[1], (0,255,0), 2)
        cv2.putText(
            img,
            f"Ref={ref_val:.6f}{unit}",
            (ref_pts[1][0]+5, ref_pts[1][1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
        )
        # label points
        labels = [f"P{i+1}" for i in range(len(pts))]
        for i, p in enumerate(pts):
            cv2.circle(img, p, 5, (0,0,255), -1)
            cv2.putText(
                img,
                labels[i],
                (p[0]+5, p[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1
            )

        # prepare tables
        tables = {}

        # draw shape and compute tables
        if shape == "Line" and len(shape_pts) >= 2:
            a, b = shape_pts[0], shape_pts[1]
            d = calculate_distance(a, b) * scale
            cv2.line(img, a, b, (255,255,0), 2)
            cv2.putText(
                img,
                f"{d:.2f}{unit}",
                ((a[0]+b[0])//2, (a[1]+b[1])//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2
            )
            tables['Line Length'] = [f"{d:.4f} {unit}"]

        elif shape == "Polygon" and len(shape_pts) >= 3:
            sides = []
            for i in range(len(shape_pts)):
                p1, p2 = shape_pts[i], shape_pts[(i+1)%len(shape_pts)]
                cv2.line(img, p1, p2, (255,0,0), 2)
                s = calculate_distance(p1, p2) * scale
                sides.append(s)
                mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                cv2.putText(
                    img,
                    f"{s:.2f}{unit}",
                    mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1
                )
            angles = calculate_internal_angles(shape_pts)
            for i, a in enumerate(angles):
                cv2.putText(
                    img,
                    f"{a:.1f}°",
                    shape_pts[i],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1
                )
            area = calculate_polygon_area(shape_pts) * (scale ** 2)
            # side table
            side_tbl = pd.DataFrame({f"Side {i+1}":[f"{s:.4f} {unit}"] for i, s in enumerate(sides)})
            st.subheader("Polygon Side Lengths")
            st.table(side_tbl)
            # angle table
            angle_tbl = pd.DataFrame({f"∠{i+1}":[f"{a:.2f}°"] for i, a in enumerate(angles)})
            st.subheader("Internal Angles")
            st.table(angle_tbl)
            # area table
            area_tbl = pd.DataFrame({"Area":[f"{area:.4f} {unit}²"]})
            st.subheader("Area")
            st.table(area_tbl)

        elif shape == "Best-Fit Circle" and len(shape_pts) >= 3:
            center, r = best_fit_circle(shape_pts)
            if center:
                d = 2 * r * scale
                area = math.pi * (d/2) ** 2
                cv2.circle(img, center, r, (255,0,255), 2)
                cv2.putText(
                    img,
                    f"D={d:.2f}{unit}",
                    (center[0]+5, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1
                )
                diam_tbl = pd.DataFrame({"Diameter":[f"{d:.4f} {unit}"]})
                st.subheader("Diameter")
                st.table(diam_tbl)
                area_tbl = pd.DataFrame({"Area":[f"{area:.4f} {unit}²"]})
                st.subheader("Area")
                st.table(area_tbl)

        elif shape == "Best-Fit Ellipse" and len(shape_pts) >= 5:
            res = best_fit_ellipse(shape_pts)
            if res:
                c, axes, ang_val = res
                maj = axes[0] * 2 * scale
                mnr = axes[1] * 2 * scale
                area = math.pi * (maj/2) * (mnr/2)
                cv2.ellipse(
                    img,
                    (c, (axes[0]*2, axes[1]*2), ang_val),
                    (0,255,255), 2
                )
                cv2.putText(
                    img,
                    f"A={area:.2f}{unit}²",
                    (c[0]+5, c[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1
                )
                axes_tbl = pd.DataFrame({
                    "Major Axis":[f"{maj:.4f} {unit}"],
                    "Minor Axis":[f"{mnr:.4f} {unit}"]
                })
                st.subheader("Axes")
                st.table(axes_tbl)
                area_tbl = pd.DataFrame({"Area":[f"{area:.4f} {unit}²"]})
                st.subheader("Area")
                st.table(area_tbl)

        # display annotated image
        st.image(img, use_column_width=True)

        # distance between any two points
        if len(pts) >= 2:
            st.subheader("Distance Between Selected Points")
            p1 = st.selectbox("First point:", labels)
            p2 = st.selectbox("Second point:", [l for l in labels if l != p1])
            i1, i2 = labels.index(p1), labels.index(p2)
            sd = calculate_distance(pts[i1], pts[i2]) * scale
            dist_tbl = pd.DataFrame({f"{p1} → {p2}":[f"{sd:.4f} {unit}"]})
            st.table(dist_tbl)

        # save for next redraw
        st.session_state.bg_img = img
    else:
        st.info("Select at least 2 points for reference length.")
else:
    st.info("Click on the image to start marking points.")