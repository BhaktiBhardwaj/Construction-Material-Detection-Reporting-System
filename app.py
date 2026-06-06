import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from datetime import datetime
import pandas as pd
import tempfile

# PAGE CONFIG
st.set_page_config(
    page_title="Construction Material Detection",
    page_icon="🏗️",
    layout="wide"
)


st.title("🏗️ Construction Material Detection & Reporting System")
st.markdown("AI-powered construction site monitoring using YOLOv8")

# Sidebar
with st.sidebar:

    st.markdown("# 🏗️ SiteVision AI")

    st.markdown("---")

    st.markdown("""
    ### 📋 Project Details

    **Model:** YOLOv8

    **Framework:** Streamlit

    **Classes:** 5 Construction Materials

    **Detection Type:** Object Detection

    **Reporting:** Automated Site Reports
    """)

    st.markdown("---")

    st.markdown("""
    ### 🔍 Detectable Materials

    ✅ Sand Piles

    ✅ Aggregate Piles

    ✅ Cement Bags

    ✅ Rebar Bundles

    ✅ Bitumen Drums
    """)

    st.markdown("---")

    st.info(
        "Upload one or multiple site images and generate a combined material inventory report."
    )


# LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")


model = load_model()

# PILE SIZE FUNCTION
def get_pile_size(area):


    if area < 15000:
        return "Small"
    elif area < 35000:
        return "Medium"
    else:
        return "Large"


# FILE UPLOAD
uploaded_files = st.file_uploader(
    "Upload Construction Site Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.write("Uploader loaded")

if uploaded_files:

    st.subheader("📷 Uploaded Images")

    cols = st.columns(3)

    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file)

        with cols[i % 3]:
            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

    if st.button("🔍 Analyze Site"):

        total_counts = Counter()
        pile_data = []
        annotated_images = []

        aggregate_num = 1
        sand_num = 1

        for uploaded_file in uploaded_files:

            image = Image.open(uploaded_file)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)

                results = model.predict(
                    source=tmp.name,
                    conf=0.4,
                    save=False
                )

                annotated_img = results[0].plot()

                annotated_images.append(
                    (
                        uploaded_file.name,
                        annotated_img
                    )
                )

            for r in results:

                # Material Counting
                for cls in r.boxes.cls:
                    total_counts[r.names[int(cls)]] += 1

                # Pile Size Estimation
                for box in r.boxes:

                    cls = int(box.cls[0])
                    class_name = r.names[cls]

                    if class_name not in [
                        "aggregate_pile",
                        "sand_pile"
                    ]:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0]

                    area = float(
                        (x2 - x1) * (y2 - y1)
                    )

                    size = get_pile_size(area)

                    if class_name == "aggregate_pile":

                        material_name = (
                            f"Aggregate Pile #{aggregate_num}"
                        )

                        aggregate_num += 1

                    else:

                        material_name = (
                            f"Sand Pile #{sand_num}"
                        )

                        sand_num += 1

                    pile_data.append({
                        "Material": material_name,
                        "Area": int(area),
                        "Estimated Size": size
                    })

        st.subheader("🎯 Detection Results")

        cols = st.columns(3)

        for i, (filename, img) in enumerate(annotated_images):

            with cols[i % 3]:

                st.image(
                    img,
                    caption=f"Detected: {filename}",
                    use_container_width=True
                )

        # SIDEBAR LIVE STATISTICS
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Detection Summary")

        st.sidebar.metric(
            "Total Materials",
            sum(total_counts.values())
        )

        st.sidebar.metric(
            "Sand Piles",
            total_counts.get("sand_pile", 0)
        )

        st.sidebar.metric(
            "Aggregate Piles",
            total_counts.get("aggregate_pile", 0)
        )

        st.sidebar.metric(
            "Cement Bags",
            total_counts.get("cement_bags", 0)
        )

        st.sidebar.metric(
            "Rebar Bundles",
            total_counts.get("rebar_bundle", 0)
        )

        st.sidebar.metric(
            "Bitumen Drums",
            total_counts.get("bitumen_drums", 0)
        )


        # METRICS
        st.subheader("📊 Site Summary")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric(
            "Total Materials",
            sum(total_counts.values())
        )

        col2.metric(
            "Sand Piles",
            total_counts.get("sand_pile", 0)
        )

        col3.metric(
            "Aggregate Piles",
            total_counts.get("aggregate_pile", 0)
        )

        col4.metric(
            "Cement Bags",
            total_counts.get("cement_bags", 0)
        )

        col5.metric(
            "Rebar Bundles",
            total_counts.get("rebar_bundle", 0)
        )

        col6.metric(
            "Bitumen Drums",
            total_counts.get("bitumen_drums", 0)
        )

       
        # INVENTORY TABLE
        st.subheader("📦 Material Inventory")

        materials = [
            "aggregate_pile",
            "sand_pile",
            "cement_bags",
            "rebar_bundle",
            "bitumen_drums"
        ]

        inventory_data = []

        for material in materials:

            inventory_data.append({
                "Material": material.replace("_", " ").title(),
                "Count": total_counts.get(material, 0)
            })

        inventory_df = pd.DataFrame(
            inventory_data
        )

        st.dataframe(
            inventory_df,
            use_container_width=True
        )

        # PILE SIZE TABLE
        st.subheader("📏 Pile Size Estimation")

        if pile_data:

            pile_df = pd.DataFrame(
                pile_data
            )

            st.dataframe(
                pile_df,
                use_container_width=True
            )
        
        # REPORT GENERATION
        report = []

        report.append("=" * 60)
        report.append("CONSTRUCTION MATERIAL REPORT")
        report.append(
            f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )
        report.append("=" * 60)
        report.append("")

        report.append("MATERIAL COUNTS")
        report.append("-" * 30)

        for material in materials:

            report.append(
                f"{material.replace('_',' ').title()}: "
                f"{total_counts.get(material,0)}"
            )

        report.append("")

        report.append("PILE SIZE ESTIMATION")
        report.append("-" * 30)

        for item in pile_data:

            report.append(
                f"{item['Material']} | "
                f"{item['Estimated Size']} | "
                f"Area: {item['Area']}"
            )

        report.append("")
        report.append("=" * 60)

        final_report = "\n".join(report)

        st.subheader("📄 Material Report")

        st.text_area(
            "Generated Report",
            final_report,
            height=350
        )

        
        st.download_button(
        label="⬇ Download Report",
        data=final_report,
        file_name="material_report.txt",
        mime="text/plain",
        key="download_report"
    )
