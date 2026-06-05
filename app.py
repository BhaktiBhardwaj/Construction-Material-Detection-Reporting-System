import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from datetime import datetime
import pandas as pd
import tempfile


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Construction Material Detection",
    page_icon="🏗️",
    layout="wide"
)
st.write("Step 1")

st.title("🏗️ Construction Material Detection & Reporting System")
st.markdown("AI-powered construction site monitoring using YOLOv8")

st.write("Step 2")
# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")


model = load_model()
st.write("Step 3")

# -------------------------------
# PILE SIZE FUNCTION
# -------------------------------
def get_pile_size(area):


    if area < 15000:
        return "Small"
    elif area < 35000:
        return "Medium"
    else:
        return "Large"


# -------------------------------
# FILE UPLOAD
# -------------------------------
st.write("Step 4")

uploaded_file = st.file_uploader(
    "Upload Construction Site Image",
    type=["jpg", "jpeg", "png"]
)

st.write("Uploader loaded")

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Materials", key="detect_btn"):

        model = load_model()

        st.success("Model Loaded")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        results = model.predict(
            source=image_path,
            conf=0.4,
            save=False
        )

        st.success("Prediction Complete")

        detected_img = results[0].plot()

        st.image(
            detected_img,
            caption="Detection Results",
            use_container_width=True
        )

        from collections import Counter

        counts = Counter()

        for r in results:
            for cls in r.boxes.cls:
                counts[r.names[int(cls)]] += 1

        st.subheader("📦 Material Inventory")

        inventory_data = []

        materials = [
            "aggregate_pile",
            "sand_pile",
            "cement_bags",
            "rebar_bundle",
            "bitumen_drums"
        ]

        for material in materials:
            inventory_data.append({
                "Material": material.replace("_", " ").title(),
                "Count": counts.get(material, 0)
            })

        inventory_df = pd.DataFrame(inventory_data)

        st.dataframe(
            inventory_df,
            use_container_width=True
        )

        st.subheader("📏 Pile Size Estimation")

        pile_data = []

        aggregate_num = 1
        sand_num = 1

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])
                class_name = r.names[cls]

                if class_name not in [
                    "aggregate_pile",
                    "sand_pile"
                ]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                area = float((x2 - x1) * (y2 - y1))

                size = get_pile_size(area)

                if class_name == "aggregate_pile":
                    material_name = f"Aggregate Pile #{aggregate_num}"
                    aggregate_num += 1
                else:
                    material_name = f"Sand Pile #{sand_num}"
                    sand_num += 1

                pile_data.append({
                    "Material": material_name,
                    "Area": int(area),
                    "Estimated Size": size
                })

        if pile_data:
            pile_df = pd.DataFrame(pile_data)
            st.dataframe(
                pile_df,
                use_container_width=True
            )
        
        st.subheader("📄 Material Report")

        report = []

        report.append("=" * 60)
        report.append("CONSTRUCTION MATERIAL REPORT")
        report.append(
            f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )
        report.append("=" * 60)
        report.append("")

        # Inventory Summary
        report.append("MATERIAL COUNTS")
        report.append("-" * 30)

        for material in materials:
            report.append(
                f"{material.replace('_',' ').title()}: {counts.get(material,0)}"
            )

        report.append("")

        # Pile Summary
        report.append("PILE SIZE ESTIMATION")
        report.append("-" * 30)

        for item in pile_data:
            report.append(
                f"{item['Material']} -> {item['Estimated Size']} "
                f"(Area: {item['Area']})"
            )

        report.append("")
        report.append("=" * 60)

        final_report = "\n".join(report)

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