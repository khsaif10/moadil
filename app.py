import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. إعدادات الصفحة العامة
st.set_page_config(
    page_title="PAU - Cell Segmentation & Counting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. قاموس اللغات الاحترافي المطور
LANGUAGES = {
    "English": {
        "title": "Medical Images Cell Nuclei Segmentation & Counting",
        "subtitle": "Advanced Computer Vision System for Biological Analysis",
        "supervisor": "Supervisor",
        "student": "Student",
        "sidebar_header": "⚙️ Control Panel",
        "lang_label": "🌐 Select Language",
        "upload_label": "Drag and drop or browse a cell image (PNG, JPG, TIF)",
        "analyze_btn": "🔬 Analyze & Count Cells",
        "results_header": "📊 Analysis Analytics",
        "count_metric": "Total Nuclei Detected",
        "time_metric": "Inference Speed",
        "orig_view": "1. Original Image",
        "enhanced_view": "2. Enhanced Image (Contrast & Sharpen)",
        "pred_view": "3. YOLO11 Delineated Output",
        "no_img": "⚠️ Please upload an image first to run the analysis.",
        "success_msg": "🎉 Analysis completed successfully!"
    },
    "Türkçe": {
        "title": "Tıbbi Görüntülerde Hücre Nükleus Segmentasyonu ve Sayımı",
        "subtitle": "Biyolojik Analiz için Gelişmiş Bilgisayarlı Görü Sistemi",
        "supervisor": "Danışman",
        "student": "Öğrenci",
        "sidebar_header": "⚙️ Kontrol Paneli",
        "lang_label": "🌐 Dil Seçiniz / Select Language",
        "upload_label": "Bir hücre görseli sürükleyip bırakın veya seçin (PNG, JPG, TIF)",
        "analyze_btn": "🔬 Hücreleri Analiz Et ve Say",
        "results_header": "📊 Analiz Sonuçları",
        "count_metric": "Tespit Edilen Çekirdek Sayısı",
        "time_metric": "Çıkarım Hızı",
        "orig_view": "1. Orijinal Görsel",
        "enhanced_view": "2. Geliştirilmiş Görsel (Kontrast & Keskinleştirme)",
        "pred_view": "3. YOLO11 Segmentasyon Çıktısı",
        "no_img": "⚠️ Analizi çalıştırmak için lütfen önce bir görsel yükleyin.",
        "success_msg": "🎉 Analiz başarıyla tamamlandı!"
    },
    "العربية": {
        "title": "تقسيم وعدّ نوى الخلايا في الصور الطبية",
        "subtitle": "نظام رؤية حاسوبية متقدم للتحليل البيولوجي المجهري",
        "supervisor": "الأستاذ المشرف",
        "student": "الطالب الباحث",
        "sidebar_header": "⚙️ لوحة التحكم الإدارية",
        "lang_label": "🌐 اختر لغة الواجهة",
        "upload_label": "قم بسحب وإسقاط أو اختيار صورة الخلايا المجهرية (PNG, JPG, TIF)",
        "analyze_btn": "🔬 بدء تحليل وعد الخلايا تلقائياً",
        "results_header": "📊 لوحة البيانات الإحصائية",
        "count_metric": "إجمالي النوى المكتشفة",
        "time_metric": "سرعة المعالجة الفورية",
        "orig_view": "1. الصورة المجهرية الأصلية",
        "enhanced_view": "2. الصورة بعد التحسين البصري الحاد",
        "pred_view": "3. مخرجات تقسيم نموذج YOLO11 الفخمة",
        "no_img": "⚠️ الرجاء رفع صورة مجهرية أولاً للبدء في معالجتها.",
        "success_msg": "🎉 تم التحليل بنجاح واكتشاف كافة الخلايا المتاحة!"
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "English"

selected_lang = st.sidebar.selectbox(
    LANGUAGES[st.session_state.lang]["lang_label"],
    options=["English", "Türkçe", "العربية"],
    index=["English", "Türkçe", "العربية"].index(st.session_state.lang)
)
st.session_state.lang = selected_lang
txt = LANGUAGES[st.session_state.lang]

st.sidebar.write("---")
st.sidebar.markdown(f"### **{txt['sidebar_header']}**")
st.sidebar.info(f"""
🏛️ **Pamukkale Üniversitesi**
🔬 **PAU Medical AI Lab**

👨‍🏫 **{txt['supervisor']}:** Doç. Dr. Muhammet Üsame ÖZİÇ

🎓 **{txt['student']}:** Mohammed Adil
""")

@st.cache_resource
def load_model():
    model_path = "best.pt"
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model 'best.pt': {e}")
        return None

model = load_model()

# ==========================================
# 🛠️ دالة تحسين ومعالجة بصرية خارقة (Sharpening & Contrast)
# ==========================================
def enhance_medical_image(img_bgr):
    """إبراز نوى الخلايا المجهرية الطبية بتباين حاد وحدود دقيقة جداً للعرض"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين التكيفي بشكل متزن لعدم حرق الألوان
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)
    
    # مصفوفة فلتر كشف الحواف والkeskinleştirme (Sharpening Kernel) لإضاءة الحدود
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(cl_img, -1, kernel)
    
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

# 5. تصميم الجسم الرئيسي للواجهة
st.markdown("<h2 style='text-align: center; color: #1f538d;'>PAMUKKALE ÜNİVERSİTESİ</h2>", unsafe_allow_html=True)
st.title(txt["title"])
st.caption(txt["subtitle"])
st.write("---")

uploaded_file = st.file_uploader(txt["upload_label"], type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    orig_image = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    enhanced_image = enhance_medical_image(opencv_img)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(txt["orig_view"])
        st.image(orig_image, use_container_width=True)
        
    with col2:
        st.subheader(txt["enhanced_view"])
        st.image(enhanced_image, use_container_width=True)
        
    run_analysis = st.button(txt["analyze_btn"], type="primary", use_container_width=True)
    
    if run_analysis:
        if model is not None:
            with st.spinner("Analyzing image... Please wait..."):
                # 🔥 رفع مستوى الـ conf إلى 0.40 لفلترة الشوائب الخفيفة التي يخطئ فيها النموذج
                results = model(opencv_img, conf=0.40)[0]
                speed_ms = results.speed.get('inference', 0.0)
                
                # خوارزمية ذكية لاستبعاد الخلايا الناقصة والمقطوعة عند الحواف الخارجية للصورة
                cell_count = 0
                if results.masks is not None:
                    h, w = opencv_img.shape[:2]
                    
                    for mask in results.masks.xy:
                        x_coords = mask[:, 0]
                        y_coords = mask[:, 1]
                        
                        # تصفية الخلايا الملتصقة بإطار وحواف الصورة الخارجية (بفارق 2 بكسل) لمنع العد الزائد
                        if (np.min(x_coords) <= 2 or np.max(x_coords) >= w - 2 or
                            np.min(y_coords) <= 2 or np.max(y_coords) >= h - 2):
                            continue
                        
                        cell_count += 1
                
                # رسم الأقنعة الأصلية الدقيقة دون صناديق أو تسميات مشوشة
                annotated_frame = results.plot(labels=False, boxes=False)
                pred_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
            with col3:
                st.subheader(txt["pred_view"])
                st.image(pred_image, use_container_width=True)
                
            st.success(txt["success_msg"])
            st.write(f"### {txt['results_header']}")
            
            m1, m2 = st.columns(2)
            m1.metric(label=txt["count_metric"], value=f"🎯 {cell_count}")
            m2.metric(label=txt["time_metric"], value=f"⚡ {speed_ms:.1f} ms")
        else:
            st.error("Model 'best.pt' could not be loaded.")
