import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. إعدادات الصفحة العامة (يجب أن تكون أول أمر في Streamlit)
st.set_page_config(
    page_title="PAU - Cell Segmentation & Counting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. قاموس اللغات الاحترافي لتغيير الواجهة فوراً
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
        "orig_view": "Original Microscopic Image",
        "pred_view": "YOLO11 Segmentation Output",
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
        "orig_view": "Orijinal Mikroskobik Görsel",
        "pred_view": "YOLO11 Segmentasyon Çıktısı",
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
        "orig_view": "الصورة المجهرية الأصلية",
        "pred_view": "مخرجات تقسيم نموذج YOLO11",
        "no_img": "⚠️ الرجاء رفع صورة مجهرية أولاً للبدء في معالجتها.",
        "success_msg": "🎉 تم التحليل بنجاح واكتشاف كافة الخلايا المتاحة!"
    }
}

# 3. إعداد شريط اللغة الجانبي وتحديد النصوص المترجمة
if "lang" not in st.session_state:
    st.session_state.lang = "English"

# وضع اختيار اللغة في أعلى القائمة الجانبية
selected_lang = st.sidebar.selectbox(
    LANGUAGES[st.session_state.lang]["lang_label"],
    options=["English", "Türkçe", "العربية"],
    index=["English", "Türkçe", "العربية"].index(st.session_state.lang)
)
st.session_state.lang = selected_lang
txt = LANGUAGES[st.session_state.lang]

# 4. تصميم الشريط الجانبي الأكاديمي (Sidebar)
st.sidebar.write("---")
st.sidebar.markdown(f"### **{txt['sidebar_header']}**")

# معلومات الجامعة والدكتور والطالب بشكل فخم وعلامات ماركداون أنيقة
st.sidebar.info(f"""
🏛️ **Pamukkale Üniversitesi**
🔬 **PAU Medical AI Lab**

👨‍🏫 **{txt['supervisor']}:** Doç. Dr. Muhammet Üsame ÖZİÇ

🎓 **{txt['student']}:** Mohammed Adil
""")

# تحميل نموذج YOLO11 المدرب بأمان وحفظه في الـ Cache لضمان السرعة الخارقة
@st.cache_resource
def load_model():
    model_path = "best.pt"
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model 'best.pt': {e}")
        return None

model = load_model()

# 5. تصميم الجسم الرئيسي للواجهة (Main Content)
# محاكاة لشعار الجامعة في العناوين
st.markdown("<h2 style='text-align: center; color: #1f538d;'>PAMUKKALE ÜNİVERSİTESİ</h2>", unsafe_allow_html=True)
st.title(txt["title"])
st.caption(txt["subtitle"])
st.write("---")

# صندوق رفع الملفات
uploaded_file = st.file_uploader(txt["upload_label"], type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # قراءة الصورة المرفوعة وتحويلها إلى مصفوفة قابلة للمعالجة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    orig_image = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    # تقسيم الشاشة بالتساوي لعرض مقارنة فورية بين الأصل والتوقع
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(txt["orig_view"])
        st.image(orig_image, use_container_width=True)
        
    # زر تشغيل الذكاء الاصطناعي
    run_analysis = st.button(txt["analyze_btn"], type="primary", use_container_width=True)
    
    if run_analysis:
        if model is not None:
            with st.spinner("Analyzing image... Please wait..."):
                # تشغيل التوقع الفوري
                results = model(opencv_img, conf=0.25)[0]
                
                # حساب الإحصائيات
                cell_count = len(results.masks) if results.masks is not None else 0
                speed_ms = results.speed.get('inference', 0.0)
                
                # رسم مخرجات الـ Segmentation وعزل الصناديق لتوضيح الأقنعة فقط
                annotated_frame = results.plot(labels=False, boxes=False)
                pred_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
            with col2:
                st.subheader(txt["pred_view"])
                st.image(pred_image, use_container_width=True)
                
            # عرض بطاقات النتائج الرقمية والرسائل التحفيزية
            st.success(txt["success_msg"])
            st.write(f"### {txt['results_header']}")
            
            m1, m2 = st.columns(2)
            m1.metric(label=txt["count_metric"], value=f"🎯 {cell_count}")
            m2.metric(label=txt["time_metric"], value=f"⚡ {speed_ms:.1f} ms")
        else:
            st.error("Model 'best.pt' could not be loaded. Please ensure it is present in the repository.")