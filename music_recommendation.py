import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration for Streamlit Page ---
st.set_page_config(
    page_title="Hệ thống Gợi ý Nhạc (ML-Powered)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define the features used for clustering/recommendation ---
FEATURES = ['energy', 'valence', 'tempo']
N_CLUSTERS = 5 # Số lượng cụm (tâm trạng) muốn tìm

# Map the cluster index to a descriptive mood name (based on data inspection)
# Bạn có thể đổi tên này sau khi chạy và kiểm tra các centroid thực tế.
MOOD_LABELS = {
    0: 'Vui vẻ & Sôi động (Happy & Energetic)',
    1: 'Thư giãn & Nhẹ nhàng (Relaxed & Mellow)',
    2: 'Nhịp độ Nhanh & Lạc quan (Uptempo & Positive)',
    3: 'Buồn bã & Trầm lắng (Sad & Acoustic)',
    4: 'Trung tính & Cân bằng (Neutral & Balanced)',
}

# --- Load dataset & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    """Loads, cleans, scales data, and performs K-Means clustering."""
    try:
        # Tải dữ liệu
        df = pd.read_csv("SpotifyFeatures.csv")
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'SpotifyFeatures.csv'. Vui lòng đảm bảo file đã được đặt cùng thư mục.")
        return None, None, None, None

    # Chọn và làm sạch dữ liệu
    df = df[['track_name', 'artists', 'track_genre'] + FEATURES].dropna(subset=FEATURES)
    df.drop_duplicates(subset=['track_name', 'artists'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 1. Chuẩn hóa dữ liệu (Scaling for ML)
    scaler = StandardScaler()
    # Chỉ fit/transform các cột features
    scaled_features = scaler.fit_transform(df[FEATURES])
    scaled_df = pd.DataFrame(scaled_features, columns=FEATURES)

    # 2. Áp dụng K-Means Clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    df['mood'] = df['cluster'].map(MOOD_LABELS)

    # 3. Kết hợp dữ liệu đã chuẩn hóa và chưa chuẩn hóa
    df_final = pd.concat([df.drop(columns=FEATURES), scaled_df], axis=1)

    return df_final, kmeans, scaler

df_processed, kmeans_model, scaler_model = load_and_preprocess_data()

if df_processed is None:
    st.stop()


# --- Recommendation Logic (Cosine Similarity) ---
def recommend_by_similarity(input_vector_scaled, n_recommendations=5):
    """
    Finds the N nearest songs to a given input vector using Cosine Similarity.
    input_vector_scaled: (1, n_features) numpy array of scaled features
    """
    # Lấy ma trận features đã được scale của toàn bộ bài hát
    scaled_matrix = df_processed[FEATURES].values

    # Tính Cosine Similarity giữa vector input và tất cả bài hát
    # Similarity càng gần 1, càng giống
    similarities = cosine_similarity(input_vector_scaled, scaled_matrix).flatten()
    
    # Lấy index của các bài hát có độ tương đồng cao nhất
    # Sử dụng np.argsort để sắp xếp giảm dần và lấy N index cuối
    top_indices = np.argsort(similarities)[::-1][:n_recommendations]
    
    # Lấy thông tin bài hát
    recommendations = df_processed.iloc[top_indices]
    
    return recommendations

# --- Visualization Function (Radar Chart Data Prep) ---
def get_radar_chart_data(recs):
    """Prepares data for a radar chart (or bar chart) showing feature means."""
    # Lấy các feature gốc (chưa scale) để dễ hiểu hơn
    # Cần re-load feature gốc từ df_processed (đã có ở bước load_and_preprocess_data)
    # Tuy nhiên, để đơn giản và nhất quán với mô hình, ta sẽ dùng scaled features:
    
    # Tính giá trị trung bình (Mean) của 5 bài hát gợi ý (trên scaled features)
    radar_df = recs[FEATURES].mean().reset_index()
    radar_df.columns = ['Feature', 'Mean_Value']
    
    # Tạo thêm 1 DF chi tiết (chỉ lấy 5 bài)
    detail_df = recs[['track_name'] + FEATURES].set_index('track_name')
    return radar_df, detail_df


# =========================================================================
# --- Streamlit Interface ---
# =========================================================================

st.title("🎶 Hệ thống Gợi ý Nhạc (ML-Powered)")
st.markdown("Hệ thống sử dụng **K-Means Clustering** và **Cosine Similarity** để tìm kiếm bài hát phù hợp với tâm trạng của bạn.")

# Sử dụng cột để chia giao diện
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Chọn Phương thức Gợi ý")
    mode = st.radio(
        "Bạn muốn chọn tâm trạng theo:",
        ('Theo Cụm (Cluster)', 'Theo Tùy chỉnh (Custom Input)'),
        index=0,
        key='mode_selector'
    )
    
    # Placeholder for recommendation results
    recommendations = pd.DataFrame()

    if mode == 'Theo Cụm (Cluster)':
        st.subheader("Chọn Tâm trạng Đã Định nghĩa (Cluster)")
        
        # Lấy danh sách tâm trạng
        mood_options = list(MOOD_LABELS.values())
        selected_mood_label = st.selectbox(
            "Chọn một cụm tâm trạng:",
            mood_options,
            key='mood_selector'
        )
        
        # Tìm chỉ số cluster
        selected_cluster = [k for k, v in MOOD_LABELS.items() if v == selected_mood_label][0]
        
        # Lấy Centroid (trung tâm) của cluster đó (dưới dạng scaled feature)
        input_vector_scaled = kmeans_model.cluster_centers_[selected_cluster].reshape(1, -1)
        
        if st.button("Gợi ý Nhạc theo Tâm trạng"):
            recommendations = recommend_by_similarity(input_vector_scaled)

    elif mode == 'Theo Tùy chỉnh (Custom Input)':
        st.subheader("Điều chỉnh Tâm trạng Tùy chỉnh")
        st.write("Sử dụng các thanh trượt để định nghĩa tâm trạng của bạn (Giá trị càng cao càng mạnh)")
        
        # Cho phép người dùng nhập giá trị Energy, Valence, Tempo
        # Giới hạn giá trị dựa trên range thực tế (0-1 cho E/V, 0-250 cho Tempo)
        user_energy = st.slider("Energy (Năng lượng)", 0.0, 1.0, 0.75, 0.01)
        user_valence = st.slider("Valence (Độ tích cực/Hạnh phúc)", 0.0, 1.0, 0.85, 0.01)
        user_tempo = st.slider("Tempo (Tốc độ)", 50.0, 200.0, 120.0, 1.0)
        
        # Tạo vector input từ user
        user_input = pd.DataFrame([[user_energy, user_valence, user_tempo]], columns=FEATURES)
        
        # Scale vector input của người dùng (Rất quan trọng!)
        input_vector_scaled = scaler_model.transform(user_input).reshape(1, -1)

        if st.button("Gợi ý Nhạc Tùy chỉnh"):
            recommendations = recommend_by_similarity(input_vector_scaled)


# --- Display Results ---
with col2:
    st.header("2. Kết quả Gợi ý")

    if not recommendations.empty:
        st.success(f"Đã tìm thấy {len(recommendations)} bài hát phù hợp nhất!")
        
        st.subheader("Danh sách Bài hát Gợi ý")
        # Chọn các cột cần hiển thị
        display_cols = ['track_name', 'artists', 'track_genre']
        
        # Re-scale lại các features về giá trị gốc để hiển thị cho người dùng dễ hiểu
        # Vì recommendations chỉ chứa scaled data cho FEATURES, ta cần un-scale chúng.
        # Tuy nhiên, để đơn giản, ta sẽ hiển thị các feature gốc từ tập dữ liệu ban đầu
        # Bằng cách lấy index từ df_processed (đã được fill index)
        
        # Lấy index của các bài hát được gợi ý
        original_indices = recommendations.index
        # Lấy dữ liệu gốc từ tập đã xử lý ban đầu (chứa cả tên bài hát và feature gốc)
        
        # Tạo DataFrame hiển thị
        final_display_df = df_processed.loc[original_indices, ['track_name', 'artists', 'track_genre']].copy()
        
        # Thêm cột tâm trạng (mood) để biết nó thuộc cluster nào
        final_display_df['Tâm trạng Phân loại'] = recommendations['mood']
        
        # Lấy lại các giá trị Energy/Valence/Tempo GỐC để hiển thị
        # Đây là bước cần thiết vì df_processed chỉ lưu scaled features
        # Để lấy feature gốc, ta cần quay lại tập dữ liệu gốc (không tiện) hoặc 
        # thêm 3 cột feature gốc vào df_processed ngay từ đầu.
        
        # Ta sẽ dùng phương pháp đơn giản hơn: In ra Markdown
        for i, row in recommendations.iterrows():
             # Un-scale các giá trị để hiển thị (tùy chọn)
             # Vì việc un-scale phức tạp, ta sẽ chỉ in tên và thể loại để giữ code đơn giản.
             st.markdown(
                 f"#### 🎵 **{row['track_name']}**"
                 f"\n* Nghệ sĩ: **{row['artists']}**"
                 f"\n* Thể loại: *{row['track_genre']}*"
             )
             st.markdown("---")

        
        st.subheader("Phân tích Đặc trưng (Scaled Features)")
        # Hiển thị biểu đồ so sánh các thuộc tính của 5 bài gợi ý
        
        # Biểu đồ thanh (Bar chart) cho các scaled features
        # Scaled features nằm trong khoảng ~[-2, 2].
        
        # Lấy trung bình các scaled feature của 5 bài hát
        mean_scaled_features = recommendations[FEATURES].mean().reset_index()
        mean_scaled_features.columns = ['Feature', 'Giá trị Trung bình']
        
        st.bar_chart(mean_scaled_features, x='Feature', y='Giá trị Trung bình')
        st.write("Biểu đồ thể hiện mức độ trung bình của các đặc trưng (Energy, Valence, Tempo) của 5 bài hát gợi ý (đã được chuẩn hóa/scaled).")

    else:
        st.info("Chọn phương thức gợi ý và nhấn nút để nhận đề xuất!")

# --- Footer/Data Info ---
st.sidebar.subheader("Thông tin Dataset")
st.sidebar.write(f"- Tổng số bài hát (sau làm sạch): **{len(df_processed)}**")
st.sidebar.write(f"- Số lượng cụm tâm trạng (K-Means): **{N_CLUSTERS}**")
st.sidebar.markdown(f"**Các thuộc tính được dùng:** {', '.join(FEATURES)}")

st.sidebar.markdown("---")
st.sidebar.markdown("##### Giải thích về các Thuộc tính")
st.sidebar.markdown("- **Energy:** Mức năng lượng. Càng cao càng mãnh liệt, nhanh và ồn ào. (0.0 - 1.0)")
st.sidebar.markdown("- **Valence:** Độ tích cực. Càng cao càng vui vẻ, tích cực, phấn khởi. (0.0 - 1.0)")
st.sidebar.markdown("- **Tempo:** Tốc độ/nhịp độ của bài hát (BPM - Beats Per Minute).")