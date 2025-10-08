import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf # FIX: Sử dụng numpy_financial để tính NPV và IRR
import json
import time

# Thư viện Gemini
from google.genai import Client
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh 📊",
    layout="wide"
)

# --- Hàm gọi API Gemini để trích xuất dữ liệu có cấu trúc ---
def extract_financial_parameters(text_data, api_key):
    """Sử dụng Gemini API để trích xuất các thông số tài chính từ văn bản."""
    
    # Chuẩn bị schema JSON cho đầu ra
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "Vốn đầu tư": {"type": "number", "description": "Tổng vốn đầu tư ban đầu (VNĐ)."},
            "Dòng đời dự án": {"type": "integer", "description": "Số năm hoạt động của dự án."},
            "Doanh thu": {"type": "number", "description": "Doanh thu thuần hàng năm (VNĐ)."},
            "Chi phí": {"type": "number", "description": "Tổng chi phí hoạt động hàng năm (VNĐ, chưa bao gồm Khấu hao nếu có)."},
            "WACC": {"type": "number", "description": "Chi phí sử dụng vốn bình quân (WACC), nhập dưới dạng thập phân (ví dụ: 0.13)."},
            "Thuế": {"type": "number", "description": "Thuế suất Thu nhập Doanh nghiệp (TNDN), nhập dưới dạng thập phân (ví dụ: 0.20)."}
        },
        "required": ["Vốn đầu tư", "Dòng đời dự án", "Doanh thu", "Chi phí", "WACC", "Thuế"]
    }

    system_prompt = (
        "Bạn là trợ lý phân tích tài chính. Nhiệm vụ của bạn là đọc phương án kinh doanh do người dùng cung cấp "
        "và trích xuất chính xác 6 thông số tài chính vào cấu trúc JSON cho trước. "
        "Hãy chuyển đổi tất cả các giá trị tiền tệ từ Tỷ, Triệu... sang đơn vị VNĐ và WACC/Thuế sang dạng thập phân."
    )
    
    user_query = f"Phương án kinh doanh:\n---\n{text_data}\n---\nHãy trích xuất 6 thông số tài chính chính xác nhất."

    try:
        # Sử dụng exponential backoff cho API call
        client = Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Thử lại tối đa 3 lần
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_query,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema
                    }
                )
                
                # Trích xuất và phân tích JSON
                json_string = response.text
                return json.loads(json_string)

            except APIError as e:
                if attempt < 2:
                    st.warning(f"Lỗi API (lần {attempt + 1}). Đang thử lại sau {2 ** attempt} giây...")
                    time.sleep(2 ** attempt)
                else:
                    raise e

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API sau 3 lần thử. Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("AI không thể tạo ra cấu trúc JSON hợp lệ. Vui lòng kiểm tra lại dữ liệu đầu vào hoặc prompt của AI.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None
    return None

# --- Hàm tính toán Dòng tiền và Chỉ số hiệu quả ---
@st.cache_data
def calculate_project_metrics(params):
    """
    Tính toán bảng dòng tiền và các chỉ số NPV, IRR, PP, DPP.
    Sử dụng numpy-financial (npf) cho các tính toán tài chính.
    """
    
    # 1. Trích xuất thông số
    C0 = -params['Vốn đầu tư']
    N = int(params['Dòng đời dự án'])
    WACC = params['WACC']
    
    # Tính toán Dòng tiền thuần hàng năm (CF_t)
    DT = params['Doanh thu']
    CP = params['Chi phí']
    Thue_suat = params['Thuế']
    
    EBIT = DT - CP
    Tax_amount = EBIT * Thue_suat
    CF_annually_after_tax = EBIT - Tax_amount
    
    # 2. Xây dựng Bảng Dòng tiền (Cash Flows)
    years = list(range(0, N + 1))
    cash_flows = [C0] + [CF_annually_after_tax] * N
    
    df_cashflow = pd.DataFrame({
        'Năm': years,
        'Dòng tiền thuần (CF)': cash_flows,
        'Hệ số chiết khấu': [1] + [1 / ((1 + WACC)**t) for t in range(1, N + 1)],
        'Dòng tiền chiết khấu (DCF)': [C0] + [CF_annually_after_tax / ((1 + WACC)**t) for t in range(1, N + 1)]
    })
    
    # 3. Tính toán các chỉ số
    cash_flows_for_npv_irr = [C0] + [CF_annually_after_tax] * N
    
    # a) NPV (Net Present Value)
    # npf.npv chỉ tính giá trị hiện tại của các dòng tiền tương lai. C0 được cộng vào sau.
    NPV = npf.npv(WACC, cash_flows_for_npv_irr[1:]) + cash_flows_for_npv_irr[0] 
    
    # b) IRR (Internal Rate of Return)
    try:
        IRR = npf.irr(cash_flows_for_npv_irr)
    except ValueError:
        IRR = np.nan # NaN nếu IRR không thể tính toán (thường xảy ra khi không có sự thay đổi dấu)

    # c) PP (Simple Payback Period)
    cumulative_cf = 0
    PP = float('inf')
    for year, cf in enumerate(cash_flows_for_npv_irr[1:], 1):
        cumulative_cf += cf
        if cumulative_cf >= abs(C0):
            PP = year - 1 + (abs(C0) - (cumulative_cf - cf)) / cf
            break
        if year == N:
            PP = float('inf')
    
    # d) DPP (Discounted Payback Period)
    cumulative_discounted_cf = 0
    DPP = float('inf')
    discounted_C0 = abs(C0) # Vốn đầu tư ban đầu không chiết khấu
    
    for year in range(1, N + 1):
        dcf = df_cashflow[df_cashflow['Năm'] == year]['Dòng tiền chiết khấu (DCF)'].iloc[0]
        cumulative_discounted_cf += dcf
        if cumulative_discounted_cf >= discounted_C0:
            DPP = year - 1 + (discounted_C0 - (cumulative_discounted_cf - dcf)) / dcf
            break
        if year == N:
            DPP = float('inf')

    # Trả về kết quả
    results = {
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP,
        'WACC': WACC
    }
    
    return df_cashflow, results

# --- Hàm AI Phân tích Chỉ số ---
def get_ai_analysis(results, df_cashflow, api_key):
    """Sử dụng Gemini để phân tích các chỉ số hiệu quả dự án."""
    
    system_prompt = (
        "Bạn là một chuyên gia thẩm định dự án và phân tích tài chính. "
        "Dựa trên các chỉ số hiệu quả dự án (NPV, IRR, PP, DPP), hãy đưa ra đánh giá chuyên sâu và khách quan "
        "về tính khả thi và rủi ro của dự án. So sánh IRR với WACC (13%). "
        "Sử dụng 3-4 đoạn văn. Nếu NPV âm, hãy giải thích tại sao dự án không khả thi."
    )
    
    # Chuẩn bị dữ liệu cho AI
    analysis_data = f"""
    Các chỉ số hiệu quả:
    - Vốn đầu tư ban đầu (C0): {df_cashflow['Dòng tiền thuần (CF)'].iloc[0]:,.0f} VNĐ
    - WACC (Suất chiết khấu): {results['WACC'] * 100:.2f}%
    - NPV (Giá trị hiện tại ròng): {results['NPV']:,.0f} VNĐ
    - IRR (Tỷ suất hoàn vốn nội bộ): {results['IRR'] * 100:.2f}%
    - PP (Thời gian hoàn vốn đơn giản): {results['PP']:.2f} năm
    - DPP (Thời gian hoàn vốn có chiết khấu): {results['DPP']:.2f} năm
    
    Bảng Dòng tiền (chi tiết):
    {df_cashflow.to_markdown(index=False)}
    """
    
    user_query = f"Phân tích dự án với các dữ liệu sau:\n{analysis_data}"

    try:
        client = Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        response = client.models.generate_content(
            model=model_name,
            contents=user_query,
            config={"system_instruction": system_prompt}
        )
        return response.text
        
    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}")
        return "Không thể thực hiện phân tích AI do lỗi kết nối hoặc cấu hình API."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình phân tích: {e}"


# --- Giao diện Streamlit ---
st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📈")
st.markdown("Sử dụng AI để trích xuất thông số và tính toán hiệu quả dự án.")

# Lấy API Key từ Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ **Lỗi Cấu hình:** Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

# Vùng nhập liệu phương án kinh doanh
st.subheader("1. Nhập Phương án Kinh doanh (Dán nội dung từ file Word)")
project_text = st.text_area(
    "Dán toàn bộ nội dung phương án kinh doanh vào đây. Đảm bảo có đủ 6 thông số: Vốn đầu tư, Dòng đời dự án, Doanh thu, Chi phí, WACC, Thuế.",
    height=400,
    value="Tổng vốn đầu tư là 30 tỷ. Dự án kéo dài 10 năm. Mỗi năm tạo ra 3.5 tỷ doanh thu và 2 tỷ chi phí. WACC của doanh nghiệp là 13% và thuế suất là 20%."
)

# State để lưu trữ tham số đã lọc
if 'params' not in st.session_state:
    st.session_state['params'] = None
    
# Nút Lọc Dữ liệu
if st.button("Lọc Dữ liệu Tài chính (AI) 🤖", disabled=not api_key or not project_text):
    if api_key and project_text:
        with st.spinner('AI đang đọc và trích xuất dữ liệu...'):
            params = extract_financial_parameters(project_text, api_key)
            if params:
                st.session_state['params'] = params
                st.success("Trích xuất dữ liệu thành công!")

# --- Hiển thị và Tính toán ---
if st.session_state['params']:
    params = st.session_state['params']
    
    st.markdown("---")
    st.subheader("2. Dữ liệu Tài chính Đã Lọc")
    
    # Hiển thị các tham số đã lọc dưới dạng Metrics
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    # Chuyển đổi các giá trị số để hiển thị đẹp
    display_wacc = f"{params['WACC'] * 100:.2f}%"
    display_thue = f"{params['Thuế'] * 100:.0f}%"
    
    col1.metric("Vốn đầu tư (C0)", f"{params['Vốn đầu tư']:,.0f} VNĐ")
    col2.metric("Dòng đời dự án (N)", f"{params['Dòng đời dự án']} năm")
    col3.metric("WACC (k)", display_wacc)
    col4.metric("Doanh thu/năm", f"{params['Doanh thu']:,.0f} VNĐ")
    col5.metric("Chi phí/năm", f"{params['Chi phí']:,.0f} VNĐ")
    col6.metric("Thuế suất", display_thue)
    
    
    # 3. Xây dựng Bảng Dòng tiền & Tính toán Chỉ số
    df_cashflow, results = calculate_project_metrics(params)
    
    st.markdown("---")
    st.subheader("3. Bảng Dòng tiền Dự án")
    st.dataframe(df_cashflow.style.format({
        'Dòng tiền thuần (CF)': '{:,.0f}',
        'Hệ số chiết khấu': '{:.4f}',
        'Dòng tiền chiết khấu (DCF)': '{:,.0f}'
    }), use_container_width=True)
    
    st.markdown("---")
    st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
    
    col_npv, col_irr, col_pp, col_dpp = st.columns(4)
    
    # Hiển thị NPV
    npv_value = f"{results['NPV']:,.0f} VNĐ"
    npv_delta = "Dự án khả thi" if results['NPV'] > 0 else "Dự án không khả thi"
    col_npv.metric("NPV (Giá trị hiện tại ròng)", npv_value, delta=npv_delta)

    # Hiển thị IRR
    irr_value = f"{results['IRR'] * 100:.2f}%" if not np.isnan(results['IRR']) else "N/A"
    irr_delta = f"WACC: {results['WACC'] * 100:.2f}%"
    col_irr.metric("IRR (Tỷ suất hoàn vốn nội bộ)", irr_value, delta=irr_delta)
    
    # Hiển thị PP
    pp_value = f"{results['PP']:.2f} năm"
    col_pp.metric("PP (Thời gian hoàn vốn)", pp_value)
    
    # Hiển thị DPP
    dpp_value = f"{results['DPP']:.2f} năm"
    col_dpp.metric("DPP (Thời gian hoàn vốn có chiết khấu)", dpp_value)
    
    # 5. Phân tích AI
    st.markdown("---")
    st.subheader("5. Phân tích Chỉ số Hiệu quả (AI)")

    if st.button("Yêu cầu AI Phân tích Chỉ số 🧠", disabled=not api_key):
        with st.spinner('Đang gửi kết quả và chờ Gemini phân tích...'):
            ai_analysis = get_ai_analysis(results, df_cashflow, api_key)
            st.info(ai_analysis)
