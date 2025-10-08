import streamlit as st
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh (AI-Powered) 📊",
    layout="wide"
)

st.title("Ứng dụng Đánh Giá Hiệu quả Dự án Kinh doanh 🚀")
st.markdown("---")

# --- Khai báo API Key và Client (Sử dụng st.secrets) ---
try:
    API_KEY = st.secrets.get("GEMINI_API_KEY")
    if not API_KEY:
        st.warning("Vui lòng cấu hình Khóa API (GEMINI_API_KEY) trong Streamlit Secrets để ứng dụng hoạt động.")
except:
    API_KEY = None
    st.warning("Không tìm thấy cấu hình Streamlit Secrets. Các chức năng AI sẽ không hoạt động.")


# --- Định nghĩa Schema JSON cho việc trích xuất dữ liệu ---
EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "investment": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (Initial Investment), giá trị tuyệt đối."},
        "life": {"type": "NUMBER", "description": "Vòng đời dự án (Project Life) bằng năm."},
        "revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (Annual Revenue)."},
        "cost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (Annual Cost)."},
        "wacc": {"type": "NUMBER", "description": "Suất chiết khấu WACC (Weighted Average Cost of Capital) dưới dạng thập phân (ví dụ: 0.13 cho 13%)."},
        "tax_rate": {"type": "NUMBER", "description": "Thuế suất thuế Thu nhập Doanh nghiệp (Tax Rate) dưới dạng thập phân (ví dụ: 0.20 cho 20%)."}
    },
    "required": ["investment", "life", "revenue", "cost", "wacc", "tax_rate"]
}

# --- Hàm AI: Trích xuất Dữ liệu (Task 1) ---
@st.cache_data(show_spinner=False)
def extract_project_data(text_input, api_key):
    """Sử dụng Gemini để trích xuất các thông số tài chính vào cấu trúc JSON."""
    if not api_key:
        st.error("Lỗi: Khóa API chưa được cấu hình.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        
        system_prompt = (
            "Bạn là một chuyên gia trích xuất dữ liệu tài chính. Nhiệm vụ của bạn là đọc "
            "nội dung phương án kinh doanh và trích xuất sáu thông số tài chính chính xác "
            "vào một đối tượng JSON tuân thủ schema được cung cấp. "
            "Dòng tiền tạo ra bắt đầu từ cuối năm thứ nhất."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Trích xuất các thông số tài chính từ phương án kinh doanh sau:\n\n---\n{text_input}",
            config={
                "systemInstruction": system_prompt,
                "responseMimeType": "application/json",
                "responseSchema": EXTRACTION_SCHEMA
            }
        )
        
        # Xử lý chuỗi JSON nhận được
        json_string = response.text.strip()
        return json.loads(json_string)

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON: AI không trả về định dạng JSON hợp lệ. Vui lòng thử lại hoặc chỉnh sửa nội dung đầu vào.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- Hàm Tính toán: Xây dựng Dòng tiền và Chỉ số (Task 2 & 3) ---
def calculate_project_metrics(params):
    """Xây dựng bảng dòng tiền và tính toán NPV, IRR, PP, DPP."""
    V0 = params['investment']
    T = int(params['life'])
    R = params['revenue']
    C = params['cost']
    WACC = params['wacc']
    Tax = params['tax_rate']

    # 1. Tính toán Dòng tiền Thuần (CF) hàng năm
    EBIT = R - C
    LNT = EBIT * (1 - Tax)
    # Giả định: Khấu hao = 0 hoặc Dòng tiền thuần hàng năm (CF) chính là LNT
    # Ta sử dụng LNTST (Lợi nhuận sau thuế) là CF hoạt động hàng năm
    CF_annual = LNT 

    # 2. Xây dựng Bảng Dòng tiền
    years = list(range(0, T + 1))
    
    # Dòng tiền ban đầu (Năm 0)
    cash_flows = [-V0]
    
    # Dòng tiền hoạt động (Năm 1 đến T)
    cash_flows.extend([CF_annual] * T)
    
    # Bảng Dòng tiền (DataFrame)
    df_cashflow = pd.DataFrame({
        'Năm': years,
        'Dòng tiền thuần (CF)': cash_flows
    })
    
    # Tính toán Chiết khấu
    df_cashflow['Hệ số chiết khấu'] = [1 / ((1 + WACC)**t) for t in years]
    df_cashflow['Dòng tiền chiết khấu (DCF)'] = df_cashflow['Dòng tiền thuần (CF)'] * df_cashflow['Hệ số chiết khấu']

    # 3. Tính toán Chỉ số Đánh giá
    
    # NPV
    # np.npv(rate, values)
    NPV = np.npv(WACC, cash_flows)
    
    # IRR
    try:
        IRR = np.irr(cash_flows)
    except ValueError:
        IRR = np.nan # Không thể tính nếu dòng tiền không đổi dấu

    # PP (Payback Period)
    cumulative_cf = np.cumsum(cash_flows)
    pp_index = np.where(cumulative_cf >= 0)[0]
    PP = pp_index[0] + (cumulative_cf[pp_index[0] - 1] / CF_annual) if len(pp_index) > 0 and pp_index[0] > 0 else T 
    PP = T if PP > T else PP # Giới hạn thời gian hoàn vốn

    # DPP (Discounted Payback Period)
    cumulative_dcf = np.cumsum(df_cashflow['Dòng tiền chiết khấu (DCF)'])
    dpp_index = np.where(cumulative_dcf >= 0)[0]
    DPP = dpp_index[0] + (cumulative_dcf[dpp_index[0] - 1] / df_cashflow['Dòng tiền chiết khấu (DCF)'].iloc[dpp_index[0]]) if len(dpp_index) > 0 and dpp_index[0] > 0 else T
    DPP = T if DPP > T else DPP

    results = {
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP
    }
    
    return df_cashflow, results

# --- Hàm AI: Phân tích Chỉ số (Task 4) ---
def get_ai_analysis(params, results, df_cashflow, api_key):
    """Gửi các chỉ số và tham số dự án đến Gemini API để nhận nhận xét."""
    if not api_key:
        return "Lỗi: Khóa API chưa được cấu hình để thực hiện phân tích."
        
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Bạn là một chuyên gia thẩm định dự án tài chính. Dựa trên các thông số dự án và kết quả đánh giá hiệu quả sau, hãy đưa ra một nhận xét chuyên sâu, khách quan, và rõ ràng (khoảng 4-5 đoạn văn) về:
        1. Tính khả thi của dự án (dựa trên NPV và IRR).
        2. Tốc độ thu hồi vốn (dựa trên PP và DPP).
        3. Khuyến nghị hành động tiếp theo cho chủ đầu tư.
        
        ---
        THÔNG SỐ DỰ ÁN:
        - Vốn đầu tư ban đầu: {params['investment']:,.0f} VNĐ
        - Dòng đời dự án: {params['life']} năm
        - Dòng tiền thuần hàng năm: {params['revenue'] - params['cost'] * (1 - params['tax_rate']):,.0f} VNĐ
        - Suất chiết khấu (WACC): {params['wacc'] * 100:.2f}%
        
        KẾT QUẢ ĐÁNH GIÁ:
        - NPV (Giá trị hiện tại ròng): {results['NPV']:,.0f} VNĐ
        - IRR (Tỷ suất hoàn vốn nội bộ): {results['IRR'] * 100:.2f}%
        - PP (Thời gian hoàn vốn): {results['PP']:.2f} năm
        - DPP (Thời gian hoàn vốn chiết khấu): {results['DPP']:.2f} năm
        ---
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Khởi tạo Session State ---
if 'extracted_params' not in st.session_state:
    st.session_state['extracted_params'] = None

# --- Giao diện người dùng ---

# 1. Input và Lọc Dữ liệu (Task 1)
st.subheader("1. Nhập liệu Phương án Kinh doanh và Trích xuất Dữ liệu (AI)")
business_plan_content = st.text_area(
    "Dán nội dung phương án kinh doanh (từ file Word) vào đây:",
    height=300,
    value="Tổng vốn đầu tư là 30 tỷ. Dự án kéo dài 10 năm. Mỗi năm dự kiến doanh thu 3.5 tỷ và chi phí là 2 tỷ. Thuế suất 20%. WACC 13%."
)

if st.button("Lọc Dữ liệu Tài chính (AI)") and API_KEY:
    if business_plan_content:
        with st.spinner('Đang gửi dữ liệu tới AI để trích xuất thông số...'):
            params = extract_project_data(business_plan_content, API_KEY)
            st.session_state['extracted_params'] = params
            if params:
                st.success("Trích xuất dữ liệu thành công! Vui lòng kiểm tra các thông số bên dưới.")
                
                # Hiển thị tóm tắt thông số
                st.markdown("### Thông số Dự án đã Trích xuất")
                col1, col2, col3 = st.columns(3)
                col1.metric("Vốn đầu tư ban đầu (V0)", f"{params['investment']:,.0f} VNĐ")
                col2.metric("Vòng đời dự án (T)", f"{params['life']} năm")
                col3.metric("WACC (k)", f"{params['wacc'] * 100:.2f}%")
                col1.metric("Doanh thu/năm (R)", f"{params['revenue']:,.0f} VNĐ")
                col2.metric("Chi phí/năm (C)", f"{params['cost']:,.0f} VNĐ")
                col3.metric("Thuế suất (t)", f"{params['tax_rate'] * 100:.2f}%")
            else:
                st.session_state['extracted_params'] = None
    else:
        st.warning("Vui lòng nhập nội dung phương án kinh doanh trước khi lọc.")

st.markdown("---")

# 2, 3. Xây dựng Bảng Dòng tiền và Tính toán Chỉ số (Tasks 2 & 3)
if st.session_state['extracted_params']:
    params = st.session_state['extracted_params']
    
    # Tính toán
    df_cashflow, results = calculate_project_metrics(params)

    st.subheader("2. Bảng Dòng tiền Thuần và Chiết khấu")
    st.dataframe(
        df_cashflow.style.format({
            'Dòng tiền thuần (CF)': '{:,.0f}',
            'Hệ số chiết khấu': '{:.4f}',
            'Dòng tiền chiết khấu (DCF)': '{:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("3. Các Chỉ số Đánh giá Hiệu quả Dự án")
    
    col_npv, col_irr, col_pp, col_dpp = st.columns(4)

    col_npv.metric(
        label="Giá trị Hiện tại Ròng (NPV)", 
        value=f"{results['NPV']:,.0f} VNĐ", 
        delta="Dự án Khả thi" if results['NPV'] > 0 else "Dự án Không khả thi"
    )
    
    col_irr.metric(
        label="Tỷ suất Hoàn vốn Nội bộ (IRR)", 
        value=f"{results['IRR'] * 100:.2f}%" if not np.isnan(results['IRR']) else "Không tính được",
        delta="Tốt (> WACC)" if results['IRR'] > params['wacc'] else "Kém (< WACC)"
    )
    
    col_pp.metric(
        label="Thời gian Hoàn vốn (PP)", 
        value=f"{results['PP']:.2f} năm", 
        delta_color="off"
    )
    
    col_dpp.metric(
        label="Thời gian Hoàn vốn Chiết khấu (DPP)", 
        value=f"{results['DPP']:.2f} năm",
        delta_color="off"
    )
    
    st.markdown("---")

    # 4. Yêu cầu AI Phân tích (Task 4)
    st.subheader("4. Phân tích Chuyên sâu Chỉ số Dự án (AI)")
    
    if st.button("Yêu cầu AI Phân tích Kết quả"):
        if API_KEY:
            with st.spinner('Đang gửi kết quả và chờ Gemini phân tích...'):
                ai_result = get_ai_analysis(params, results, df_cashflow, API_KEY)
                st.markdown("### Kết quả Phân tích từ Gemini AI")
                st.info(ai_result)
        else:
            st.error("Lỗi: Vui lòng cấu hình Khóa API trước khi yêu cầu phân tích.")
