# main.py
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CityX Crime Analytics Platform",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HIDE DEFAULT STREAMLIT PAGES NAV (AUTO SIDEBAR)
# ============================================================
st.markdown("""
<style>
/* hide the built-in multipage nav */
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNavItems"] { display: none !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION (Custom VBar)
# ============================================================
with st.sidebar:
    st.title("🚔 CityX Platform")
    st.caption("Navigation")

    st.page_link("main.py", label="Home")
    st.page_link("pages/cityx_crime_dashboard.py", label="Crime Dashboard")
    st.page_link("pages/streamlit_app_level4.py", label="Advanced Prediction")

    #st.divider()
    #st.caption("System Info")
    #st.info("Use the sidebar to switch between dashboards.")

# ============================================================
# MAIN CONTENT
# ============================================================
def main():
    # Main header
    st.title("🚔 CityX Crime Analytics Platform")
    st.markdown("""
    Welcome to the comprehensive crime analysis and prediction platform for CityX.  
    This platform provides powerful tools for law enforcement agencies to analyze crime patterns, 
    predict crime categories, and enhance public safety operations.
    """)
    st.markdown("---")

    # Platform overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Crime Dashboard")
        st.markdown("""
        **Interactive crime data visualization and analysis:**
        - Real-time crime heatmaps and clustering  
        - Temporal and spatial analysis  
        - District-level performance metrics  
        - Multi-dimensional filtering capabilities  

        *Ideal for operational planning and resource allocation.*
        """)

        if st.button("🚀 Launch Crime Dashboard", use_container_width=True):
            st.switch_page("pages/cityx_crime_dashboard.py")

    with col2:
        st.subheader("🔮 Advanced Prediction")
        st.markdown("""
        **AI-powered crime classification and severity assessment:**
        - PDF report extraction and parsing  
        - Machine learning classification  
        - Crime severity scoring  
        - Batch processing capabilities  

        *Perfect for intelligence analysis and case prioritization.*
        """)

        if st.button("🚀 Launch Prediction System", use_container_width=True):
            st.switch_page("pages/streamlit_app_level4.py")

    st.markdown("---")

    # Features grid
    st.subheader("🎯 Platform Capabilities")

    features_col1, features_col2, features_col3 = st.columns(3)

    with features_col1:
        st.markdown("### 📍 Spatial Analysis")
        st.markdown("""
        - Crime density heatmaps  
        - Geographic clustering  
        - District boundary analysis  
        - Location-based insights
        """)

    with features_col2:
        st.markdown("### ⏰ Temporal Analysis")
        st.markdown("""
        - Time-series trends  
        - Hourly/day-of-week patterns  
        - Seasonal variations  
        - Predictive forecasting
        """)

    with features_col3:
        st.markdown("### 🤖 AI Intelligence")
        st.markdown("""
        - Natural language processing  
        - Document classification  
        - Severity assessment  
        - Confidence scoring
        """)

    st.markdown("---")

    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Getting Started

        1. **For Data Analysis**: Use the **Crime Dashboard** to:
           - Upload your crime dataset (CSV format)  
           - Explore interactive maps and charts  
           - Apply filters by time, location, and crime type  
           - Generate operational insights  

        2. **For Prediction**: Use the **Advanced Prediction** to:
           - Upload police report PDFs  
           - Extract key information automatically  
           - Classify crime types using AI  
           - Assess severity levels  
           - Export results for further analysis  

        ### Data Requirements

        **Crime Dashboard** accepts CSV files with columns like:
        - Latitude/Longitude coordinates  
        - Date and time information  
        - Crime categories/types  
        - District/precinct information  

        **Prediction System** processes PDF police reports containing:
        - Incident descriptions  
        - Location details  
        - Date/time information  
        - Officer narratives  
        """)

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()
