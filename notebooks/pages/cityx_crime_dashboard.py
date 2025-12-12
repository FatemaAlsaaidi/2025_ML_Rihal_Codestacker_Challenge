import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# -------------------- Page config --------------------
st.set_page_config(
    page_title="CityX Crime Watch Dashboard", 
    page_icon="🚔", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Simplified Map Display --------------------
def show_map(m, width=1200, height=600):
    """Display Folium map reliably using static method"""
    try:
        folium_static(m, width=width, height=height)
    except Exception as e:
        st.error(f"Map display error: {str(e)}")
        st.info("Try checking your coordinate data in the debug section below")

# -------------------- Smart Data Processing --------------------
COMMON_DATE_COLS = ["Dates", "Date", "Datetime", "datetime", "timestamp", "Timestamp", "time", "Time"]
# Support common and varying coordinate column names
COMMON_LAT_COLS = ["Latitude", "LATITUDE", "lat", "Lat", "Y", "y", "latitude", "Latitude (Y)"]
COMMON_LON_COLS = ["Longitude", "LONGITUDE", "lon", "Lon", "lng", "Lng", "Long", "long", "X", "x", "longitude", "Longitude (X)"]
COMMON_CATEGORY_COLS = ["Category", "category", "Type", "type", "Offense", "offense", "Crime Type"]
COMMON_DISTRICT_COLS = ["PdDistrict", "District", "district", "Division", "division", "Precinct", "precinct"]

def _normalize_colname(s: str) -> str:
    """lower + trim inner spaces for robust matching"""
    return " ".join(str(s).strip().split()).lower()

def detect_column(df, possible_names):
    """
    Robust column detection:
    - case-insensitive
    - trims spaces
    - tries smart hints for lat/lon variants
    """
    normalized_map = { _normalize_colname(c): c for c in df.columns }
    candidates = set(_normalize_colname(n) for n in possible_names)

    # direct match
    for norm, original in normalized_map.items():
        if norm in candidates:
            return original

    # smart hints for latitude-like
    for norm, original in normalized_map.items():
        if any(k in norm for k in ["latitude (y)", "lat (y)", " lat ", "(y)", " y "]):
            return original
    # smart hints for longitude-like
    for norm, original in normalized_map.items():
        if any(k in norm for k in ["longitude (x)", "lon (x)", " long ", "(x)", " x "]):
            return original

    # fallback contains
    for norm, original in normalized_map.items():
        if "lat" in norm:
            return original
    for norm, original in normalized_map.items():
        if "lon" in norm or "long" in norm or "lng" in norm:
            return original

    return None

def smart_data_processing(df, skip_coord_filtering=True):
    """Process data with intelligent column detection and cleaning"""
    
    # Store original info for debugging
    original_info = {
        'columns': df.columns.tolist(),
        'original_shape': df.shape,
        'detected_columns': {}
    }
    
    # Detect and standardize columns
    df_clean = df.copy()
    
    # Date detection and processing
    date_col = detect_column(df, COMMON_DATE_COLS)
    if date_col:
        df_clean['Date'] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean['Hour'] = df_clean['Date'].dt.hour.fillna(12).astype(int)
        df_clean['Month'] = df_clean['Date'].dt.month.fillna(1).astype(int)
        df_clean['Year'] = df_clean['Date'].dt.year.fillna(datetime.datetime.now().year).astype(int)
        df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name().fillna('Unknown')
        original_info['detected_columns']['date'] = date_col
    else:
        # Fallback defaults if no usable date column is found
        df_clean['Hour'] = 12
        df_clean['Month'] = 1
        df_clean['Year'] = datetime.datetime.now().year
        df_clean['DayOfWeek'] = 'Unknown'
    
    # Coordinate detection and processing
    lat_col = detect_column(df, COMMON_LAT_COLS + ["Latitude (Y)", "Y"])
    lon_col = detect_column(df, COMMON_LON_COLS + ["Longitude (X)", "X"])
    
    if lat_col and lon_col:
        # 1) Text cleaning: strip spaces and replace Arabic comma with dot
        lat_raw = df_clean[lat_col].astype(str).str.strip().str.replace(",", ".", regex=False)
        lon_raw = df_clean[lon_col].astype(str).str.strip().str.replace(",", ".", regex=False)

        # 2) Convert to numeric
        lat_num = pd.to_numeric(lat_raw, errors='coerce')
        lon_num = pd.to_numeric(lon_raw, errors='coerce')

        # 3) Check if coordinates are swapped (swap heuristic)
        normal_valid = (lat_num.between(-90, 90) & lon_num.between(-180, 180)).sum()
        swapped_valid = (lon_num.between(-90, 90) & lat_num.between(-180, 180)).sum()

        if swapped_valid > normal_valid * 1.5:
            # Swap if improvement is clear
            df_clean['Latitude']  = lon_num
            df_clean['Longitude'] = lat_num
            original_info['detected_columns']['latitude']  = lat_col + " (swapped)"
            original_info['detected_columns']['longitude'] = lon_col + " (swapped)"
        else:
            df_clean['Latitude']  = lat_num
            df_clean['Longitude'] = lon_num
            original_info['detected_columns']['latitude']  = lat_col
            original_info['detected_columns']['longitude'] = lon_col
        
        if not skip_coord_filtering:
            # Keep only valid coordinate ranges
            valid_coords = (
                df_clean['Latitude'].between(-90, 90) & 
                df_clean['Longitude'].between(-180, 180)
            )
            df_clean = df_clean[valid_coords].copy()
    else:
        # Create standard empty columns to avoid downstream errors
        if 'Latitude' not in df_clean.columns:  df_clean['Latitude'] = np.nan
        if 'Longitude' not in df_clean.columns: df_clean['Longitude'] = np.nan
    
    # Category detection
    category_col = detect_column(df, COMMON_CATEGORY_COLS)
    if category_col:
        df_clean['Category'] = df_clean[category_col].astype(str)
        original_info['detected_columns']['category'] = category_col
    
    # District detection  
    district_col = detect_column(df, COMMON_DISTRICT_COLS)
    if district_col:
        df_clean['District'] = df_clean[district_col].astype(str)
        original_info['detected_columns']['district'] = district_col
    
    original_info['final_shape'] = df_clean.shape
    original_info['rows_with_coords'] = (
        df_clean[['Latitude', 'Longitude']].notna().all(axis=1).sum()
        if 'Latitude' in df_clean.columns and 'Longitude' in df_clean.columns
        else 0
    )
    
    return df_clean, original_info


# -------------------- Dashboard Setup --------------------
st.sidebar.title("🚔 CityX Crime Watch")
st.sidebar.markdown("---")

# Data Source Section
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio("Choose data source:", 
                              ["Use Competition Dataset", "Upload CSV"], 
                              index=0)  # Default choice

df = None
data_info = {}

if data_source == "Use Competition Dataset":
    try:
        raw_df = pd.read_csv("../data/raw/Competition_Dataset.csv")
        skip_filter = st.sidebar.checkbox("Skip coordinate filtering", value=True)
        df, data_info = smart_data_processing(raw_df, skip_coord_filtering=skip_filter)
        st.sidebar.success(f"✅ Loaded {len(df):,} records from Competition_Dataset.csv")
    except Exception as e:
        st.sidebar.error(f"Error loading Competition_Dataset.csv: {str(e)}")
        st.stop()

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            skip_filter = st.sidebar.checkbox("Skip coordinate filtering", value=True)
            df, data_info = smart_data_processing(raw_df, skip_coord_filtering=skip_filter)
            st.sidebar.success(f"✅ Loaded {len(df):,} records")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file or switch to another data source")
        st.stop()

if df is None or df.empty:
    st.error("No data available. Please check your data source.")
    st.stop()

# -------------------- Filters --------------------
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("---")

# Year Filter
available_years = sorted(df['Year'].unique())
if len(available_years) > 0:
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(min(available_years)),
        max_value=int(max(available_years)),
        value=(int(min(available_years)), int(max(available_years))) )
else:
    year_range = (2023, 2023)

# Hour Filter  
hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# Category Filter
if 'Category' in df.columns:
    categories = sorted(df['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Crime Categories",
        options=categories,
        default=categories[:min(5, len(categories))]
    )
else:
    selected_categories = []

# District Filter
if 'District' in df.columns:
    districts = sorted(df['District'].unique())
    selected_districts = st.sidebar.multiselect(
        "Police Districts", 
        options=districts,
        default=districts
    )
else:
    selected_districts = []

# Apply Filters
filter_mask = (
    df['Year'].between(year_range[0], year_range[1]) & 
    df['Hour'].between(hour_range[0], hour_range[1])
)

if selected_categories:
    filter_mask &= df['Category'].isin(selected_categories)
if selected_districts:
    filter_mask &= df['District'].isin(selected_districts)

filtered_df = df[filter_mask].copy()

# -------------------- Main Dashboard --------------------
st.title("🚔 CityX Crime Watch: Operation Safe Streets")
st.markdown("Real-time crime analysis and visualization dashboard")
st.markdown("---")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_incidents = len(filtered_df)
    st.metric("Total Incidents", f"{total_incidents:,}")

with col2:
    unique_categories = filtered_df['Category'].nunique() if 'Category' in filtered_df.columns else 0
    st.metric("Crime Categories", unique_categories)

with col3:
    if 'District' in filtered_df.columns and not filtered_df.empty:
        top_district = filtered_df['District'].mode()[0] if not filtered_df['District'].mode().empty else "N/A"
    else:
        top_district = "N/A"
    st.metric("Top District", top_district)

with col4:
    if 'Category' in filtered_df.columns and not filtered_df.empty:
        top_crime = filtered_df['Category'].mode()[0] if not filtered_df['Category'].mode().empty else "N/A"
    else:
        top_crime = "N/A"
    st.metric("Most Common Crime", top_crime)

# -------------------- Main Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Crime Heatmap", "📊 Crime Clusters", "📈 Analytics", "👮 District Analysis"])

with tab1:
    st.subheader("Crime Density Heatmap")
    
    # ✅ Always use the standardized columns
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        coords = filtered_df[['Latitude', 'Longitude']].copy()
        # Safety: numeric conversion here as well
        coords['Latitude'] = pd.to_numeric(coords['Latitude'], errors='coerce')
        coords['Longitude'] = pd.to_numeric(coords['Longitude'], errors='coerce')

        valid_coords = coords.dropna()
        valid_coords = valid_coords[
            (valid_coords['Latitude'].between(-90, 90)) & 
            (valid_coords['Longitude'].between(-180, 180))
        ]
        
        if not valid_coords.empty:
            # Heatmap controls
            col1, col2, col3 = st.columns(3)
            with col1:
                radius = st.slider("Heat Radius", 5, 25, 12)
            with col2:
                blur = st.slider("Blur Intensity", 5, 25, 15)
            with col3:
                max_zoom = st.slider("Max Zoom Level", 10, 18, 13)
            
            # Create map
            center_lat = valid_coords['Latitude'].mean()
            center_lon = valid_coords['Longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='CartoDB positron',
                control_scale=True
            )
            
            # Add heatmap
            heat_data = valid_coords[['Latitude', 'Longitude']].values.tolist()
            HeatMap(
                heat_data, 
                radius=radius, 
                blur=blur, 
                max_zoom=max_zoom,
                gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(m)
            
            # Fit bounds (✅ names are correct now)
            m.fit_bounds([
                [valid_coords['Latitude'].min(), valid_coords['Longitude'].min()],
                [valid_coords['Latitude'].max(), valid_coords['Longitude'].max()]
            ])
            
            show_map(m)
            
            # Stats
            st.info(f"📍 Displaying {len(valid_coords):,} incidents on map")
        else:
            st.warning("No valid coordinates available for mapping. Try enabling 'Skip coordinate filtering' in sidebar.")
    else:
        st.error("No coordinate data found. Please ensure your data contains latitude and longitude columns.")

with tab2:
    st.subheader("Crime Clusters by Category")
    
    if 'Category' in filtered_df.columns and 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        valid_data = filtered_df[['Latitude', 'Longitude', 'Category']].dropna()
        valid_data = valid_data[
            (valid_data['Latitude'].between(-90, 90)) & 
            (valid_data['Longitude'].between(-180, 180))
        ]
        
        if not valid_data.empty:
            # Get top categories for coloring
            top_categories = valid_data['Category'].value_counts().head(8).index.tolist()
            
            # Create map
            center_lat = valid_data['Latitude'].mean()
            center_lon = valid_data['Longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='CartoDB positron'
            )
            
            # Color palette
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'darkblue']
            
            # Add clustered markers for each category
            for i, category in enumerate(top_categories):
                color = colors[i % len(colors)]
                category_data = valid_data[valid_data['Category'] == category]
                
                # Create feature group
                feature_group = folium.FeatureGroup(name=f"{category} ({len(category_data)})")
                
                # Add marker cluster
                marker_cluster = MarkerCluster().add_to(feature_group)
                
                # Sample for performance
                sample_size = min(500, len(category_data))
                sample_data = category_data.sample(n=sample_size, random_state=42)
                
                for _, row in sample_data.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=4,
                        color=color,
                        fill=True,
                        fillOpacity=0.7,
                        popup=folium.Popup(f"<b>{category}</b>", max_width=200)
                    ).add_to(marker_cluster)
                
                m.add_child(feature_group)
            
            # Add layer control
            folium.LayerControl(collapsed=False).add_to(m)
            m.fit_bounds(valid_data[['Latitude', 'Longitude']].values.tolist())
            
            show_map(m)
            
            # Category summary
            st.subheader("Category Summary")
            category_summary = valid_data['Category'].value_counts()
            fig = px.bar(
                x=category_summary.values, 
                y=category_summary.index,
                orientation='h',
                title="Crimes by Category",
                labels={'x': 'Number of Incidents', 'y': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No valid data available for clustering.")
    else:
        st.error("Required columns (Category, Latitude, Longitude) not found.")

with tab3:
    st.subheader("Crime Analytics & Trends")
    
    if not filtered_df.empty:
        # Create subplots grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Crimes by Hour of Day', 'Crimes by Month', 'Top Crime Categories', 'Crimes by District'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Hourly distribution
        hourly_counts = filtered_df['Hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='By Hour'),
            row=1, col=1
        )
        
        # Monthly distribution
        if 'Month' in filtered_df.columns:
            monthly_counts = filtered_df['Month'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=monthly_counts.index, y=monthly_counts.values, name='By Month'),
                row=1, col=2
            )
        
        # Top categories
        if 'Category' in filtered_df.columns:
            top_cats = filtered_df['Category'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=top_cats.values, y=top_cats.index, orientation='h', name='Top Categories'),
                row=2, col=1
            )
        
        # District distribution
        if 'District' in filtered_df.columns:
            district_counts = filtered_df['District'].value_counts()
            fig.add_trace(
                go.Bar(x=district_counts.index, y=district_counts.values, name='By District'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Crime Analysis Overview")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            if 'DayOfWeek' in filtered_df.columns:
                day_counts = filtered_df['DayOfWeek'].value_counts()
                fig_dow = px.pie(
                    values=day_counts.values, 
                    names=day_counts.index,
                    title="Crimes by Day of Week"
                )
                st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            if 'Year' in filtered_df.columns:
                yearly_trend = filtered_df['Year'].value_counts().sort_index()
                fig_trend = px.line(
                    x=yearly_trend.index, 
                    y=yearly_trend.values,
                    title="Yearly Crime Trend",
                    labels={'x': 'Year', 'y': 'Incidents'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
    else:
        st.info("No data available for analysis with current filters.")

with tab4:
    st.subheader("Police District Analysis")
    
    if 'District' in filtered_df.columns and 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        valid_data = filtered_df[['Latitude', 'Longitude', 'District']].dropna()
        valid_data = valid_data[
            (valid_data['Latitude'].between(-90, 90)) & 
            (valid_data['Longitude'].between(-180, 180))
        ]
        
        if not valid_data.empty:
            # District statistics summary
            district_stats = filtered_df.groupby('District').agg({
                'Hour': ['mean', 'std'],
                'Category': 'count'
            }).round(2)
            
            district_stats.columns = ['Avg Hour', 'Hour Std', 'Total Crimes']
            district_stats = district_stats.sort_values('Total Crimes', ascending=False)
            
            # Create district map
            center_lat = valid_data['Latitude'].mean()
            center_lon = valid_data['Longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add district centers (circle markers sized by count)
            for district in district_stats.index:
                district_data = valid_data[valid_data['District'] == district]
                if not district_data.empty:
                    center_lat_dist = district_data['Latitude'].mean()
                    center_lon_dist = district_data['Longitude'].mean()
                    count = len(district_data)
                    
                    folium.CircleMarker(
                        location=[center_lat_dist, center_lon_dist],
                        radius=max(10, min(50, count / 50)),
                        popup=folium.Popup(
                            f"<b>{district} District</b><br>"
                            f"Total Crimes: {count:,}<br>"
                            f"Avg Hour: {district_stats.loc[district, 'Avg Hour']}",
                            max_width=250
                        ),
                        color='blue',
                        fill=True,
                        fillOpacity=0.6,
                        fillColor='lightblue'
                    ).add_to(m)
            
            show_map(m)
            
            # District statistics table
            st.subheader("District Performance Metrics")
            st.dataframe(district_stats, use_container_width=True)
            
            # District comparison chart
            fig_dist = px.bar(
                x=district_stats.index,
                y=district_stats['Total Crimes'],
                title="Crime Distribution by District",
                labels={'x': 'District', 'y': 'Number of Crimes'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        else:
            st.warning("No valid coordinate data available for district analysis.")
    else:
        st.error("District or coordinate data not available.")

# -------------------- Debug Information --------------------
with st.sidebar.expander("🔧 Debug & Info"):
    st.write("**Data Information:**")
    st.write(f"Original columns: {data_info.get('columns', 'N/A')}")
    st.write(f"Original shape: {data_info.get('original_shape', 'N/A')}")
    st.write(f"Final shape: {data_info.get('final_shape', 'N/A')}")
    st.write(f"Rows with coordinates: {data_info.get('rows_with_coords', 0)}")
    
    st.write("**Detected Columns:**")
    for col_type, col_name in data_info.get('detected_columns', {}).items():
        st.write(f"- {col_type}: {col_name}")
    
    st.write("**Current Filters:**")
    st.write(f"- Years: {year_range[0]} - {year_range[1]}")
    st.write(f"- Hours: {hour_range[0]} - {hour_range[1]}")
    st.write(f"- Categories: {len(selected_categories)} selected")
    st.write(f"- Districts: {len(selected_districts)} selected")
    st.write(f"- Displaying: {len(filtered_df):,} records")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "**CityX Crime Watch Dashboard** | "
    "Real-time crime analysis and visualization | "
    "For official use only"
)
