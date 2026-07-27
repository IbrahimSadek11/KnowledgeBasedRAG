"""
Analytics Dashboard - SportLLM KPIs
====================================
Interactive dashboard to visualize key performance indicators and statistics
from the equestrian sports knowledge graph stored in Neo4j.

Features:
- Main KPIs (horses, events, trainings, riders, sensors)
- Sensor distribution analysis (positions, frequencies)
- Event type distribution
- Training analysis (intensity, frequency)
- Horse performance and participation metrics
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from backend.graph_rag.graph_service import init_graph, execute_query

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# COLOR PALETTE - SportLLM Theme
# ============================================================================

SPORTLLM_COLORS = {
    'primary_rose': '#C9A5A0',
    'secondary_blue': '#6B7E94',
    'primary_cream': '#F5EFE7',
    'secondary_beige': '#E8DDD0',
    'accent_brown': '#8B6F5C',
    'text_dark': '#2C3E50',
    'text_medium': '#5D6D7E',
    'text_light': '#95A5A6',
}

# Custom color scales for charts
SPORTLLM_SEQUENTIAL = ['#F5EFE7', '#E8DDD0', '#D4ACA7', '#C9A5A0', '#B89590', '#8B6F5C', '#7A5F4D']
SPORTLLM_DIVERGING = ['#6B7E94', '#8B9AAB', '#C9A5A0', '#B89590', '#8B6F5C']
SPORTLLM_CATEGORICAL = ['#C9A5A0', '#6B7E94', '#8B6F5C', '#E8DDD0', '#B89590', '#7A8B99']

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


# Beautiful UI Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden !important;}
header {visibility: hidden;}
.stDeployButton {display: none;}

.main {background: #FDFBF7;}
.block-container {padding: 2rem 3rem; max-width: 1400px;}

h1, h2, h3 {font-family: 'Playfair Display', serif; color: #C9A5A0; font-weight: 600;}
body, p, div, span {font-family: 'Inter', sans-serif;}

.main-header {font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 600; color: #C9A5A0; margin-bottom: 2rem;}
.section-header {font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 600; margin: 2rem 0 1rem 0; color: #C9A5A0;}

[data-testid="stMetric"] {background: white; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #C9A5A0; box-shadow: 0 4px 12px rgba(201, 165, 160, 0.15); transition: all 0.3s ease;}
[data-testid="stMetric"]:hover {box-shadow: 0 6px 16px rgba(201, 165, 160, 0.25); transform: translateY(-2px);}
[data-testid="stMetric"] label {color: #5D6D7E; font-family: 'Inter', sans-serif; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.85rem;}
[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #C9A5A0; font-weight: 600; font-size: 2rem;}

[data-testid="stSidebar"] {background: linear-gradient(180deg, #F5EFE7 0%, #E8DDD0 100%); border-right: 2px solid #D4ACA7;}

hr {border-color: #D4ACA7; margin: 2rem 0;}

::-webkit-scrollbar {width: 8px;}
::-webkit-scrollbar-track {background: #F5EFE7;}
::-webkit-scrollbar-thumb {background: #C9A5A0; border-radius: 4px;}
::-webkit-scrollbar-thumb:hover {background: #B89590;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_graph():
    """
    Initialize connection to Neo4j graph database
    Uses Streamlit caching to persist connection across reruns
    
    Returns:
        Neo4jGraph: Graph database connection object, or None on error
    """
    try:
        graph = init_graph()
        return graph
    except Exception as e:
        st.error(f"Erreur de connexion à Neo4j: {e}")
        return None

# ============================================================================
# KPI RETRIEVAL FUNCTIONS
# ============================================================================

def get_kpis(graph):
    """
    Retrieve main Key Performance Indicators from the knowledge graph
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        dict: Dictionary containing counts for horses, events, trainings, riders, sensors, objectives
    """
    kpis = {}
    
    # Count total horses in the system
    query = "MATCH (h:Horse) RETURN count(h) as count"
    result = execute_query(graph, query)
    kpis['horses'] = result[0]['count'] if result else 0
    
    # Count total events (all types: ShowJumping, Dressage, Cross)
    query = "MATCH (e) WHERE e:ShowJumping OR e:Dressage OR e:Cross RETURN count(e) as count"
    result = execute_query(graph, query)
    kpis['events'] = result[0]['count'] if result else 0
    
    # Count total training sessions (all stages)
    query = "MATCH (t) WHERE t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage RETURN count(t) as count"
    result = execute_query(graph, query)
    kpis['trainings'] = result[0]['count'] if result else 0
    
    # Count total riders
    query = "MATCH (r:Rider) RETURN count(r) as count"
    result = execute_query(graph, query)
    kpis['riders'] = result[0]['count'] if result else 0
    
    # Count total inertial sensors
    query = "MATCH (s:InertialSensors) RETURN count(s) as count"
    result = execute_query(graph, query)
    kpis['sensors'] = result[0]['count'] if result else 0
    
    # Count experimental objectives (gait classification, fatigue detection)
    query = """
    MATCH (o)
    WHERE o.id IS NOT NULL 
    AND (o.id CONTAINS 'Gait' OR o.id CONTAINS 'Fatigue' OR o.id CONTAINS 'Classif' OR o.id CONTAINS 'Detection')
    RETURN count(DISTINCT o) as count
    """
    result = execute_query(graph, query)
    kpis['objectives'] = result[0]['count'] if result else 0
    
    return kpis

# ============================================================================
# SENSOR ANALYSIS FUNCTIONS
# ============================================================================

def get_sensor_positions_distribution(graph):
    """
    Get distribution of sensors by anatomical position
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Sensor positions and their counts
    """
    query = """
    MATCH (s:InertialSensors)
    WITH labels(s) as all_labels
    UNWIND all_labels as label
    WITH label WHERE label <> 'InertialSensors'
    RETURN label as position, count(*) as count
    ORDER BY count DESC
    """
    result = execute_query(graph, query)
    return pd.DataFrame(result)


def get_sensor_frequencies(graph):
    """
    Get distribution of sensor sampling frequencies
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Sampling frequencies and their counts
    """
    query = """
    MATCH (s:InertialSensors)
    WHERE s.hasSensorTime IS NOT NULL
    RETURN s.hasSensorTime as frequency, count(*) as count
    ORDER BY frequency
    """
    result = execute_query(graph, query)
    return pd.DataFrame(result)

# ============================================================================
# EVENT ANALYSIS FUNCTIONS
# ============================================================================

def get_event_types_distribution(graph):
    """
    Get distribution of event types (ShowJumping, Dressage, Cross)
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Event types and their counts
    """
    query = """
    MATCH (e)
    WHERE e:ShowJumping OR e:Dressage OR e:Cross
    WITH labels(e) as labels
    UNWIND labels as label
    WITH label WHERE label IN ['ShowJumping', 'Dressage', 'Cross']
    RETURN label as type, count(*) as count
    """
    result = execute_query(graph, query)
    
    if result:
        return pd.DataFrame(result)
    
    return pd.DataFrame(columns=['type', 'count'])

# ============================================================================
# TRAINING ANALYSIS FUNCTIONS
# ============================================================================

def get_training_intensity_distribution(graph):
    """
    Get distribution of training sessions by intensity level
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Intensity levels and their counts
    """
    query = """
    MATCH (t)
    WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
    AND t.Intensity IS NOT NULL
    RETURN t.Intensity as intensity, count(*) as count
    ORDER BY count DESC
    """
    result = execute_query(graph, query)
    
    return pd.DataFrame(result)


def get_training_frequency_stats(graph):
    """
    Get statistics on training frequencies (sessions per week)
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Training frequencies and their counts
    """
    query = """
    MATCH (t)
    WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
    AND t.Frequency IS NOT NULL
    RETURN t.Frequency as frequency, count(*) as count
    ORDER BY frequency
    """
    result = execute_query(graph, query)
    
    return pd.DataFrame(result)

# ============================================================================
# HORSE PERFORMANCE FUNCTIONS
# ============================================================================

def get_horse_participation(graph):
    """
    Analyze horse participation in events
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Horses and their event participation counts
    """
    # Try first relationship pattern
    query = """
    MATCH (h:Horse)-[:COMPETESIN]->(e)
    WHERE e:ShowJumping OR e:Dressage OR e:Cross
    RETURN h.hasName as horse_name, count(e) as event_count
    ORDER BY event_count DESC
    LIMIT 10
    """
    result = execute_query(graph, query)
    
    # Try alternative pattern if first one returns no results
    if not result:
        query = """
        MATCH (e)-[:HASPARTICIPATION]->(p:EventParticipation)-[:HASHORSE]->(h:Horse)
        WHERE e:ShowJumping OR e:Dressage OR e:Cross
        RETURN h.hasName as horse_name, count(DISTINCT e) as event_count
        ORDER BY event_count DESC
        LIMIT 10
        """
        result = execute_query(graph, query)
    
    return pd.DataFrame(result)


def get_horse_training_intensity(graph):
    """
    Get relationship between horses and their training intensity levels
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        DataFrame: Horses, intensity levels, and session counts
    """
    # Try primary query pattern
    query = """
    MATCH (h:Horse)-[:TRAINSIN]->(t)
    WHERE (t:PreparationStage OR t:PreCompetitionStage OR t:CompetitionStage OR t:TransitionStage)
    AND t.Intensity IS NOT NULL AND h.hasName IS NOT NULL
    RETURN h.hasName as horse, t.Intensity as intensity, count(*) as count
    """
    result = execute_query(graph, query)
    
    # Fallback queries if needed
    if not result:
        query = """
        MATCH (h:Horse)-[:TRAINSIN]->(t)
        WHERE t.Intensity IS NOT NULL AND h.hasName IS NOT NULL
        RETURN h.hasName as horse, t.Intensity as intensity, count(*) as count
        """
        result = execute_query(graph, query)
    
    if not result:
        query = """
        MATCH (h:Horse)-[r]->(t)
        WHERE t.Intensity IS NOT NULL AND h.hasName IS NOT NULL
        RETURN h.hasName as horse, t.Intensity as intensity, count(*) as count, type(r) as relation_type
        """
        result = execute_query(graph, query)
    
    return pd.DataFrame(result)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main dashboard application function
    Orchestrates the display of all KPIs and visualizations
    """
    
    # Page header
    st.markdown('<h1 class="main-header">Dashboard Analytics</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Refresh button to reload data
        refresh_button = st.button("Actualiser les données", use_container_width=True)
        
        
        st.markdown("---")
        st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Initialize graph connection
    graph = get_graph()
    
    if graph is None:
        st.error("Impossible de se connecter à Neo4j. Vérifiez vos paramètres de connexion.")
        st.info("Assurez-vous que Neo4j est en cours d'exécution et que les identifiants sont corrects.")
        return
    
    # Retrieve KPIs from database
    with st.spinner("Chargement des données..."):
        kpis = get_kpis(graph)
    
    # ========================================================================
    # SECTION 1: MAIN KPIs
    # ========================================================================
    
    st.markdown("## KPIs Principaux")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Chevaux",
            value=kpis['horses'],
            help="Nombre total de chevaux dans le système"
        )
    
    with col2:
        st.metric(
            label="Événements",
            value=kpis['events'],
            help="Nombre total d'événements sportifs"
        )
    
    with col3:
        st.metric(
            label="Entraînements",
            value=kpis['trainings'],
            help="Nombre total de sessions d'entraînement"
        )
    
    with col4:
        st.metric(
            label="Cavaliers",
            value=kpis['riders'],
            help="Nombre total de cavaliers enregistrés"
        )
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: SENSOR KPIs
    # ========================================================================
    
    st.markdown("## KPIs Capteurs")
    
    st.metric(
        label="Capteurs IMU",
        value=kpis['sensors'],
        help="Nombre total de capteurs dans le système"
    )
    
    # Display sensor details if sensors exist
    if kpis['sensors'] > 0:
        st.markdown("### Détails des Capteurs")
        
        col_sensor1, col_sensor2 = st.columns(2)
        
        with col_sensor1:
            st.markdown("**Positions anatomiques**")
            sensor_positions = get_sensor_positions_distribution(graph)
            if not sensor_positions.empty:
                # Clean position names for display
                sensor_positions['position_clean'] = sensor_positions['position'].str.replace('IMU_', '')
                fig = px.pie(
                    sensor_positions,
                    values='count',
                    names='position_clean',
                    title='Répartition par position',
                    color_discrete_sequence=SPORTLLM_CATEGORICAL
                )
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    marker=dict(line=dict(color='white', width=2))
                )
                fig.update_layout(
                    height=400, 
                    showlegend=False,
                    font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée de position disponible")
        
        with col_sensor2:
            st.markdown("**Fréquences d'échantillonnage**")
            sensor_freq = get_sensor_frequencies(graph)
            if not sensor_freq.empty:
                fig = px.bar(
                    sensor_freq,
                    x='frequency',
                    y='count',
                    title='Distribution des fréquences',
                    labels={'frequency': 'Fréquence', 'count': 'Nombre'}
                )
                fig.update_traces(
                    marker_color=SPORTLLM_COLORS['primary_rose'],
                    marker_line_color=SPORTLLM_COLORS['accent_brown'],
                    marker_line_width=1.5
                )
                fig.update_layout(
                    height=400, 
                    showlegend=False,
                    font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#E8DDD0')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée de fréquence disponible")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 3: EVENT DISTRIBUTION
    # ========================================================================
    
    st.markdown("## Distribution des Types d'Événements")
    
    event_dist = get_event_types_distribution(graph)
    
    if not event_dist.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart for event type distribution
            fig = px.pie(
                event_dist, 
                values='count', 
                names='type',
                title='Répartition des types d\'événements',
                color_discrete_sequence=SPORTLLM_CATEGORICAL,
                hole=0.4
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='white', width=2))
            )
            fig.update_layout(
                font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Detailed breakdown with progress bars
            st.markdown("### Détails")
            for _, row in event_dist.iterrows():
                st.markdown(f"**{row['type']}**: {row['count']} événement(s)")
                percentage = (row['count'] / event_dist['count'].sum()) * 100
                st.progress(percentage / 100)
                st.caption(f"{percentage:.1f}% du total")
                st.markdown("---")
    else:
        st.info("Aucune donnée d'événement disponible.")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 4: TRAINING ANALYSIS
    # ========================================================================
    
    st.markdown("## Analyse des Entraînements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribution par Intensité")
        intensity_dist = get_training_intensity_distribution(graph)
        
        if not intensity_dist.empty:
            # Normalize intensity values (handle French/English variations)
            intensity_normalize = {
                'Low': 'Low',
                'Moderate': 'Moderate',
                'Modérée': 'Moderate',
                'High': 'High',
                'Élevée': 'High',
                'Peak': 'Peak'
            }
            
            intensity_dist['intensity_normalized'] = intensity_dist['intensity'].map(lambda x: intensity_normalize.get(x, x))
            intensity_grouped = intensity_dist.groupby('intensity_normalized')['count'].sum().reset_index()
            intensity_grouped.columns = ['intensity', 'count']
            
            # Order intensities logically
            order = ['Low', 'Moderate', 'High', 'Peak']
            intensity_grouped['intensity'] = pd.Categorical(intensity_grouped['intensity'], categories=order, ordered=True)
            intensity_grouped = intensity_grouped.sort_values('intensity')
            
            # Custom colors for intensity levels
            intensity_colors = {
                'Low': '#E8DDD0',
                'Moderate': '#C9A5A0',
                'High': '#8B6F5C',
                'Peak': '#6B7E94'
            }
            
            fig = px.bar(
                intensity_grouped,
                x='intensity',
                y='count',
                title='Nombre d\'entraînements par intensité',
                labels={'intensity': 'Intensité', 'count': 'Nombre'},
                color='intensity',
                color_discrete_map=intensity_colors
            )
            fig.update_layout(
                showlegend=False,
                font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E8DDD0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée d'intensité disponible.")
    
    with col2:
        st.markdown("### Distribution par Fréquence")
        frequency_stats = get_training_frequency_stats(graph)
        
        if not frequency_stats.empty:
            fig = px.line(
                frequency_stats,
                x='frequency',
                y='count',
                title='Nombre d\'entraînements par fréquence',
                markers=True,
                labels={'frequency': 'Fréquence (séances/semaine)', 'count': 'Nombre'}
            )
            fig.update_traces(
                line_color=SPORTLLM_COLORS['primary_rose'], 
                line_width=3, 
                marker=dict(size=10, color=SPORTLLM_COLORS['accent_brown'], line=dict(color='white', width=2))
            )
            fig.update_layout(
                font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E8DDD0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de fréquence disponible.")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 5: HORSE PERFORMANCE
    # ========================================================================
    
    st.markdown("## Performance et Participation des Chevaux")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Participation aux Événements")
        horse_participation = get_horse_participation(graph)
        
        if not horse_participation.empty:
            # Horizontal bar chart for horse participation
            fig = px.bar(
                horse_participation,
                x='event_count',
                y='horse_name',
                orientation='h',
                title='Chevaux les plus actifs en compétition',
                labels={'event_count': 'Nombre d\'événements', 'horse_name': 'Cheval'}
            )
            fig.update_traces(
                marker_color=SPORTLLM_SEQUENTIAL,
                marker_line_color=SPORTLLM_COLORS['accent_brown'],
                marker_line_width=1
            )
            fig.update_layout(
                showlegend=False, 
                height=400,
                font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#E8DDD0'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de participation disponible.")
    
    with col2:
        st.markdown("### Intensité d'Entraînement par Cheval")
        horse_training = get_horse_training_intensity(graph)
        
        if not horse_training.empty:
            # Normalize intensity values
            intensity_normalize = {
                'Low': 'Low',
                'Moderate': 'Moderate',
                'Modérée': 'Moderate',
                'High': 'High',
                'Élevée': 'High',
                'Peak': 'Peak'
            }
            horse_training['intensity_normalized'] = horse_training['intensity'].map(lambda x: intensity_normalize.get(x, x))
            
            # Group by horse and normalized intensity
            horse_training_grouped = horse_training.groupby(['horse', 'intensity_normalized'])['count'].sum().reset_index()
            
            # Pivot data for stacked bar chart
            pivot_data = horse_training_grouped.pivot_table(
                index='horse', 
                columns='intensity_normalized', 
                values='count', 
                fill_value=0
            ).reset_index()
            
            # Create stacked bar chart
            fig = go.Figure()
            
            # Define intensity colors - SportLLM palette
            intensity_config = [
                ('Low', '#E8DDD0'),
                ('Moderate', '#C9A5A0'),
                ('High', '#8B6F5C'),
                ('Peak', '#6B7E94')
            ]
            
            # Add bars for each intensity level
            for intensity_name, color in intensity_config:
                if intensity_name in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        name=intensity_name,
                        x=pivot_data['horse'],
                        y=pivot_data[intensity_name],
                        marker_color=color,
                        marker_line_color='white',
                        marker_line_width=1
                    ))
            
            fig.update_layout(
                barmode='stack',
                title='Répartition des intensités d\'entraînement',
                xaxis_title='Cheval',
                yaxis_title='Nombre d\'entraînements',
                height=400,
                font=dict(family='Inter', color=SPORTLLM_COLORS['text_dark']),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E8DDD0')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée d'intensité par cheval disponible.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()