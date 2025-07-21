import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Forest Plot - Xeljanz Safety Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3498DB;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498DB;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div > select {
        background-color: #FFFFFF;
        border: 2px solid #BDC3C7;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Analyse des Taux d\'Incidence - Xeljanz</h1>', unsafe_allow_html=True)

st.markdown("""
### 📋 Description
Cette application présente l'analyse des taux d'incidence des effets indésirables pour différentes dosages de Xeljanz 
avec leurs intervalles de confiance à 95%.
""")

# Sidebar for controls
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

# Data preparation
@st.cache_data
def load_data():
    data = {
        'Effet indésirable': [
            'Décès', 'Décès', 'Décès',
            'Infections graves', 'Infections graves', 'Infections graves',
            'Zona (non grave et grave)', 'Zona (non grave et grave)', 'Zona (non grave et grave)',
            'Zona grave', 'Zona grave', 'Zona grave',
            'Infections opportunistes', 'Infections opportunistes', 'Infections opportunistes',
            'Cancers (excluant NMSC)', 'Cancers (excluant NMSC)', 'Cancers (excluant NMSC)',
            'NMSC', 'NMSC', 'NMSC',
            'MACE', 'MACE', 'MACE',
            'Perforations gastro-intestinales', 'Perforations gastro-intestinales', 'Perforations gastro-intestinales',
            'Thrombose veineuse profonde', 'Thrombose veineuse profonde', 'Thrombose veineuse profonde',
            'Embolie pulmonaire', 'Embolie pulmonaire', 'Embolie pulmonaire'
        ],
        'Groupe': [
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global',
            'Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global'
        ],
        'TI': [0.0, 0.33, 0.25, 1.25, 1.74, 1.61, 2.08, 3.55, 3.16, 0.16, 0.33, 0.29,
               0.62, 0.72, 0.87, 1.25, 1.03, 1.06, 0.93, 0.85, 0.75, 0.31, 0.31, 0.16,
               0.16, 0.06, 0.04, 0.0, 0.06, 0.04, 0.0, 0.06, 0.2],
        'IC95_min': [0.0, 0.12, 0.09, 0.54, 1.18, 1.14, 1.11, 2.71, 2.47, 0.0, 0.12, 0.12,
                     0.17, 0.39, 0.54, 0.5, 0.6, 0.73, 0.34, 0.43, 0.45, 0.04, 0.08, 0.04,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07],
        'IC95_max': [0.57, 0.73, 0.54, 2.6, 2.47, 2.2, 3.55, 4.58, 3.97, 0.87, 0.73, 0.59,
                     1.58, 1.22, 1.33, 2.57, 1.59, 1.5, 2.03, 1.48, 1.19, 1.13, 0.8, 0.42,
                     0.87, 0.31, 0.23, 0.57, 0.31, 0.23, 0.57, 0.31, 0.48]
    }
    
    df = pd.DataFrame(data)
    df['Error_Low'] = df['TI'] - df['IC95_min']
    df['Error_High'] = df['IC95_max'] - df['TI']
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.subheader("📊 Filtres")

# Group selection
groups_available = df['Groupe'].unique()
selected_groups = st.sidebar.multiselect(
    "Sélectionner les groupes:",
    options=groups_available,
    default=groups_available,
    help="Choisissez les groupes de dosage à afficher"
)

# Effect selection
effects_available = df['Effet indésirable'].unique()
selected_effects = st.sidebar.multiselect(
    "Sélectionner les effets indésirables:",
    options=effects_available,
    default=effects_available,
    help="Choisissez les effets indésirables à afficher"
)

# Plot dimensions
st.sidebar.subheader("📐 Dimensions du graphique")
plot_height = st.sidebar.slider("Hauteur", min_value=600, max_value=1200, value=900, step=50)
plot_width = st.sidebar.slider("Largeur", min_value=800, max_value=1500, value=1300, step=50)

# Color theme selection
st.sidebar.subheader("🎨 Thème de couleurs")
color_theme = st.sidebar.selectbox(
    "Choisir un thème:",
    options=["Classique", "Médical", "Moderne"],
    index=0
)

# Define color themes
color_themes = {
    "Classique": {
        'Xeljanz 5 mg 2x/j': '#1f77b4',
        'Xeljanz 10 mg 2x/j': '#ff7f0e',
        'Xeljanz global': '#2ca02c'
    },
    "Médical": {
        'Xeljanz 5 mg 2x/j': '#2E86C1',
        'Xeljanz 10 mg 2x/j': '#E74C3C',
        'Xeljanz global': '#27AE60'
    },
    "Moderne": {
        'Xeljanz 5 mg 2x/j': '#6C5CE7',
        'Xeljanz 10 mg 2x/j': '#FD79A8',
        'Xeljanz global': '#00CEC9'
    }
}

group_colors = color_themes[color_theme]

# Filter data based on selections
filtered_df = df[
    (df['Groupe'].isin(selected_groups)) & 
    (df['Effet indésirable'].isin(selected_effects))
].copy()

# Show data summary
st.markdown("### 📈 Résumé des données")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Effets sélectionnés", len(selected_effects))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Groupes sélectionnés", len(selected_groups))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Points de données", len(filtered_df))
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    max_ti = filtered_df['TI'].max()
    st.metric("TI Maximum", f"{max_ti:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Create the forest plot
def create_forest_plot(data, height, width, colors):
    if data.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés.")
        return None
    
    # Create y-axis positions and labels
    unique_effects = data['Effet indésirable'].unique()
    n_effects = len(unique_effects)
    
    group_order = ['Xeljanz 5 mg 2x/j', 'Xeljanz 10 mg 2x/j', 'Xeljanz global']
    
    y_positions = []
    y_labels = []
    
    for idx, row in data.iterrows():
        effect = row['Effet indésirable']
        group = row['Groupe']
        
        effect_idx = list(unique_effects).index(effect)
        if group in group_order:
            group_idx = group_order.index(group)
        else:
            continue
        
        base_pos = (n_effects - effect_idx - 1) * 3
        y_pos = base_pos - group_idx * 0.7
        y_positions.append(y_pos)
        
        if group == 'Xeljanz 10 mg 2x/j':
            y_labels.append(effect)
        else:
            y_labels.append('')
    
    data = data.copy()
    data['y_pos'] = y_positions[:len(data)]
    
    # Calculate x-axis range
    x_min = data['IC95_min'].min()
    x_max = data['IC95_max'].max()
    x_range_min = max(0, x_min - 0.1)
    x_range_max = x_max + 0.1
    
    # Create the plot
    fig = go.Figure()
    
    # Add background regions
    fig.add_vrect(
        x0=x_range_min, x1=1,
        fillcolor="lightgreen", opacity=0.2,
        layer="below", line_width=0,
    )
    
    fig.add_vrect(
        x0=1, x1=x_range_max,
        fillcolor="lightcoral", opacity=0.2,
        layer="below", line_width=0,
    )
    
    # Add data points
    for group in group_order:
        if group not in selected_groups:
            continue
            
        group_data = data[data['Groupe'] == group]
        if group_data.empty:
            continue
        
        error_low = np.maximum(group_data['Error_Low'], 0)
        error_high = np.maximum(group_data['Error_High'], 0)
        
        fig.add_trace(go.Scatter(
            x=group_data['TI'],
            y=group_data['y_pos'],
            mode='markers',
            marker=dict(
                color=colors[group],
                size=10 if group == 'Xeljanz global' else 8,
                symbol='diamond' if group == 'Xeljanz global' else 'circle',
                line=dict(width=1, color='black')
            ),
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_high,
                arrayminus=error_low,
                color=colors[group],
                thickness=2,
                width=3
            ),
            name=group,
            hovertemplate=f'<b>{group}</b><br>' +
                         'Effet: %{customdata[0]}<br>' +
                         'TI: %{x:.3f}<br>' +
                         'IC 95%: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>',
            customdata=np.column_stack((group_data['Effet indésirable'],
                                       group_data['IC95_min'],
                                       group_data['IC95_max']))
        ))
    
    # Add reference line
    fig.add_vline(x=1, line_dash="dash", line_color="black", line_width=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Taux d\'incidence avec intervalles de confiance à 95%',
            x=0.5,
            font=dict(size=18, color="#2C3E50")
        ),
        xaxis=dict(
            title=dict(text='Taux d\'incidence (IC 95%)', font=dict(size=14)),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[x_range_min, x_range_max],
            dtick=0.5,
        ),
        yaxis=dict(
            title=dict(text='Effets indésirables', font=dict(size=14)),
            tickmode='array',
            tickvals=[pos for i, pos in enumerate(y_positions[:len(data)]) if i < len(y_labels) and y_labels[i] != ''],
            ticktext=[label for label in y_labels if label != ''],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[-2, max(y_positions[:len(data)]) + 2] if y_positions else [0, 1]
        ),
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=300, r=50, t=100, b=100),
        plot_bgcolor='white'
    )
    
    return fig

# Generate and display the plot
st.markdown("### 📊 Forest Plot")

if len(selected_groups) > 0 and len(selected_effects) > 0:
    fig = create_forest_plot(filtered_df, plot_height, plot_width, group_colors)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.markdown("### 💾 Téléchargement")
        if st.button("📥 Télécharger le graphique (HTML)", type="primary"):
            fig.write_html("forest_plot_streamlit.html")
            st.success("✅ Graphique sauvegardé sous 'forest_plot_streamlit.html'")
else:
    st.warning("⚠️ Veuillez sélectionner au moins un groupe et un effet indésirable.")

# Data table
st.markdown("### 📋 Tableau des données")
if st.checkbox("Afficher les données détaillées"):
    st.dataframe(
        filtered_df[['Effet indésirable', 'Groupe', 'TI', 'IC95_min', 'IC95_max']].round(3),
        use_container_width=True
    )

# Information section
st.markdown("---")
st.markdown("""
### ℹ️ Informations
- **Zone verte** : Taux d'incidence favorable (TI < 1)
- **Zone rouge** : Taux d'incidence défavorable (TI > 1)
- **Ligne verticale** : Référence (TI = 1)
- **Barres d'erreur** : Intervalles de confiance à 95%
""")

st.markdown("---")
st.markdown("*Application développée avec Streamlit et Plotly*")

st.markdown("*Jasmine kadji*")