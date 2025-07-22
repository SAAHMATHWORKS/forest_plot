import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import openai # Importation de la bibliothèque OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')

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
    
    .chat-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Analyse des Taux d\'Incidence - Xeljanz</h1>', unsafe_allow_html=True)

st.markdown("""
### 📋 Description
Cette application présente l'analyse des taux d'incidence des effets indésirables pour différents dosages de Xeljanz 
avec leurs intervalles de confiance à 95%.
""")

# Sidebar for controls
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

# Data preparation
@st.cache_data
def load_data():
    # Données corrigées avec des valeurs cohérentes
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
        'Nombre de cas': [
            0, 7, 8,
            8, 59, 67,
            13, 120, 134,
            1, 11, 12,
            4, 24, 34,
            8, 40, 55,
            6, 33, 39,
            2, 12, 14,
            1, 2, 3,
            0, 2, 2,
            0, 2, 5
        ],
        'Total Patients': [
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125,
            175, 772, 1125
        ],
        'TI': [0.0, 0.33, 0.25, 1.25, 1.74, 1.61, 2.08, 3.55, 3.16, 0.16, 0.33, 0.29,
               0.63, 0.96, 0.87, 1.09, 1.00, 1.03, 0.96, 0.68, 0.75, 0.31, 0.11, 0.16,
               0.16, 0.06, 0.08, 0.0, 0.06, 0.04, 0.0, 0.28, 0.21],
        'IC95_min': [0.0, 0.12, 0.00, 0.54, 1.18, 1.14, 1.11, 2.71, 2.47, 0.0, 0.12, 0.12,
                     0.17, 0.56, 0.54, 0.44, 0.6, 0.67, 0.35, 0.35, 0.45, 0.04, 0.01, 0.04,
                     0.0, 0.0, 0.01, 0.0, 0.0, 0.00, 0.0, 0.09, 0.07],
        'IC95_max': [0.57, 0.73, 0.54, 2.46, 2.47, 2.2, 3.55, 4.58, 3.97, 0.87, 0.73, 0.59,
                     1.60, 1.53, 1.33, 2.25, 1.59, 1.52, 2.08, 1.19, 1.19, 1.13, 0.40, 0.42,
                     0.87, 0.31, 0.30, 0.52, 0.32, 0.23, 0.57, 0.65, 0.48]
    }
    
    df = pd.DataFrame(data)
    # Recalculate percentages based on actual data
    df['Pourcentage'] = (df['Nombre de cas'] / df['Total Patients'] * 100).round(2)
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
    max_ti = filtered_df['TI'].max() if not filtered_df.empty else 0
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
                         'IC 95%: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>' +
                         'Nombre de cas: %{customdata[3]}<br>' +
                         'Total Patients: %{customdata[4]}<br>' +
                         'Pourcentage: %{customdata[5]:.2f}%<extra></extra>',
            customdata=np.column_stack((group_data['Effet indésirable'],
                                       group_data['IC95_min'],
                                       group_data['IC95_max'],
                                       group_data['Nombre de cas'],
                                       group_data['Total Patients'],
                                       group_data['Pourcentage']))
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
            try:
                fig.write_html("forest_plot_streamlit.html")
                st.success("✅ Graphique sauvegardé sous 'forest_plot_streamlit.html'")
            except Exception as e:
                st.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")
else:
    st.warning("⚠️ Veuillez sélectionner au moins un groupe et un effet indésirable.")

# Data table
st.markdown("### 📋 Tableau des données")
if st.checkbox("Afficher les données détaillées"):
    st.dataframe(
        filtered_df[['Effet indésirable', 'Groupe', 'Nombre de cas', 'Total Patients', 'Pourcentage', 'TI', 'IC95_min', 'IC95_max']].round(3),
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
- **TI (Taux d'Incidence)** : Mesure du risque relatif d'occurrence d'un effet indésirable
- **IC 95%** : Intervalle de confiance à 95% - plage dans laquelle la vraie valeur a 95% de chance de se trouver
""")

st.markdown("---")
st.markdown("*Application développée avec Streamlit et Plotly*")
# Section Chatbot (Réintégration du chatbot OpenAI)
st.markdown("---")
st.markdown("### 💬 Chatbot - Analyse des données")
st.markdown("Posez-moi une question sur les données des effets indésirables du Xeljanz.")

st.markdown('<div class="chat-container">', unsafe_allow_html=True) # Utilisation du style .chat-container

# Initialiser la connexion à l'API OpenAI
# Assurez-vous que votre clé API est bien configurée dans .streamlit/secrets.toml
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except AttributeError:
    st.error("❌ Clé API OpenAI non trouvée. Veuillez la configurer dans .streamlit/secrets.toml.")
    st.stop() # Arrête l'exécution de l'application si la clé n'est pas trouvée

# Initialiser l'historique de la conversation dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Créer un prompt avec le contexte des données
# On utilise la version de la DataFrame sans filtres pour donner un contexte complet
data_context = df.to_string() # df.to_string() inclura toutes les colonnes
system_prompt = (
    "Tu es un assistant expert en analyse de données médicales pour le médicament Xeljanz. "
    "Réponds aux questions de l'utilisateur de manière concise et précise, en te basant exclusivement "
    "sur les données fournies ci-dessous. Si une information n'est pas présente, précise-le. "
    "Les colonnes sont : 'Effet indésirable', 'Groupe' (dosage), 'Nombre de cas', 'Total Patients', 'Pourcentage', 'TI' (Taux d'Incidence), 'IC95_min' et 'IC95_max' "
    "(intervalle de confiance à 95%). "
    "Explique clairement les concepts si l'utilisateur semble ne pas les connaître (ex: TI, IC95%, Nombre de cas, Total Patients). "
    "Ne fais pas de spéculations au-delà des données fournies. "
    "Voici les données : \n\n"
    f"{data_context}"
)

# Accepter l'entrée de l'utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appeler l'API OpenAI pour obtenir une réponse
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            messages_to_send = [
                {"role": "system", "content": system_prompt}
            ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini", # Vous pouvez changer pour un autre modèle si besoin "gpt-4o-mini", "gpt-3.5-turbo"
                    messages=messages_to_send
                )
                assistant_response = response.choices[0].message.content
                st.markdown(assistant_response)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel à l'API OpenAI: {str(e)}")
                assistant_response = "Désolé, une erreur est survenue lors de la communication avec l'IA. Veuillez réessayer plus tard."
                st.markdown(assistant_response)
    
    # Ajouter la réponse de l'assistant à l'historique
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

st.markdown('</div>', unsafe_allow_html=True) # Fermeture du style .chat-container

