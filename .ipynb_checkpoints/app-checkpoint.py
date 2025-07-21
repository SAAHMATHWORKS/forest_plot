import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Forest Plot Xeljanz - Analyse Interactive",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .risk-high {
        background-color: #ffe6e6;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
    }
    
    .risk-low {
        background-color: #e6ffe6;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    
    .sidebar .stSelectbox {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charger les données depuis le CSV"""
    # Créer les données directement dans le code pour la démo
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
    
    # Calculer les barres d'erreur
    df['Error_Low'] = df['TI'] - df['IC95_min']
    df['Error_High'] = df['IC95_max'] - df['TI']
    
    return df

def create_forest_plot(df_filtered, title_suffix=""):
    """Créer le forest plot avec les données filtrées"""
    
    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour les critères sélectionnés.")
        return None
    
    # Inverser l'ordre pour afficher le premier effet en haut
    df_plot = df_filtered[::-1].reset_index(drop=True)
    df_plot['y_position'] = range(len(df_plot))
    
    # Couleurs pour chaque groupe
    couleurs = {
        'Xeljanz 5 mg 2x/j': '#1f77b4',
        'Xeljanz 10 mg 2x/j': '#ff7f0e', 
        'Xeljanz global': '#2ca02c'
    }
    
    fig = go.Figure()
    
    # Ajouter les traces pour chaque groupe présent
    for groupe in df_plot['Groupe'].unique():
        data_groupe = df_plot[df_plot['Groupe'] == groupe]
        
        fig.add_trace(go.Scatter(
            x=data_groupe['TI'],
            y=data_groupe['y_position'],
            mode='markers',
            name=groupe,
            marker=dict(
                color=couleurs.get(groupe, '#333333'),
                size=14,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            error_x=dict(
                type='data',
                symmetric=False,
                array=data_groupe['Error_High'],
                arrayminus=data_groupe['Error_Low'],
                thickness=3,
                width=8,
                color=couleurs.get(groupe, '#333333')
            ),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Groupe: %{customdata[1]}<br>'
                'TI: %{x:.3f}<br>'
                'IC95%: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>'
                '<extra></extra>'
            ),
            customdata=data_groupe[['Effet indésirable', 'Groupe', 'IC95_min', 'IC95_max']].values
        ))
    
    # Ligne de référence
    fig.add_vline(
        x=1, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text="Référence (TI = 1)",
        annotation_position="top"
    )
    
    # Zones de risque
    fig.add_vrect(
        x0=0.001, x1=1, 
        fillcolor="lightgreen", opacity=0.1,
        annotation_text="Effet protecteur", 
        annotation_position="top left"
    )
    
    fig.add_vrect(
        x0=1, x1=100, 
        fillcolor="lightcoral", opacity=0.1,
        annotation_text="Risque accru", 
        annotation_position="top right"
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': f'<b>Forest Plot - Effets Indésirables Xeljanz{title_suffix}</b><br><sub>Taux d\'Incidence avec IC 95%</sub>',
            'font': {'size': 18, 'color': '#2c3e50'},
            'x': 0.5
        },
        
        xaxis={
            'title': 'Taux d\'Incidence (TI) - Échelle logarithmique',
            'type': 'log',
            'range': [-3, 2],
            'showgrid': True,
            'gridcolor': 'lightgray',
            'tickfont': {'size': 11}
        },
        
        yaxis={
            'title': 'Effets Indésirables',
            'tickmode': 'array',
            'tickvals': df_plot['y_position'],
            'ticktext': df_plot['Effet indésirable'],
            'showgrid': True,
            'gridcolor': 'lightgray',
            'tickfont': {'size': 11}
        },
        
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(400, len(df_plot) * 30 + 150),
        
        legend={
            'x': 1.02,
            'y': 1,
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': 'gray',
            'borderwidth': 1
        },
        
        font={'family': 'Arial, sans-serif'}
    )
    
    return fig

def calculate_stats(df):
    """Calculer les statistiques descriptives"""
    stats = {}
    
    stats['total_effects'] = df['Effet indésirable'].nunique()
    stats['total_observations'] = len(df)
    stats['total_groups'] = df['Groupe'].nunique()
    
    stats['highest_ti'] = df['TI'].max()
    stats['lowest_ti'] = df['TI'].min()
    stats['mean_ti'] = df['TI'].mean()
    stats['median_ti'] = df['TI'].median()
    
    stats['protective_effects'] = len(df[df['TI'] < 1])
    stats['risk_effects'] = len(df[df['TI'] > 1])
    stats['neutral_effects'] = len(df[df['TI'] == 1])
    
    stats['significant_protective'] = len(df[df['IC95_max'] < 1])
    stats['significant_risk'] = len(df[df['IC95_min'] > 1])
    
    return stats

def main():
    # En-tête de l'application
    st.markdown('<h1 class="main-header">📊 Forest Plot Xeljanz - Analyse Interactive</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse des effets indésirables par groupe de traitement</p>', unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    
    # Sidebar pour les contrôles
    st.sidebar.header("🎛️ Paramètres d'analyse")
    
    # Sélection du groupe
    groupes_disponibles = ['Tous les groupes'] + list(df['Groupe'].unique())
    groupe_selectionne = st.sidebar.selectbox(
        "🏥 Sélectionner le groupe de traitement:",
        groupes_disponibles,
        index=0
    )
    
    # Sélection des effets indésirables
    effets_disponibles = ['Tous les effets'] + list(df['Effet indésirable'].unique())
    effets_selectionnes = st.sidebar.multiselect(
        "🔍 Filtrer par effets indésirables:",
        effets_disponibles,
        default=['Tous les effets']
    )
    
    # Filtrage des données
    df_filtered = df.copy()
    
    if groupe_selectionne != 'Tous les groupes':
        df_filtered = df_filtered[df_filtered['Groupe'] == groupe_selectionne]
    
    if 'Tous les effets' not in effets_selectionnes and effets_selectionnes:
        df_filtered = df_filtered[df_filtered['Effet indésirable'].isin(effets_selectionnes)]
    
    # Titre dynamique
    title_suffix = f" - {groupe_selectionne}" if groupe_selectionne != 'Tous les groupes' else ""
    
    # Layout principal en colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Forest Plot")
        
        # Créer et afficher le forest plot
        fig = create_forest_plot(df_filtered, title_suffix)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Bouton de téléchargement
            if st.button("💾 Télécharger le graphique (PNG)"):
                img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                st.download_button(
                    label="📁 Cliquez pour télécharger",
                    data=img_bytes,
                    file_name="forest_plot_xeljanz.png",
                    mime="image/png"
                )
    
    with col2:
        st.subheader("📊 Statistiques Descriptives")
        
        # Calculer les statistiques
        stats = calculate_stats(df_filtered)
        
        # Métriques principales
        st.metric("🔢 Nombre d'effets", stats['total_effects'])
        st.metric("📋 Total d'observations", stats['total_observations'])
        st.metric("👥 Groupes de traitement", stats['total_groups'])
        
        st.markdown("---")
        
        # Statistiques TI
        st.markdown("**📈 Taux d'Incidence (TI)**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Maximum", f"{stats['highest_ti']:.3f}")
            st.metric("Moyenne", f"{stats['mean_ti']:.3f}")
        with col_b:
            st.metric("Minimum", f"{stats['lowest_ti']:.3f}")
            st.metric("Médiane", f"{stats['median_ti']:.3f}")
        
        st.markdown("---")
        
        # Répartition des effets
        st.markdown("**🎯 Répartition des Effets**")
        
        st.markdown(f'<div class="risk-low">🛡️ Effets protecteurs (TI < 1): {stats["protective_effects"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-high">⚠️ Risques accrus (TI > 1): {stats["risk_effects"]}</div>', unsafe_allow_html=True)
        
        if stats['neutral_effects'] > 0:
            st.info(f"⚖️ Effets neutres (TI = 1): {stats['neutral_effects']}")
        
        st.markdown("---")
        
        # Effets significatifs
        st.markdown("**🎯 Effets Statistiquement Significatifs**")
        st.success(f"🛡️ Protecteurs significatifs: {stats['significant_protective']}")
        st.error(f"⚠️ Risques significatifs: {stats['significant_risk']}")
    
    # Tableau de données
    st.subheader("📋 Tableau de Données")
    
    # Options d'affichage du tableau
    col_table1, col_table2, col_table3 = st.columns(3)
    with col_table1:
        show_all_columns = st.checkbox("Afficher toutes les colonnes", value=True)
    with col_table2:
        highlight_significant = st.checkbox("Surligner les effets significatifs", value=True)
    with col_table3:
        sort_by_ti = st.checkbox("Trier par TI décroissant", value=False)
    
    # Préparation du tableau
    df_display = df_filtered.copy()
    
    if sort_by_ti:
        df_display = df_display.sort_values('TI', ascending=False)
    
    # Colonnes à afficher
    if show_all_columns:
        columns_to_show = df_display.columns.tolist()
    else:
        columns_to_show = ['Effet indésirable', 'Groupe', 'TI', 'IC95_min', 'IC95_max']
        df_display = df_display[columns_to_show]
    
    # Formatage du tableau
    if highlight_significant:
        def highlight_rows(row):
            if row['IC95_max'] < 1:
                return ['background-color: #d4edda'] * len(row)  # Vert pour protecteur
            elif row['IC95_min'] > 1:
                return ['background-color: #f8d7da'] * len(row)  # Rouge pour risque
            else:
                return [''] * len(row)
        
        styled_df = df_display.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df_display, use_container_width=True)
    
    # Bouton de téléchargement du tableau
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📁 Télécharger les données (CSV)",
        data=csv,
        file_name="xeljanz_data_filtered.csv",
        mime="text/csv"
    )
    
    # Analyse détaillée par groupe
    if groupe_selectionne == 'Tous les groupes':
        st.subheader("🔍 Analyse Comparative par Groupe")
        
        # Statistiques par groupe
        stats_by_group = df_filtered.groupby('Groupe').agg({
            'TI': ['count', 'mean', 'median', 'min', 'max', 'std'],
            'IC95_min': 'mean',
            'IC95_max': 'mean'
        }).round(3)
        
        stats_by_group.columns = ['Nb_Effets', 'TI_Moyen', 'TI_Médian', 'TI_Min', 'TI_Max', 'TI_Écart_type', 'IC95_min_Moyen', 'IC95_max_Moyen']
        
        st.dataframe(stats_by_group, use_container_width=True)
        
        # Top 3 des risques par groupe
        st.subheader("🏆 Top 3 des Risques les Plus Élevés par Groupe")
        
        for groupe in df_filtered['Groupe'].unique():
            data_groupe = df_filtered[df_filtered['Groupe'] == groupe]
            top_3 = data_groupe.nlargest(3, 'TI')
            
            st.markdown(f"**{groupe}:**")
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                st.markdown(f"   {idx}. {row['Effet indésirable']}: TI = {row['TI']:.3f} [{row['IC95_min']:.3f}-{row['IC95_max']:.3f}]")
            st.markdown("")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            💡 <em>Application développée avec Streamlit et Plotly</em><br>
            🔍 <em>Utilisez les filtres dans la barre latérale pour explorer les données</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()