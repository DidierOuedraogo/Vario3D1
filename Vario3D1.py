import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import math
import time
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="GeoMinApp - Analyse Variographique",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        padding: 10px 16px;
        background-color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    .stat-box {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .stat-box:hover {
        transform: translateY(-3px);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563eb;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
    }
    .filter-panel {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    .filter-tag {
        display: inline-flex;
        align-items: center;
        background-color: #3b82f6;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .help-text {
        font-size: 0.8rem;
        color: #64748b;
        font-style: italic;
    }
    .domain-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .color-square {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        margin-right: 8px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Titre et sous-titre
st.markdown('<div class="main-title">GeoMinApp - Analyse Variographique</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Outil d\'analyse géostatistique des données d\'exploration minière</div>', unsafe_allow_html=True)

# Initialisation de la session state si nécessaire
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'domains' not in st.session_state:
    st.session_state.domains = []
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = []
if 'variogram_data' not in st.session_state:
    st.session_state.variogram_data = None
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = 'spherical'

# Sidebar pour le chargement des données et les paramètres
with st.sidebar:
    st.header("Données & Paramètres")
    
    # Chargement de fichier
    uploaded_file = st.file_uploader("Charger un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Charger les données
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            data = pd.read_csv(string_data)
            st.session_state.data = data
            st.session_state.filtered_data = None
            st.success(f"Fichier chargé avec succès: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
    
    # Charger un exemple
    if st.button("Charger un exemple de données"):
        # Générer des données d'exemple
        np.random.seed(42)
        n_samples = 500
        x = np.random.uniform(0, 100, n_samples)
        y = np.random.uniform(0, 100, n_samples)
        z = np.random.uniform(0, 50, n_samples)
        
        # Générer des valeurs avec une tendance et une structure spatiale
        trend = 0.01 * x + 0.02 * y - 0.03 * z
        random_field = np.zeros(n_samples)
        
        # Simuler une structure spatiale simple
        for i in range(n_samples):
            for j in range(n_samples):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                if dist < 20:  # Range
                    correlation = 0.8 * (1 - 1.5*dist/20 + 0.5*(dist/20)**3)  # Modèle sphérique
                    random_field[i] += correlation * np.random.normal(0, 0.5)
        
        # Normaliser le champ aléatoire
        random_field = random_field / np.std(random_field)
        
        # Combiner tendance et champ aléatoire
        values = trend + random_field + np.random.normal(1, 0.2, n_samples)
        
        # Créer des domaines géologiques
        domains = np.array(['A'] * n_samples)
        domains[z > 25] = 'B'
        domains[x > 70] = 'C'
        domains[(x < 30) & (y < 30)] = 'D'
        
        # Créer le DataFrame
        example_data = pd.DataFrame({
            'X': x,
            'Y': y,
            'Z': z,
            'Valeur': values,
            'Domaine': domains
        })
        
        st.session_state.data = example_data
        st.session_state.filtered_data = None
        st.success("Données d'exemple chargées")
    
    if st.session_state.data is not None:
        # Configuration des colonnes
        st.subheader("Configuration des données")
        
        columns = st.session_state.data.columns.tolist()
        
        # Essayer de deviner les colonnes x, y, z, valeur et domaine
        default_x = next((col for col in columns if col.lower() in ['x', 'east', 'easting']), columns[0])
        default_y = next((col for col in columns if col.lower() in ['y', 'north', 'northing']), columns[1] if len(columns) > 1 else columns[0])
        default_z = next((col for col in columns if col.lower() in ['z', 'elev', 'elevation']), columns[2] if len(columns) > 2 else columns[0])
        default_val = next((col for col in columns if col.lower() in ['val', 'value', 'grade', 'teneur']), columns[3] if len(columns) > 3 else columns[0])
        default_domain = next((col for col in columns if col.lower() in ['domain', 'domaine', 'zone', 'litho', 'category', 'cat']), None)
        
        col_x = st.selectbox("Colonne X", columns, index=columns.index(default_x))
        col_y = st.selectbox("Colonne Y", columns, index=columns.index(default_y))
        col_z = st.selectbox("Colonne Z", columns, index=columns.index(default_z))
        col_val = st.selectbox("Colonne Valeur", columns, index=columns.index(default_val))
        
        domain_options = ["Aucune"] + columns
        col_domain = st.selectbox(
            "Colonne Domaine/Catégorie",
            domain_options,
            index=domain_options.index(default_domain) if default_domain in domain_options else 0
        )
        
        # Si un domaine est sélectionné, identifier les domaines uniques
        if col_domain != "Aucune":
            unique_domains = st.session_state.data[col_domain].unique().tolist()
            
            # Définir des couleurs pour chaque domaine
            color_scale = px.colors.qualitative.Plotly
            
            # Créer ou mettre à jour les domaines
            if len(st.session_state.domains) == 0 or len(st.session_state.domains) != len(unique_domains):
                st.session_state.domains = []
                for i, domain in enumerate(unique_domains):
                    st.session_state.domains.append({
                        'name': domain,
                        'color': color_scale[i % len(color_scale)],
                        'selected': True
                    })
            
            # Afficher les filtres de domaine
            st.subheader("Filtres par domaine")
            show_all_domains = st.checkbox("Afficher tous les domaines", True)
            
            # Afficher les domaines avec des cases à cocher
            for i, domain in enumerate(st.session_state.domains):
                col1, col2 = st.columns([1, 8])
                with col1:
                    domain['selected'] = st.checkbox("", domain['selected'], key=f"domain_{i}")
                with col2:
                    st.markdown(f"""
                    <div class="domain-item">
                        <div class="color-square" style="background-color: {domain['color']};"></div>
                        <span>{domain['name']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            if show_all_domains:
                for domain in st.session_state.domains:
                    domain['selected'] = True
        
        # Paramètres du variogramme
        st.subheader("Paramètres du variogramme")
        
        lag_distance = st.number_input(
            "Distance de lag (m)",
            min_value=0.1,
            value=5.0,
            step=0.1,
            help="Pas de distance pour le calcul du variogramme"
        )
        
        num_lags = st.number_input(
            "Nombre de lags",
            min_value=1,
            value=10,
            step=1,
            help="Nombre d'intervalles de distance"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            azimuth = st.number_input(
                "Azimut (degrés)",
                min_value=0,
                max_value=360,
                value=0,
                step=15,
                help="Angle horizontal (0 = Nord, 90 = Est)"
            )
        with col2:
            dip = st.number_input(
                "Plongement (degrés)",
                min_value=-90,
                max_value=90,
                value=0,
                step=15,
                help="Angle vertical (-90 = bas, 90 = haut)"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            tolerance = st.number_input(
                "Tolérance angulaire (degrés)",
                min_value=1,
                max_value=90,
                value=22.5,
                step=1,
                help="Tolérance pour les directions"
            )
        with col2:
            bandwidth = st.number_input(
                "Bande passante (m)",
                min_value=1,
                value=10,
                step=1,
                help="Largeur de la bande de recherche perpendiculaire à la direction"
            )
        
        # Direction prédéfinie
        direction_options = {
            "custom": "Personnalisée",
            "x": "Axe X",
            "y": "Axe Y",
            "z": "Axe Z",
            "xy45": "XY 45°",
            "xy-45": "XY -45°"
        }
        
        selected_direction = st.selectbox(
            "Direction prédéfinie",
            list(direction_options.keys()),
            format_func=lambda x: direction_options[x]
        )
        
        if selected_direction != "custom":
            if selected_direction == "x":
                azimuth, dip = 90, 0
            elif selected_direction == "y":
                azimuth, dip = 0, 0
            elif selected_direction == "z":
                azimuth, dip = 0, 90
            elif selected_direction == "xy45":
                azimuth, dip = 45, 0
            elif selected_direction == "xy-45":
                azimuth, dip = 315, 0
            
            st.info(f"Direction définie: Azimut = {azimuth}°, Plongement = {dip}°")
        
        # Bouton pour calculer le variogramme
        if st.button("Calculer le variogramme"):
            with st.spinner("Calcul du variogramme en cours..."):
                try:
                    # Déterminer quelles données utiliser
                    data_to_use = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data
                    
                    if col_domain != "Aucune":
                        # Filtrer par domaines sélectionnés
                        selected_domains = [d['name'] for d in st.session_state.domains if d['selected']]
                        if selected_domains:
                            data_to_use = data_to_use[data_to_use[col_domain].isin(selected_domains)]
                    
                    # Calculer le variogramme
                    variogram_data = calculate_experimental_variogram(
                        data_to_use, 
                        col_x, col_y, col_z, col_val, 
                        lag_distance, num_lags, 
                        azimuth, dip, 
                        tolerance, bandwidth
                    )
                    
                    st.session_state.variogram_data = variogram_data
                    st.success("Variogramme calculé avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors du calcul du variogramme: {str(e)}")

# Corps principal avec onglets
if st.session_state.data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Données 3D", "Filtres", "Variogramme", "Modélisation"])
    
    # Onglet Données 3D
    with tab1:
        # Filtres rapides
        with st.expander("Filtres rapides", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                min_value = st.number_input("Valeur minimale", value=float(st.session_state.data[col_val].min()))
            with col2:
                max_value = st.number_input("Valeur maximale", value=float(st.session_state.data[col_val].max()))
            with col3:
                st.write("&nbsp;")
                apply_quick_filter = st.button("Appliquer", use_container_width=True)
            
            if apply_quick_filter:
                # Réinitialiser les filtres existants
                st.session_state.active_filters = []
                
                # Ajouter les nouveaux filtres
                if min_value is not None:
                    min_filter = {
                        'id': int(time.time() * 1000),
                        'column': col_val,
                        'operator': 'gte',
                        'value': min_value,
                        'value2': None
                    }
                    st.session_state.active_filters.append(min_filter)
                
                if max_value is not None:
                    max_filter = {
                        'id': int(time.time() * 1000) + 1,
                        'column': col_val,
                        'operator': 'lte',
                        'value': max_value,
                        'value2': None
                    }
                    st.session_state.active_filters.append(max_filter)
                
                # Appliquer les filtres
                apply_filters()
                st.success("Filtres appliqués")
        
        # Graphique 3D
        if col_domain != "Aucune" and len(st.session_state.domains) > 0:
            # Créer une trace pour chaque domaine sélectionné
            fig = go.Figure()
            
            for domain in st.session_state.domains:
                if domain['selected']:
                    domain_data = st.session_state.data[st.session_state.data[col_domain] == domain['name']]
                    if len(domain_data) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=domain_data[col_x],
                            y=domain_data[col_y],
                            z=domain_data[col_z],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color=domain['color'],
                                opacity=0.8
                            ),
                            name=domain['name'],
                            hovertemplate=f"{col_x}: %{{x}}<br>{col_y}: %{{y}}<br>{col_z}: %{{z}}<br>{col_val}: %{{text}}<br>{col_domain}: {domain['name']}<extra></extra>",
                            text=domain_data[col_val]
                        ))
        else:
            # Une seule trace avec coloration selon les valeurs
            fig = go.Figure(data=[go.Scatter3d(
                x=st.session_state.data[col_x],
                y=st.session_state.data[col_y],
                z=st.session_state.data[col_z],
                mode='markers',
                marker=dict(
                    size=4,
                    color=st.session_state.data[col_val],
                    colorscale='Viridis',
                    colorbar=dict(title=col_val),
                    opacity=0.8
                ),
                hovertemplate=f"{col_x}: %{{x}}<br>{col_y}: %{{y}}<br>{col_z}: %{{z}}<br>{col_val}: %{{marker.color}}<extra></extra>"
            )])
        
        fig.update_layout(
            title="Visualisation 3D des données",
            scene=dict(
                xaxis_title=col_x,
                yaxis_title=col_y,
                zaxis_title=col_z,
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques
        st.subheader("Statistiques")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        data_to_show = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data
        values = data_to_show[col_val].dropna()
        
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{len(values)}</div>
                <div class="stat-label">Échantillons</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{values.min():.3f}</div>
                <div class="stat-label">Minimum</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{values.max():.3f}</div>
                <div class="stat-label">Maximum</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{values.mean():.3f}</div>
                <div class="stat-label">Moyenne</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{values.var():.3f}</div>
                <div class="stat-label">Variance</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Onglet Filtres
    with tab2:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.subheader("Filtres avancés")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_column = st.selectbox("Colonne", st.session_state.data.columns.tolist())
            
            # Déterminer si la colonne est numérique ou catégorielle
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.data[filter_column])
            
            if is_numeric:
                filter_operator = st.selectbox(
                    "Opérateur", 
                    ["Égal à", "Différent de", "Supérieur à", "Supérieur ou égal à", "Inférieur à", "Inférieur ou égal à", "Entre"]
                )
            else:
                filter_operator = st.selectbox(
                    "Opérateur", 
                    ["Égal à", "Différent de", "Contient"]
                )
        
        with col2:
            filter_value = st.text_input("Valeur")
            
            if filter_operator == "Entre":
                filter_value2 = st.text_input("Valeur 2")
            else:
                filter_value2 = None
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ajouter filtre", use_container_width=True):
                if not filter_value or (filter_operator == "Entre" and not filter_value2):
                    st.error("Veuillez remplir tous les champs du filtre")
                else:
                    # Convertir l'opérateur en code
                    op_mapping = {
                        "Égal à": "eq",
                        "Différent de": "neq",
                        "Supérieur à": "gt",
                        "Supérieur ou égal à": "gte",
                        "Inférieur à": "lt",
                        "Inférieur ou égal à": "lte",
                        "Entre": "between",
                        "Contient": "contains"
                    }
                    
                    # Créer le filtre
                    new_filter = {
                        'id': int(time.time() * 1000),
                        'column': filter_column,
                        'operator': op_mapping[filter_operator],
                        'value': filter_value,
                        'value2': filter_value2 if filter_operator == "Entre" else None
                    }
                    
                    # Ajouter à la liste
                    st.session_state.active_filters.append(new_filter)
                    st.success(f"Filtre ajouté: {get_filter_description(new_filter)}")
        
        with col2:
            if st.button("Appliquer tous les filtres", use_container_width=True):
                if len(st.session_state.active_filters) == 0:
                    st.warning("Aucun filtre à appliquer")
                else:
                    apply_filters()
                    st.success(f"{len(st.session_state.active_filters)} filtres appliqués")
        
        # Afficher les filtres actifs
        if len(st.session_state.active_filters) > 0:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Filtres actifs")
            
            for i, filter_item in enumerate(st.session_state.active_filters):
                st.markdown(f"""
                <div class="filter-tag">
                    {get_filter_description(filter_item)}
                </div>
                """, unsafe_allow_html=True)
                
            if st.button("Effacer tous les filtres"):
                st.session_state.active_filters = []
                st.session_state.filtered_data = None
                st.success("Tous les filtres ont été effacés")
        else:
            st.info("Aucun filtre actif")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Afficher les données filtrées
        if st.session_state.filtered_data is not None:
            st.subheader(f"Données filtrées ({len(st.session_state.filtered_data)} points)")
            st.dataframe(st.session_state.filtered_data.head(100))
            
            # Graphique des données filtrées
            fig = go.Figure(data=[go.Scatter3d(
                x=st.session_state.filtered_data[col_x],
                y=st.session_state.filtered_data[col_y],
                z=st.session_state.filtered_data[col_z],
                mode='markers',
                marker=dict(
                    size=4,
                    color=st.session_state.filtered_data[col_val],
                    colorscale='Viridis',
                    colorbar=dict(title=col_val),
                    opacity=0.8
                ),
                hovertemplate=f"{col_x}: %{{x}}<br>{col_y}: %{{y}}<br>{col_z}: %{{z}}<br>{col_val}: %{{marker.color}}<extra></extra>"
            )])
            
            fig.update_layout(
                title="Visualisation 3D des données filtrées",
                scene=dict(
                    xaxis_title=col_x,
                    yaxis_title=col_y,
                    zaxis_title=col_z,
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Onglet Variogramme
    with tab3:
        if st.session_state.variogram_data is None:
            st.info("Calculez d'abord un variogramme en utilisant le bouton dans le panneau latéral.")
        else:
            # Options de normalisation
            normalize_options = {
                "none": "Non normalisé",
                "total": "Normalisé / variance",
                "sill": "Normalisé / palier"
            }
            
            normalize_type = st.selectbox(
                "Type de normalisation",
                list(normalize_options.keys()),
                format_func=lambda x: normalize_options[x]
            )
            
            # Tracer le variogramme expérimental
            fig = plot_variogram(st.session_state.variogram_data, normalize_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les détails
            st.subheader("Détails du variogramme")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Paramètres directionnels:**
                - Azimut: {st.session_state.variogram_data['direction']['azimuth']}°
                - Plongement: {st.session_state.variogram_data['direction']['dip']}°
                - Tolérance angulaire: {st.session_state.variogram_data['lagParams']['tolerance']}°
                - Bande passante: {st.session_state.variogram_data['lagParams']['bandwidth']} m
                """)
            with col2:
                st.markdown(f"""
                **Paramètres de distance:**
                - Distance de lag: {st.session_state.variogram_data['lagParams']['lagDistance']} m
                - Nombre de lags: {st.session_state.variogram_data['lagParams']['numLags']}
                - Variance totale: {st.session_state.variogram_data['totalVariance']:.3f}
                """)
            
            # Tableau des valeurs du variogramme
            variogram_df = pd.DataFrame({
                'Distance': st.session_state.variogram_data['distances'],
                'Semi-variance': st.session_state.variogram_data['variances'],
                'Paires': st.session_state.variogram_data['pairs']
            })
            
            st.dataframe(variogram_df)
    
    # Onglet Modélisation
    with tab4:
        if st.session_state.variogram_data is None:
            st.info("Calculez d'abord un variogramme en utilisant le bouton dans le panneau latéral.")
        else:
            # Modèle de variogramme
            model_types = {
                "spherical": "Sphérique",
                "exponential": "Exponentiel",
                "gaussian": "Gaussien",
                "power": "Puissance"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                current_model_type = st.selectbox(
                    "Type de modèle",
                    list(model_types.keys()),
                    format_func=lambda x: model_types[x],
                    index=list(model_types.keys()).index(st.session_state.current_model_type)
                )
                st.session_state.current_model_type = current_model_type
            
            with col2:
                normalize_type = st.selectbox(
                    "Type de normalisation",
                    list(normalize_options.keys()),
                    format_func=lambda x: normalize_options[x],
                    key="model_normalize"
                )
            
            # Paramètres du modèle
            total_variance = st.session_state.variogram_data['totalVariance']
            
            col1, col2 = st.columns(2)
            with col1:
                nugget_pct = st.slider(
                    "Effet pépite (%)",
                    min_value=0,
                    max_value=100,
                    value=10,
                    step=1
                )
                
                sill_pct = st.slider(
                    "Palier (%)",
                    min_value=0,
                    max_value=100,
                    value=90,
                    step=1
                )
                
                if total_variance > 0:
                    nugget = nugget_pct / 100 * total_variance
                    sill = sill_pct / 100 * total_variance
                else:
                    nugget = nugget_pct / 100
                    sill = sill_pct / 100
            
            with col2:
                # Calcul automatique d'une portée initiale
                max_dist = max(st.session_state.variogram_data['distances'])
                default_range = max_dist / 3
                
                range_value = st.slider(
                    "Portée",
                    min_value=1.0,
                    max_value=float(max_dist * 1.5),
                    value=float(default_range),
                    step=1.0
                )
                
                if current_model_type == "power":
                    power_value = st.slider(
                        "Exposant",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.5,
                        step=0.1
                    )
                else:
                    power_value = 1.5
            
            # Tracer le modèle
            fig = plot_variogram_model(
                st.session_state.variogram_data,
                current_model_type,
                nugget,
                sill,
                range_value,
                power_value,
                normalize_type
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Résumé du modèle
            st.subheader("Paramètres du modèle")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Type de modèle:** {model_types[current_model_type]}
                
                **Effet pépite:** {nugget:.3f} ({nugget_pct}%)
                """)
            
            with col2:
                st.markdown(f"""
                **Palier:** {sill:.3f} ({sill_pct}%)
                
                **Portée:** {range_value:.2f} m
                """)
            
            with col3:
                if current_model_type == "power":
                    st.markdown(f"""
                    **Exposant:** {power_value}
                    
                    **Variance totale:** {total_variance:.3f}
                    """)
                else:
                    st.markdown(f"""
                    **Palier total:** {nugget + sill:.3f}
                    
                    **Variance totale:** {total_variance:.3f}
                    """)
            
            # Formule du modèle
            st.subheader("Formule du modèle")
            
            if current_model_type == "spherical":
                model_formula = f"γ(h) = {nugget:.3f} + {sill:.3f} × [1.5(h/{range_value:.1f}) - 0.5(h/{range_value:.1f})³] pour h < {range_value:.1f}\nγ(h) = {nugget + sill:.3f} pour h ≥ {range_value:.1f}"
            elif current_model_type == "exponential":
                model_formula = f"γ(h) = {nugget:.3f} + {sill:.3f} × [1 - exp(-3h/{range_value:.1f})]"
            elif current_model_type == "gaussian":
                model_formula = f"γ(h) = {nugget:.3f} + {sill:.3f} × [1 - exp(-3(h/{range_value:.1f})²)]"
            elif current_model_type == "power":
                model_formula = f"γ(h) = {nugget:.3f} + {sill:.3f} × (h/{range_value:.1f})^{power_value}"
            
            st.code(model_formula)
            
            # Exporter le modèle
            if st.button("Exporter le modèle"):
                model_data = {
                    "type": current_model_type,
                    "nugget": nugget,
                    "sill": sill,
                    "range": range_value,
                    "power": power_value if current_model_type == "power" else None,
                    "variance": total_variance,
                    "direction": {
                        "azimuth": st.session_state.variogram_data['direction']['azimuth'],
                        "dip": st.session_state.variogram_data['direction']['dip']
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                model_df = pd.DataFrame([model_data])
                csv = model_df.to_csv(index=False)
                
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name=f"variogram_model_{current_model_type}.csv",
                    mime="text/csv"
                )

# Fonctions utilitaires
def apply_filters():
    """Applique les filtres actifs aux données."""
    if len(st.session_state.active_filters) == 0:
        st.session_state.filtered_data = None
        return
    
    # Appliquer les filtres
    filtered_data = st.session_state.data.copy()
    
    for filter_item in st.session_state.active_filters:
        column = filter_item['column']
        operator = filter_item['operator']
        value = filter_item['value']
        value2 = filter_item['value2']
        
        # Convertir les valeurs si la colonne est numérique
        is_numeric = pd.api.types.is_numeric_dtype(filtered_data[column])
        if is_numeric and not pd.isna(value):
            value = float(value)
            if value2 and not pd.isna(value2):
                value2 = float(value2)
        
        # Appliquer l'opérateur
        if operator == 'eq':
            filtered_data = filtered_data[filtered_data[column] == value]
        elif operator == 'neq':
            filtered_data = filtered_data[filtered_data[column] != value]
        elif operator == 'gt':
            filtered_data = filtered_data[filtered_data[column] > value]
        elif operator == 'gte':
            filtered_data = filtered_data[filtered_data[column] >= value]
        elif operator == 'lt':
            filtered_data = filtered_data[filtered_data[column] < value]
        elif operator == 'lte':
            filtered_data = filtered_data[filtered_data[column] <= value]
        elif operator == 'between':
            filtered_data = filtered_data[(filtered_data[column] >= value) & (filtered_data[column] <= value2)]
        elif operator == 'contains':
            filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(str(value), case=False)]
    
    st.session_state.filtered_data = filtered_data

def get_filter_description(filter_item):
    """Renvoie une description textuelle d'un filtre."""
    operator_map = {
        'eq': '=',
        'neq': '≠',
        'gt': '>',
        'gte': '≥',
        'lt': '<',
        'lte': '≤',
        'between': 'entre',
        'contains': 'contient'
    }
    
    op_text = operator_map.get(filter_item['operator'], filter_item['operator'])
    
    if filter_item['operator'] == 'between':
        return f"{filter_item['column']} {op_text} {filter_item['value']} et {filter_item['value2']}"
    else:
        return f"{filter_item['column']} {op_text} {filter_item['value']}"

def calculate_experimental_variogram(data, x_col, y_col, z_col, value_col, lag_distance, num_lags, azimuth, dip, tolerance, bandwidth):
    """Calcule le variogramme expérimental 3D."""
    # Conversion des angles en radians
    azimuth_rad = (azimuth * math.pi) / 180
    dip_rad = (dip * math.pi) / 180
    tolerance_rad = (tolerance * math.pi) / 180
    
    # Calcul des vecteurs de direction
    dx = math.cos(azimuth_rad) * math.cos(dip_rad)
    dy = math.sin(azimuth_rad) * math.cos(dip_rad)
    dz = math.sin(dip_rad)
    
    # Structure pour stocker les résultats
    variogram = {
        'distances': [],
        'variances': [],
        'pairs': [],
        'direction': {'azimuth': azimuth, 'dip': dip},
        'lagParams': {'lagDistance': lag_distance, 'numLags': num_lags, 'tolerance': tolerance, 'bandwidth': bandwidth}
    }
    
    # Initialisation des compteurs pour chaque lag
    for lag in range(num_lags):
        variogram['distances'].append((lag + 0.5) * lag_distance)
        variogram['variances'].append(0)
        variogram['pairs'].append(0)
    
    # Extraire les colonnes nécessaires
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    z = data[z_col].to_numpy()
    values = data[value_col].to_numpy()
    
    # Calcul de la variance totale
    mean_value = np.mean(values)
    total_variance = np.var(values)
    
    # Parcourir toutes les paires de points
    n = len(data)
    for i in range(n):
        for j in range(i+1, n):
            # Calcul du vecteur entre les points
            hx = x[j] - x[i]
            hy = y[j] - y[i]
            hz = z[j] - z[i]
            
            # Longueur du vecteur (distance euclidienne)
            distance = math.sqrt(hx*hx + hy*hy + hz*hz)
            
            # Projection sur la direction
            proj = hx*dx + hy*dy + hz*dz
            proj_distance = abs(proj)
            
            # Calcul de l'angle entre le vecteur et la direction
            dot_product = (hx*dx + hy*dy + hz*dz) / distance if distance > 0 else 0
            angle = math.acos(min(max(dot_product, -1), 1))
            
            # Vérifier si la paire est dans la direction et la tolérance
            if angle <= tolerance_rad:
                # Calcul de la distance perpendiculaire (bande passante)
                perp_distance = math.sqrt(distance*distance - proj*proj) if abs(proj) < distance else 0
                
                if perp_distance <= bandwidth:
                    # Déterminer le lag
                    lag_index = int(proj_distance / lag_distance)
                    
                    if lag_index < num_lags:
                        # Calcul de la semi-variance
                        value1 = values[i]
                        value2 = values[j]
                        semi_variance = (value1 - value2)**2 / 2
                        
                        # Ajouter au variogramme
                        variogram['variances'][lag_index] += semi_variance
                        variogram['pairs'][lag_index] += 1
    
    # Calcul des moyennes
    for i in range(num_lags):
        if variogram['pairs'][i] > 0:
            variogram['variances'][i] /= variogram['pairs'][i]
        else:
            variogram['variances'][i] = float('nan')
    
    # Ajouter la variance totale
    variogram['totalVariance'] = total_variance
    
    return variogram

def plot_variogram(variogram_data, normalize_type="none"):
    """Trace le variogramme expérimental."""
    # Filtrer les points sans paires
    valid_indices = [i for i, p in enumerate(variogram_data['pairs']) if p > 0]
    distances = [variogram_data['distances'][i] for i in valid_indices]
    variances = [variogram_data['variances'][i] for i in valid_indices]
    pairs = [variogram_data['pairs'][i] for i in valid_indices]
    
    # Normalisation éventuelle
    y_values = variances.copy()
    y_axis_title = 'Semi-variance'
    y_max = variogram_data['totalVariance'] * 1.2
    total_variance_value = variogram_data['totalVariance']
    
    if normalize_type == 'total':
        y_values = [v / variogram_data['totalVariance'] for v in variances]
        y_axis_title = 'Semi-variance normalisée (γ/σ²)'
        y_max = 1.2
        total_variance_value = 1
    elif normalize_type == 'sill' and variances:
        max_variance = max(variances)
        y_values = [v / max_variance for v in variances]
        y_axis_title = 'Semi-variance normalisée (γ/C)'
        y_max = 1.2
        total_variance_value = 1
    
    # Créer la figure
    fig = go.Figure()
    
    # Trace pour le variogramme
    fig.add_trace(go.Scatter(
        x=distances,
        y=y_values,
        mode='markers+lines',
        marker=dict(
            size=[max(5, min(15, math.sqrt(p))) for p in pairs],
            color=pairs,
            colorscale='Viridis',
            colorbar=dict(title='Nombre de paires')
        ),
        line=dict(dash='dot', width=1),
        name='Variogramme expérimental'
    ))
    
    # Ligne horizontale pour la variance totale
    fig.add_trace(go.Scatter(
        x=[0, max(distances) * 1.1],
        y=[total_variance_value, total_variance_value],
        mode='lines',
        line=dict(dash='dash', width=1, color='red'),
        name='Variance totale'
    ))
    
    # Mise en page
    fig.update_layout(
        title=f"Variogramme Expérimental (Azimut: {variogram_data['direction']['azimuth']}°, Plongement: {variogram_data['direction']['dip']}°)",
        xaxis=dict(
            title='Distance (m)',
            range=[0, max(distances) * 1.1]
        ),
        yaxis=dict(
            title=y_axis_title,
            range=[0, y_max]
        ),
        legend=dict(x=0.7, y=0.1),
        height=500
    )
    
    return fig

def variogram_model(distance, sill, range_val, model_type, power=1.5):
    """Calcule la valeur du modèle de variogramme à une distance donnée."""
    if distance == 0:
        return 0
    
    if model_type == 'spherical':
        if distance >= range_val:
            return sill
        return sill * (1.5 * (distance / range_val) - 0.5 * (distance / range_val)**3)
    
    elif model_type == 'exponential':
        return sill * (1 - math.exp(-3 * distance / range_val))
    
    elif model_type == 'gaussian':
        return sill * (1 - math.exp(-3 * (distance / range_val)**2))
    
    elif model_type == 'power':
        return sill * (distance / range_val)**power
    
    return 0

def plot_variogram_model(variogram_data, model_type, nugget, sill, range_val, power, normalize_type="none"):
    """Trace le variogramme expérimental avec un modèle théorique."""
    # Filtrer les points sans paires
    valid_indices = [i for i, p in enumerate(variogram_data['pairs']) if p > 0]
    distances = [variogram_data['distances'][i] for i in valid_indices]
    variances = [variogram_data['variances'][i] for i in valid_indices]
    pairs = [variogram_data['pairs'][i] for i in valid_indices]
    
    # Normalisation éventuelle
    y_values = variances.copy()
    y_axis_title = 'Semi-variance'
    y_max = variogram_data['totalVariance'] * 1.2
    total_variance_value = variogram_data['totalVariance']
    
    if normalize_type == 'total':
        y_values = [v / variogram_data['totalVariance'] for v in variances]
        y_axis_title = 'Semi-variance normalisée (γ/σ²)'
        y_max = 1.2
        total_variance_value = 1
        nugget = nugget / variogram_data['totalVariance']
        sill = sill / variogram_data['totalVariance']
    elif normalize_type == 'sill' and variances:
        max_variance = max(variances)
        y_values = [v / max_variance for v in variances]
        y_axis_title = 'Semi-variance normalisée (γ/C)'
        y_max = 1.2
        total_variance_value = 1
        nugget = nugget / max_variance
        sill = sill / max_variance
    
    # Créer la figure
    fig = go.Figure()
    
    # Trace pour le variogramme expérimental
    fig.add_trace(go.Scatter(
        x=distances,
        y=y_values,
        mode='markers',
        marker=dict(
            size=[max(5, min(15, math.sqrt(p))) for p in pairs],
            color=pairs,
            colorscale='Viridis',
            colorbar=dict(title='Nombre de paires')
        ),
        name='Variogramme expérimental'
    ))
    
    # Générer des points pour le modèle
    max_distance = max(distances) * 1.1
    model_x = np.linspace(0, max_distance, 100)
    model_y = [nugget + variogram_model(d, sill, range_val, model_type, power) for d in model_x]
    
    # Trace pour le modèle
    fig.add_trace(go.Scatter(
        x=model_x,
        y=model_y,
        mode='lines',
        line=dict(width=2, color='rgba(255, 0, 0, 0.7)'),
        name=f'Modèle {model_type}'
    ))
    
    # Ligne horizontale pour la variance totale
    fig.add_trace(go.Scatter(
        x=[0, max_distance],
        y=[total_variance_value, total_variance_value],
        mode='lines',
        line=dict(dash='dash', width=1, color='gray'),
        name='Variance totale'
    ))
    
    # Mise en page
    fig.update_layout(
        title='Modélisation du Variogramme',
        xaxis=dict(
            title='Distance (m)',
            range=[0, max_distance]
        ),
        yaxis=dict(
            title=y_axis_title,
            range=[0, y_max]
        ),
        legend=dict(x=0.7, y=0.1),
        height=500
    )
    
    return fig

# Pied de page
st.markdown("""
<div class="footer">
    <p>GeoMinApp - Développé par Didier Ouedraogo, P.Geo</p>
    <p>© 2025 - Analyse variographique pour les données d'exploration minière</p>
</div>
""", unsafe_allow_html=True)