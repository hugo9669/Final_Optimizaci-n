import streamlit as st
import osmnx as ox
import networkx as nx
import pulp
import numpy as np
import pandas as pd
import folium
from streamlit_folium import folium_static
import random

# --- 1. CONFIGURACIÓN INICIAL Y DATOS FICTICIOS ---

# Configuración del área de estudio (Bello, Antioquia - Mismo lugar que las imágenes)
LATITUDE_CENTRAL = 6.333
LONGITUDE_CENTRAL = -75.568
DISTANCIA_KM = 0.5 # Radio para un área de ~1 km²

# Tipos de Flujo (Ambulancias) y sus costos operativos
TIPOS_AMBULANCIA = {
    'Leve': {'costo_operativo': 100, 'nombre': 'Ambulancia de Transporte Simple', 'color': 'green', 'priority': 1},
    'Media': {'costo_operativo': 250, 'nombre': 'Ambulancia de Cuidados Intermedios', 'color': 'orange', 'priority': 2},
    'Critica': {'costo_operativo': 500, 'nombre': 'Ambulancia de Cuidados Críticos', 'color': 'red', 'priority': 3},
}

# --- 2. FUNCIONES DE GENERACIÓN DE DATOS ---

@st.cache_data(show_spinner="Descargando y preparando el grafo vial con OSMnx...")
def cargar_grafo_inicial():
    """Descarga el grafo vial de OSMnx y calcula el tiempo de viaje."""
    
    # Descargar el grafo vial como MultiDiGraph
    G = ox.graph_from_point(
        (LATITUDE_CENTRAL, LONGITUDE_CENTRAL), 
        dist=DISTANCIA_KM * 1000, # Convertir a metros
        network_type="drive", 
        retain_all=True
    )

    # Proyectar el grafo
    G = ox.project_graph(G)

    # Convertir a MultiDiGraph (Asegura las claves)
    G = G.to_undirected().to_directed() 
    
    # Calcular el tiempo de viaje (usando 'length' y velocidad por defecto)
    G = ox.add_edge_speeds(G)
    G = ox.add_travel_times(G)

    # Identificar el nodo central más cercano a la base
    LUGAR_CENTRAL = ox.nearest_nodes(G, LONGITUDE_CENTRAL, LATITUDE_CENTRAL)
    
    return G, LUGAR_CENTRAL

def generar_capacidades_y_costos(G, C_MIN, C_MAX):
    """Genera capacidades viales aleatorias y costos de tiempo actualizados."""
    
    # Usamos una copia del grafo para no interferir con el original
    H = G.copy()
    
    for u, v, k, data in H.edges(keys=True, data=True):
        # Generar capacidad vial aleatoria (C_min a C_max)
        capacidad_vial_kmh = random.uniform(C_MIN, C_MAX)
        data['capacidad_vial_kmh'] = capacidad_vial_kmh
        
        # Calcular el tiempo de viaje (costo) en minutos
        # Longitud en metros / (Capacidad en km/h * 1000/60)
        # Nota: Usamos la capacidad simulada como velocidad efectiva
        length_m = data.get('length', 1) 
        if capacidad_vial_kmh > 0:
            # Tiempo de viaje en minutos
            costo_tiempo_min = (length_m / 1000) / (capacidad_vial_kmh / 60)
        else:
            costo_tiempo_min = float('inf')
        
        data['costo_tiempo_min'] = costo_tiempo_min

    return H

def generar_flujos_emergencia(G, LUGAR_CENTRAL, NUM_EMERGENCIAS, R_MIN, R_MAX):
    """Genera flujos de emergencia con origen, destino y velocidad requerida."""
    
    nodos_destino = random.sample(list(G.nodes), NUM_EMERGENCIAS)
    flujos_k = []

    for i, d_k in enumerate(nodos_destino):
        tipo = random.choice(list(TIPOS_AMBULANCIA.keys()))
        flujo_data = TIPOS_AMBULANCIA[tipo]
        
        # Velocidad Requerida (R_i)
        R_k = random.uniform(R_MIN, R_MAX)
        
        flujos_k.append({
            'id': f"Flujo_{i+1}",
            'tipo_urgencia': tipo,
            'origen': LUGAR_CENTRAL,
            'destino': d_k,
            'velocidad_requerida_kmh': R_k,
            'costo_operativo': flujo_data['costo_operativo'],
            'color': flujo_data['color']
        })
        
    return flujos_k


# --- 3. MODELO DE OPTIMIZACIÓN MULTIFLUJO (PU L P) ---

def resolver_modelo_multifluido(G, flujos_k):
    """Implementa y resuelve el modelo de Flujo Multicommodity con Big M."""

    A_keys = list(G.edges(keys=True)) 
    modelo = pulp.LpProblem("Enrutamiento_Ambulancias_Multiflujo", pulp.LpMinimize)

    # --- Variables ---
    variables_x = pulp.LpVariable.dicts("Ruta",
                                       ((u, v, k, flujo['id']) for u, v, k in A_keys for flujo in flujos_k),
                                       lowBound=0, upBound=1, cat=pulp.LpBinary)

    variables_z = pulp.LpVariable.dicts("Violacion_Velocidad",
                                        ((u,v,k, flujo['id']) for u,v,k in A_keys for flujo in flujos_k),
                                        lowBound=0, upBound=1, cat=pulp.LpBinary)

    BIG_M = 1000000 # Costo de penalización alto
    costo_total_viaje = []
    costo_operativo_fijo = 0
    penalizacion_total = []

    # --- Restricciones de Enlace (Big M) y Función Objetivo ---
    for flujo in flujos_k:
        flujo_id = flujo['id']
        costo_operativo_fijo += flujo['costo_operativo']
        
        for u, v, k in A_keys:
            edge_data = G.edges[(u, v, k)]
            capacidad_vial_kmh = edge_data.get('capacidad_vial_kmh', 0)
            costo_tiempo_min = edge_data.get('costo_tiempo_min', 0)
            R_k = flujo['velocidad_requerida_kmh']

            # 1. Costo normal (minutos de viaje)
            costo_total_viaje.append(costo_tiempo_min * variables_x[(u, v, k, flujo_id)])
            
            # 2. Penalización (Costo Big M)
            penalizacion_total.append(BIG_M * variables_z[(u, v, k, flujo_id)])
            
            # 3. Restricciones de Enlace (Lógica Big M)
            if R_k > capacidad_vial_kmh:
                # CASO A: Arista INAPROPIADA (Rk > μij). Forzar Z >= X
                # Si x_ijk = 1, z_ijk debe ser 1 (penalizado)
                modelo += variables_z[(u, v, k, flujo_id)] >= variables_x[(u, v, k, flujo_id)], \
                           f"Penal_Req_Enlace_{u}_{v}_{k}_{flujo_id}"
            else:
                # CASO B: Arista APROPIADA (Rk <= μij). Forzar Z = 0
                # Si la arista cumple, no hay penalización
                modelo += variables_z[(u, v, k, flujo_id)] == 0, f"Penal_Req_CUMPLIDA_{u}_{v}_{k}_{flujo_id}"


    # --- Función Objetivo Final ---
    modelo += pulp.lpSum(costo_total_viaje) + pulp.lpSum(penalizacion_total) + costo_operativo_fijo, "Costo_Total_Minimo"

    # --- Restricciones de Conservación de Flujo ---
    for flujo in flujos_k:
        flujo_id = flujo['id']
        s = flujo['origen']
        d_k = flujo['destino']

        for i in G.nodes:
            # Flujo saliente: (i, j, k, flujo_id)
            flujo_saliente = [variables_x.get((i, j, m, flujo_id), 0) 
                              for j in G.neighbors(i) 
                              for m in G.get_edge_data(i, j, default={}) 
                              if (i, j, m, flujo_id) in variables_x]

            # Flujo entrante: (j, i, k, flujo_id)
            flujo_entrante = [variables_x.get((j, i, m, flujo_id), 0) 
                              for j in G.predecessors(i) 
                              for m in G.get_edge_data(j, i, default={}) 
                              if (j, i, m, flujo_id) in variables_x]

            balance = pulp.lpSum(flujo_saliente) - pulp.lpSum(flujo_entrante)

            rhs = 0
            if i == s:
                rhs = 1
            elif i == d_k:
                rhs = -1

            modelo += balance == rhs, f"Continuidad_{i}_{flujo_id}"
            
    # --- Solución ---
    try:
        modelo.solve(pulp.PULP_CBC_CMD(msg=0)) # Silenciar el output del solver
        
        if modelo.status == 1: # Optimal
            rutas_optimas = {}
            total_tiempo_viaje = pulp.value(pulp.lpSum(costo_total_viaje))
            total_penalizacion = pulp.value(pulp.lpSum(penalizacion_total))

            # Extracción de Rutas
            for u, v, k, flujo_id in variables_x: 
                if variables_x[(u, v, k, flujo_id)].varValue is not None and variables_x[(u, v, k, flujo_id)].varValue > 0.9:
                    if flujo_id not in rutas_optimas:
                        rutas_optimas[flujo_id] = []
                    rutas_optimas[flujo_id].append((u, v, k))
            
            return rutas_optimas, modelo.status, pulp.value(modelo.objective), total_tiempo_viaje, total_penalizacion
        
        else:
            return {}, modelo.status, None, None, None
            
    except Exception as e:
        st.error(f"Error grave al resolver el modelo: {e}")
        return {}, -2, None, None, None


# --- 4. FUNCIONES DE VISUALIZACIÓN ---

def dibujar_rutas_en_mapa(G, flujos_data, rutas_optimas):
    """Dibuja el grafo, el origen, los destinos y las rutas óptimas con Folium."""
    
    # Obtener coordenadas del centro para inicializar el mapa
    center_lat = G.nodes[flujos_data[0]['origen']]['y']
    center_lon = G.nodes[flujos_data[0]['origen']]['x']
    
    # Crear el mapa de Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")
    
    # Dibujar todas las calles del grafo (fondo)
    ox.plot_graph_folium(G, graph_map=m, edge_width=0.5, edge_color='#CCCCCC', node_size=0, popup_attribute=None)
    
    # Diccionario para almacenar información de la ruta
    ruta_info = {}

    for flujo in flujos_data:
        flujo_id = flujo['id']
        
        if flujo_id in rutas_optimas:
            ruta = rutas_optimas[flujo_id]
            s = flujo['origen']
            d_k = flujo['destino']
            R_k = flujo['velocidad_requerida_kmh']
            tipo_urgencia = flujo['tipo_urgencia']
            flujo_color = flujo['color']
            
            # --- 4.1 Dibujar la Ruta Óptima ---
            for u, v, k in ruta:
                
                # Obtener la data de la arista
                edge_data = G.edges[(u, v, k)]
                μ_ij = edge_data['capacidad_vial_kmh']
                costo_min = edge_data['costo_tiempo_min']
                
                # Coordenadas de los nodos
                u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
                v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']

                # Verificar si se violó la restricción de velocidad
                violacion = "❌ Violada (Rk > μij)" if R_k > μ_ij else "✅ Cumplida"

                # Información para el popup
                popup_html = f"""
                <b>Flujo:</b> {flujo_id} ({tipo_urgencia})<br>
                <b>Requerida (Rk):</b> {R_k:.1f} km/h<br>
                <b>Capacidad (μij):</b> {μ_ij:.1f} km/h<br>
                <b>Restricción:</b> {violacion}<br>
                <b>Costo:</b> {costo_min:.2f} min
                """
                
                # Dibujar la arista
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color=flujo_color,
                    weight=4,
                    opacity=0.8,
                    tooltip=flujo_id,
                ).add_child(folium.Popup(popup_html)).add_to(m)

                # Almacenar info detallada
                if flujo_id not in ruta_info: ruta_info[flujo_id] = {'tiempo_total': 0, 'aristas_violadas': 0, 'tipo': tipo_urgencia, 'Rk': R_k, 'ruta': []}
                
                ruta_info[flujo_id]['tiempo_total'] += costo_min
                if R_k > μ_ij:
                    ruta_info[flujo_id]['aristas_violadas'] += 1

            
            # --- 4.2 Marcador de Origen (Base) ---
            folium.Marker(
                location=[G.nodes[s]['y'], G.nodes[s]['x']],
                popup=f"BASE: {s} - Centro de Ambulancias",
                icon=folium.Icon(color='blue', icon='fa-ambulance', prefix='fa')
            ).add_to(m)
            
            # --- 4.3 Marcador de Destino (Incidente) ---
            folium.Marker(
                location=[G.nodes[d_k]['y'], G.nodes[d_k]['x']],
                popup=f"DESTINO: {d_k} - Emergencia {flujo_id} ({tipo_urgencia})",
                icon=folium.Icon(color=flujo_color, icon='fa-hospital', prefix='fa')
            ).add_to(m)
            
    # Mostrar el mapa en Streamlit
    folium_static(m)
    
    return ruta_info

# --- 5. INTERFAZ DE STREAMLIT ---

def main():
    st.set_page_config(layout="wide", page_title="Modelo de Enrutamiento Multiflujo para Ambulancias")

    st.title("🚑 Modelo de Enrutamiento Multiflujo (Ambulancias)")
    st.caption("Implementación de Flujo Multicommodity con PuLP y OSMnx.")
    
    # 5.1 CARGAR GRAFO Y MANEJAR ESTADO INICIAL
    G_base, LUGAR_CENTRAL = cargar_grafo_inicial()

    # Inicializar el estado de la aplicación
    if 'G_actual' not in st.session_state:
        
        # 1. Asignar parámetros por defecto y generar capacidades iniciales
        R_MIN_DEFAULT = 30.0
        R_MAX_DEFAULT = 80.0
        C_MIN_DEFAULT = 40.0
        C_MAX_DEFAULT = 120.0
        
        G_with_capacity = generar_capacidades_y_costos(
            G_base, C_MIN_DEFAULT, C_MAX_DEFAULT
        )

        st.session_state.G_actual = G_with_capacity
        st.session_state.num_emergencias = 5 # Valor por defecto
        st.session_state.R_MIN = R_MIN_DEFAULT
        st.session_state.R_MAX = R_MAX_DEFAULT
        st.session_state.C_MIN = C_MIN_DEFAULT
        st.session_state.C_MAX = C_MAX_DEFAULT
        
        # 2. Generar los flujos iniciales
        st.session_state.flujos_k = generar_flujos_emergencia(
            st.session_state.G_actual, LUGAR_CENTRAL, st.session_state.num_emergencias, 
            st.session_state.R_MIN, st.session_state.R_MAX
        )
        
    G_actual = st.session_state.G_actual
    
    # 5.2 BARRA LATERAL DE CONFIGURACIÓN
    st.sidebar.header("⚙️ Configuración del Modelo")

    # Controles de Flujo (R)
    st.sidebar.subheader("Requerimientos de Velocidad (Rᵢ)")
    st.session_state.R_MIN = st.sidebar.slider("R_MÍN (km/h)", 10.0, 100.0, st.session_state.R_MIN, 5.0)
    st.session_state.R_MAX = st.sidebar.slider("R_MÁX (km/h)", 50.0, 150.0, st.session_state.R_MAX, 5.0)
    
    # Controles de Capacidad (C)
    st.sidebar.subheader("Capacidades Viales (μᵢⱼ)")
    st.session_state.C_MIN = st.sidebar.slider("C_MÍN (km/h)", 10.0, 100.0, st.session_state.C_MIN, 5.0)
    st.session_state.C_MAX = st.sidebar.slider("C_MÁX (km/h)", 50.0, 150.0, st.session_state.C_MAX, 5.0)

    # Control de Número de Emergencias
    st.session_state.num_emergencias = st.sidebar.number_input(
        "Número de Emergencias (Flujos)", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.num_emergencias, 
        step=1
    )

    # 5.3 LÓGICA DE BOTONES Y ACCIONES
    
    col_cap, col_flujo = st.sidebar.columns(2)
    
    if col_cap.button("🔄 Recalcular Capacidades", key='recalc_cap'):
        with st.spinner("Regenerando capacidades viales..."):
            st.session_state.G_actual = generar_capacidades_y_costos(
                G_base, st.session_state.C_MIN, st.session_state.C_MAX
            )
        st.sidebar.success("Capacidades actualizadas.")

    if col_flujo.button("📍 Recalcular Flujos/Destinos", key='recalc_flujo'):
        with st.spinner("Generando nuevos flujos y destinos..."):
            st.session_state.flujos_k = generar_flujos_emergencia(
                st.session_state.G_actual, LUGAR_CENTRAL, st.session_state.num_emergencias, 
                st.session_state.R_MIN, st.session_state.R_MAX
            )
        st.sidebar.success("Flujos/Emergencias reasignados.")

    # 5.4 EJECUCIÓN DEL MODELO Y VISUALIZACIÓN DE RESULTADOS
    
    st.header("1. Mapa de Rutas Óptimas")
    
    # Botón principal para ejecutar el solver
    if st.button("🚀 Optimizar y Mostrar Rutas (Ejecutar Modelo)", type="primary"):
        
        # Ejecutar el modelo
        with st.spinner("Resolviendo el modelo de optimización PuLP (Flujo Multicommodity)..."):
            rutas_optimas, status, costo_total_minimo, total_tiempo_viaje, total_penalizacion = \
                resolver_modelo_multifluido(st.session_state.G_actual, st.session_state.flujos_k)
        
        # 5.4.1 VISUALIZACIÓN DE RESULTADOS
        
        if status == 1: # Optimal
            st.success(f"✅ Modelo Resuelto: ÓPTIMO (Costo Total: ${costo_total_minimo:.2f})")
            
            # Dibujar en el mapa
            ruta_info = dibujar_rutas_en_mapa(st.session_state.G_actual, st.session_state.flujos_k, rutas_optimas)

            # 5.4.2 Dashboard de Métricas
            st.header("2. Métricas y Análisis")
            
            col1, col2, col3 = st.columns(3)
            
            # Columna 1: Costo Total
            col1.metric(
                "Costo Total Mínimo", 
                f"${costo_total_minimo:.2f}", 
                "Tiempo + Operación + Penalización"
            )
            # Columna 2: Tiempo de Viaje
            col2.metric(
                "Tiempo de Viaje (Σ Minutos)", 
                f"{total_tiempo_viaje:.2f} min", 
                f"Costo Operativo Fijo: ${costo_operativo_fijo(st.session_state.flujos_k):.2f}"
            )
            # Columna 3: Penalización
            if total_penalizacion > 0:
                col3.metric(
                    "Costo de Penalización (Big M)", 
                    f"${total_penalizacion:,.0f}", 
                    "¡Advertencia! Se violó Rk"
                )
            else:
                col3.metric(
                    "Costo de Penalización (Big M)", 
                    f"$0", 
                    "Todas las rutas cumplieron Rk"
                )


            # 5.4.3 Tabla de Análisis de Flujos
            st.subheader("3. Detalle por Flujo (Emergencia)")
            
            datos_tabla = []
            for flujo_id, info in ruta_info.items():
                datos_tabla.append({
                    "Flujo ID": flujo_id,
                    "Urgencia": info['tipo'],
                    "R_Requerida (km/h)": f"{info['Rk']:.1f}",
                    "Tiempo Ruta (min)": f"{info['tiempo_total']:.2f}",
                    "Aristas Violadas": info['aristas_violadas'],
                    "Costo Operativo": TIPOS_AMBULANCIA[info['tipo']]['costo_operativo']
                })

            df_resultados = pd.DataFrame(datos_tabla)
            st.dataframe(df_resultados, use_container_width=True)

        elif status == -1:
            st.error("❌ El modelo no encontró una solución (Infactible). Esto solo puede ocurrir si Origen y Destino están desconectados en el grafo original.")
        else:
            st.warning(f"El solver terminó con el estado: {pulp.LpStatus[status]}. Intente recalcular.")
        
    st.subheader("Información de la Red")
    colA, colB, colC = st.columns(3)
    colA.metric("Nodos", len(G_actual.nodes))
    colB.metric("Aristas", len(G_actual.edges))
    colC.metric("Base Central", LUGAR_CENTRAL)
    

def costo_operativo_fijo(flujos_k):
    return sum(f['costo_operativo'] for f in flujos_k)

if __name__ == '__main__':
    main()