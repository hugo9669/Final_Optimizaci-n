import streamlit as st
# import osmnx as ox # Ya no se usa directamente
import networkx as nx
import pulp # Mantenemos PuLP
import numpy as np
import pandas as pd
import folium
from streamlit_folium import folium_static
import random

# --- 1. CONFIGURACIÓN INICIAL Y DATOS FICTICIOS ---

# Configuración del área de estudio (Bello, Antioquia)
LATITUDE_CENTRAL = 6.333
LONGITUDE_CENTRAL = -75.568
DISTANCIA_KM = 0.5 # Radio para un área de ~1 km²

# Tipos de Flujo (Ambulancias) y sus costos operativos
TIPOS_AMBULANCIA = {
    'Leve': {'costo_operativo': 100, 'nombre': 'Ambulancia de Transporte Simple', 'color': 'green', 'priority': 1},
    'Media': {'costo_operativo': 250, 'nombre': 'Ambulancia de Cuidados Intermedios', 'color': 'orange', 'priority': 2},
    'Critica': {'costo_operativo': 500, 'nombre': 'Ambulancia de Cuidados Críticos', 'color': 'red', 'priority': 3},
}

# --- 2. FUNCIONES AUXILIARES ---

def costo_operativo_fijo(flujos_k):
    """Calcula el costo operativo fijo total de todos los flujos."""
    return sum(f['costo_operativo'] for f in flujos_k)

@st.cache_data(show_spinner="Simulando y preparando el grafo vial...")
def cargar_grafo_inicial():
    """Crea un grafo simulado de NetworkX con atributos necesarios para la optimización y visualización."""
    
    # === SIMULACIÓN DE GRAFO (Reemplaza a OSMnx) ===
    G = nx.MultiDiGraph()
    
    # Crear 10 nodos con coordenadas simuladas
    nodos = [i + 1 for i in range(10)]
    coords = {}
    
    # Coordenadas alrededor del punto central para visualización
    base_lat = LATITUDE_CENTRAL
    base_lon = LONGITUDE_CENTRAL
    
    for i in nodos:
        lat = base_lat + random.uniform(-0.005, 0.005)
        lon = base_lon + random.uniform(-0.005, 0.005)
        coords[i] = {'y': lat, 'x': lon}
        G.add_node(i, y=lat, x=lon)

    # Agregar aristas simuladas con atributos
    aristas = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (1, 10), (3, 8)
    ]
    
    for u, v in aristas:
        # Añadir arista de ida
        length_m = random.uniform(50, 500)
        G.add_edge(u, v, key=0, length=length_m, travel_time=(length_m / 1000) / (60 / 60))
        # Añadir arista de vuelta (bidireccionalidad)
        length_m = random.uniform(50, 500)
        G.add_edge(v, u, key=0, length=length_m, travel_time=(length_m / 1000) / (60 / 60))

    # El nodo central será el nodo 1
    LUGAR_CENTRAL = 1
    
    return G, LUGAR_CENTRAL
    # === FIN SIMULACIÓN DE GRAFO ===


def generar_capacidades_y_costos(G, C_MIN, C_MAX):
    """Genera capacidades viales aleatorias y costos de tiempo actualizados."""
    
    H = G.copy()
    
    for u, v, k, data in H.edges(keys=True, data=True):
        capacidad_vial_kmh = random.uniform(C_MIN, C_MAX)
        data['capacidad_vial_kmh'] = capacidad_vial_kmh
        
        length_m = data.get('length', 1) 
        if capacidad_vial_kmh > 0:
            # Cálculo de costo_tiempo_min (tiempo de viaje) basado en la capacidad
            costo_tiempo_min = (length_m / 1000) / (capacidad_vial_kmh / 60)
        else:
            costo_tiempo_min = float('inf')
        
        data['costo_tiempo_min'] = costo_tiempo_min

    return H

def generar_flujos_emergencia(G, LUGAR_CENTRAL, NUM_EMERGENCIAS, R_MIN, R_MAX):
    """Genera flujos de emergencia con origen, destino y velocidad requerida."""
    
    # Aseguramos que el origen (BASE) no sea un destino
    nodos_disponibles = [n for n in G.nodes if n != LUGAR_CENTRAL]
    
    # Aseguramos que el número de destinos no exceda los nodos disponibles
    if NUM_EMERGENCIAS > len(nodos_disponibles):
        NUM_EMERGENCIAS = len(nodos_disponibles)
        
    nodos_destino = random.sample(nodos_disponibles, NUM_EMERGENCIAS)
    flujos_k = []

    for i, d_k in enumerate(nodos_destino):
        tipo = random.choice(list(TIPOS_AMBULANCIA.keys()))
        flujo_data = TIPOS_AMBULANCIA[tipo]
        
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


# --- 3. MODELO DE OPTIMIZACIÓN MULTIFLUJO (PULP) ---

def resolver_modelo_multifluido(G, flujos_k):
    """Implementa y resuelve el modelo de Flujo Multicommodity con Big M usando PuLP."""

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
    penalizacion_total = []

    # --- Restricciones de Enlace (Big M) y Función Objetivo ---
    for flujo in flujos_k:
        flujo_id = flujo['id']
        
        for u, v, k in A_keys:
            # Solo si la variable existe para este flujo y arista
            if (u, v, k, flujo_id) in variables_x:
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
                    modelo += variables_z[(u, v, k, flujo_id)] == 0, f"Penal_Req_CUMPLIDA_{u}_{v}_{k}_{flujo_id}"


    # Costo operativo fijo
    costo_op_fijo = costo_operativo_fijo(flujos_k)
    
    # --- Función Objetivo Final ---
    modelo += pulp.lpSum(costo_total_viaje) + pulp.lpSum(penalizacion_total) + costo_op_fijo, "Costo_Total_Minimo"

    # --- Restricciones de Conservación de Flujo ---
    for flujo in flujos_k:
        flujo_id = flujo['id']
        s = flujo['origen']
        d_k = flujo['destino']

        for i in G.nodes:
            # Flujo saliente: (i, j, k, flujo_id)
            # Aseguramos que 'get' devuelva 0 si la variable no existe
            flujo_saliente = [variables_x.get((i, j, m, flujo_id), 0) 
                              for j in G.neighbors(i) 
                              for m in G.get_edge_data(i, j, default={}) 
                              if (i, j, m, flujo_id) in variables_x]

            # Flujo entrante: (j, i, k, flujo_id)
            # Aseguramos que 'get' devuelva 0 si la variable no existe
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
        # Usa el solver predeterminado con msg=0 para silenciar el output
        modelo.solve(pulp.PULP_CBC_CMD(msg=0)) 
        
        if modelo.status == pulp.LpStatus["Optimal"]: 
            rutas_optimas = {}
            total_tiempo_viaje = pulp.value(pulp.lpSum(costo_total_viaje))
            total_penalizacion = pulp.value(pulp.lpSum(penalizacion_total))

            # Extracción de Rutas
            for u, v, k, flujo_id in variables_x: 
                if variables_x[(u, v, k, flujo_id)].varValue is not None and variables_x[(u, v, k, flujo_id)].varValue > 0.9:
                    if flujo_id not in rutas_optimas:
                        rutas_optimas[flujo_id] = []
                    rutas_optimas[flujo_id].append((u, v, k))
            
            # Usamos 1 para estado Optimal, igual que en el código anterior.
            return rutas_optimas, 1, pulp.value(modelo.objective), total_tiempo_viaje, total_penalizacion, costo_op_fijo
        
        elif modelo.status == pulp.LpStatus["Infeasible"]:
            return {}, -1, None, None, None, costo_op_fijo
            
        else:
            # -3 es un valor de estado distinto de Optimal/Infeasible
            return {}, -3, None, None, None, costo_op_fijo
            
    except Exception as e:
        # Error grave del solver (e.g., PuLP no encontró CBC)
        st.error(f"Error grave al resolver el modelo. Asegúrese de que el solver CBC esté instalado. Error: {e}")
        return {}, -2, None, None, None, costo_op_fijo


# --- 4. FUNCIONES DE VISUALIZACIÓN ---

def dibujar_rutas_en_mapa(G, flujos_data, rutas_optimas):
    """Dibuja el grafo, el origen, los destinos y las rutas óptimas con Folium."""
    
    s_node = flujos_data[0]['origen']
    if s_node not in G.nodes:
        # En el grafo simulado, esto no debería pasar si LUGAR_CENTRAL = 1
        st.error("Error de datos: El nodo de origen no se encontró en el grafo.") 
        return {} 
        
    # Usar las coordenadas del nodo central para centrar el mapa
    center_lat = G.nodes[s_node]['y']
    center_lon = G.nodes[s_node]['x']
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")
    
    ruta_info = {}
    
    # 1. Dibujar todos los nodos (puntos) y aristas (líneas grises)
    
    # Dibujar Nodos
    for node, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
             folium.CircleMarker(
                location=[data['y'], data['x']],
                radius=3,
                color='#666666',
                fill=True,
                fill_color='#AAAAAA',
                fill_opacity=0.7,
                tooltip=f"Nodo {node}"
            ).add_to(m)

    # Dibujar Aristas de Fondo (Simuladas)
    for u, v, k, data in G.edges(keys=True, data=True):
        u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
        v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
        
        # Evitar dibujar dos veces la misma línea para grafo bidireccional
        if u < v:
             folium.PolyLine(
                locations=[(u_lat, u_lon), (v_lat, v_lon)],
                color='#CCCCCC',
                weight=1,
                opacity=0.5
            ).add_to(m)


    # 2. Dibujar las Rutas Optimas y Marcadores
    for flujo in flujos_data:
        flujo_id = flujo['id']
        
        if flujo_id in rutas_optimas:
            ruta = rutas_optimas[flujo_id]
            s = flujo['origen']
            d_k = flujo['destino']
            R_k = flujo['velocidad_requerida_kmh']
            tipo_urgencia = flujo['tipo_urgencia']
            flujo_color = flujo['color']
            
            tiempo_ruta_acumulado = 0
            aristas_violadas_contador = 0
            
            # Dibujar Aristas de la Ruta
            for u, v, k in ruta:
                
                # Manejo de Nodos en el Grafo Simulado
                if u not in G.nodes or v not in G.nodes or 'y' not in G.nodes[u]:
                    continue
                    
                edge_data = G.edges[(u, v, k)]
                μ_ij = edge_data['capacidad_vial_kmh']
                costo_min = edge_data['costo_tiempo_min']
                
                u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
                v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']

                violacion = "❌ Violada (Rk > μij)" if R_k > μ_ij else "✅ Cumplida"
                if R_k > μ_ij:
                    aristas_violadas_contador += 1
                
                tiempo_ruta_acumulado += costo_min
                
                popup_html = f"""
                <b>Flujo:</b> {flujo_id} ({tipo_urgencia})<br>
                <b>Origen/Destino:</b> {u} -> {v}<br>
                <b>Requerida (Rk):</b> {R_k:.1f} km/h<br>
                <b>Capacidad (μij):</b> {μ_ij:.1f} km/h<br>
                <b>Restricción:</b> {violacion}<br>
                <b>Costo:</b> {costo_min:.2f} min
                """
                
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color=flujo_color,
                    weight=5, # Más grueso para ruta optimizada
                    opacity=0.9,
                    tooltip=flujo_id,
                ).add_child(folium.Popup(popup_html)).add_to(m)

            # Recopilación de Info de Ruta
            ruta_info[flujo_id] = {
                'tiempo_total': tiempo_ruta_acumulado, 
                'aristas_violadas': aristas_violadas_contador, 
                'tipo': tipo_urgencia, 
                'Rk': R_k, 
                'costo_operativo': flujo['costo_operativo']
            }
            
            # Marcador de Origen (Base)
            folium.Marker(
                location=[G.nodes[s]['y'], G.nodes[s]['x']],
                popup=f"BASE: {s} - Centro de Ambulancias",
                icon=folium.Icon(color='blue', icon='fa-ambulance', prefix='fa')
            ).add_to(m)
            
            # Marcador de Destino (Emergencia)
            folium.Marker(
                location=[G.nodes[d_k]['y'], G.nodes[d_k]['x']],
                popup=f"DESTINO: {d_k} - Emergencia {flujo_id} ({tipo_urgencia})",
                icon=folium.Icon(color=flujo_color, icon='fa-hospital', prefix='fa')
            ).add_to(m)
            
    folium_static(m)
    
    return ruta_info

# --- 5. INTERFAZ DE STREAMLIT ---

def main():
    st.set_page_config(layout="wide", page_title="Modelo de Enrutamiento Multiflujo para Ambulancias")

    st.title("🚑 Modelo de Enrutamiento Multiflujo (Ambulancias) - [Simulado]")
    st.caption("Implementación de Flujo Multicommodity con PuLP en un grafo de red simulado para evitar conflictos de OSMnx.")
    
    try:
        # Carga del grafo simulado
        G_base, LUGAR_CENTRAL = cargar_grafo_inicial()
        # Nota: LUGAR_CENTRAL = 1 en el grafo simulado
        
    except Exception as e:
        # Este error solo debería ocurrir si NetworkX falla.
        st.error(f"Error al inicializar NetworkX. Error: {e}")
        return

    if 'G_actual' not in st.session_state:
        
        R_MIN_DEFAULT = 30.0
        R_MAX_DEFAULT = 80.0
        C_MIN_DEFAULT = 40.0
        C_MAX_DEFAULT = 120.0
        # Reducimos el número de emergencias para el grafo simulado pequeño (máx 9 destinos)
        NUM_EMERGENCIAS_DEFAULT = 3 
        
        G_with_capacity = generar_capacidades_y_costos(
            G_base, C_MIN_DEFAULT, C_MAX_DEFAULT
        )

        st.session_state.G_actual = G_with_capacity
        st.session_state.num_emergencias = NUM_EMERGENCIAS_DEFAULT
        st.session_state.R_MIN = R_MIN_DEFAULT
        st.session_state.R_MAX = R_MAX_DEFAULT
        st.session_state.C_MIN = C_MIN_DEFAULT
        st.session_state.C_MAX = C_MAX_DEFAULT
        
        st.session_state.flujos_k = generar_flujos_emergencia(
            st.session_state.G_actual, LUGAR_CENTRAL, NUM_EMERGENCIAS_DEFAULT, 
            R_MIN_DEFAULT, R_MAX_DEFAULT
        )
        
    G_actual = st.session_state.G_actual
    
    st.sidebar.header("⚙️ Configuración del Modelo")

    st.sidebar.subheader("Requerimientos de Velocidad (Rᵢ)")
    st.session_state.R_MIN = st.sidebar.slider("R_MÍN (km/h)", 10.0, 100.0, st.session_state.R_MIN, 5.0, key='r_min_slider')
    st.session_state.R_MAX = st.sidebar.slider("R_MÁX (km/h)", 50.0, 150.0, st.session_state.R_MAX, 5.0, key='r_max_slider')
    
    st.sidebar.subheader("Capacidades Viales (μᵢⱼ)")
    st.session_state.C_MIN = st.sidebar.slider("C_MÍN (km/h)", 10.0, 100.0, st.session_state.C_MIN, 5.0, key='c_min_slider')
    st.session_state.C_MAX = st.sidebar.slider("C_MÁX (km/h)", 50.0, 150.0, st.session_state.C_MAX, 5.0, key='c_max_slider')

    st.session_state.num_emergencias = st.sidebar.number_input(
        "Número de Emergencias (Flujos)", 
        min_value=1, 
        max_value=9, # Máximo 9 destinos en el grafo simulado (10 nodos - 1 base)
        value=st.session_state.num_emergencias, 
        step=1,
        key='num_emergencias_input'
    )

    col_cap, col_flujo = st.sidebar.columns(2)
    
    if col_cap.button("🔄 Recalcular Capacidades", key='recalc_cap'):
        with st.spinner("Regenerando capacidades viales..."):
            st.session_state.G_actual = generar_capacidades_y_costos(
                G_base, st.session_state.C_MIN, st.session_state.C_MAX
            )
        st.sidebar.success("Capacidades actualizadas. Presione 'Optimizar' para usar las nuevas capacidades.")

    if col_flujo.button("📍 Recalcular Flujos/Destinos", key='recalc_flujo'):
        with st.spinner("Generando nuevos flujos y destinos..."):
            st.session_state.flujos_k = generar_flujos_emergencia(
                G_base, LUGAR_CENTRAL, st.session_state.num_emergencias, 
                st.session_state.R_MIN, st.session_state.R_MAX
            )
            st.session_state.G_actual = generar_capacidades_y_costos(
                G_base, st.session_state.C_MIN, st.session_state.C_MAX
            )
        st.sidebar.success("Flujos/Emergencias reasignados. Presione 'Optimizar' para usar los nuevos flujos.")

    st.header("1. Mapa de Rutas Óptimas")
    
    if st.button("🚀 Optimizar y Mostrar Rutas (Ejecutar Modelo)", type="primary", key='optimizar_button'):
        
        with st.spinner("Resolviendo el modelo de optimización PuLP (Flujo Multicommodity)..."):
            rutas_optimas, status, costo_total_minimo, total_tiempo_viaje, total_penalizacion, costo_op_fijo_calc = \
                resolver_modelo_multifluido(st.session_state.G_actual, st.session_state.flujos_k)
        
        st.session_state.rutas_optimas = rutas_optimas
        st.session_state.status = status
        st.session_state.costo_total_minimo = costo_total_minimo
        st.session_state.total_tiempo_viaje = total_tiempo_viaje
        st.session_state.total_penalizacion = total_penalizacion
        st.session_state.costo_op_fijo = costo_op_fijo_calc
        st.session_state.show_results = True
        
    
    if 'show_results' in st.session_state and st.session_state.show_results:
        
        status = st.session_state.status
        costo_total_minimo = st.session_state.costo_total_minimo
        rutas_optimas = st.session_state.rutas_optimas
        total_tiempo_viaje = st.session_state.total_tiempo_viaje
        total_penalizacion = st.session_state.total_penalizacion
        costo_op_fijo = st.session_state.costo_op_fijo

        if status == 1: # Optimal
            st.success(f"✅ Modelo Resuelto: ÓPTIMO (Costo Total: ${costo_total_minimo:.2f})")
            
            ruta_info = dibujar_rutas_en_mapa(st.session_state.G_actual, st.session_state.flujos_k, rutas_optimas)

            st.header("2. Métricas y Análisis")
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                "Costo Total Mínimo", 
                f"${costo_total_minimo:.2f}", 
                "Tiempo + Operación + Penalización"
            )
            col2.metric(
                "Tiempo de Viaje (Σ Minutos)", 
                f"{total_tiempo_viaje:.2f} min", 
                f"Costo Operativo Fijo: ${costo_op_fijo:.2f}"
            )
            
            if total_penalizacion is not None and total_penalizacion > 0:
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

            st.subheader("3. Detalle por Flujo (Emergencia)")
            
            datos_tabla = []
            for info in ruta_info.values():
                datos_tabla.append({
                    "Urgencia": info['tipo'],
                    "R_Requerida (km/h)": f"{info['Rk']:.1f}",
                    "Tiempo Ruta (min)": f"{info['tiempo_total']:.2f}",
                    "Aristas Violadas": info['aristas_violadas'],
                    "Costo Operativo": info['costo_operativo']
                })

            df_resultados = pd.DataFrame(datos_tabla)
            st.dataframe(df_resultados, use_container_width=True)

        elif status == -1:
            st.error("❌ El modelo no encontró una solución (Infactible). Esto solo puede ocurrir si Origen y Destino están desconectados en el grafo original.")
        elif status == -2:
             st.error("❌ Error de ejecución del solver. Por favor, revise la configuración de su entorno.")
        elif status == -3:
             st.warning(f"El solver terminó con el estado: {pulp.LpStatus[status]}. Intente recalcular.")
        else:
            st.warning(f"El solver terminó con un estado inesperado: {status}. Intente recalcular.")
        
    st.subheader("Información de la Red Simulada")
    colA, colB, colC = st.columns(3)
    colA.metric("Nodos", len(G_actual.nodes))
    colB.metric("Aristas", len(G_actual.edges))
    colC.metric("Base Central (Nodo)", LUGAR_CENTRAL)
    

if __name__ == '__main__':
    main()
