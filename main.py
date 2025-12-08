import streamlit as st
import os
from modules.ontology_manager import OntologyManager

st.set_page_config(page_title="Seismic Ontology System", layout="wide")

# オントロジーの初期化
if "manager" not in st.session_state:
    # uploadされたファイルがあればそれを使うロジックなどをここに
    st.session_state["manager"] = OntologyManager()

mgr = st.session_state["manager"]

st.title("🏗️ Dynamic Seismic Ontology System")
st.markdown("SWRLルールに基づく自動分類とナレッジグラフ構築")

# --- タブ構成 ---
tab_add, tab_reason, tab_visual = st.tabs(["➕ 建築物登録", "🧠 推論・検証", "📊 可視化"])

# --- TAB 1: 建築物登録 ---
with tab_add:
    st.header("新しい建築物の登録")
    st.info("ここでデータを入力すると、SWRLルールにより推論タブで自動的に「高層」や「耐震基準」が判定されます。")

    if mgr.ontology:
        with st.form("building_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # 基本情報の入力（Data Properties）
                name = st.text_input("名称 (必須)", placeholder="例: 新宿パークタワー")
                year = st.number_input("建築年", min_value=1800, max_value=2100, value=2024)
                height = st.number_input("高さ (m)", min_value=0.0, value=30.0)
                floors = st.number_input("階数", min_value=1, value=5)
            
            with col2:
                # マスタデータの取得（Object Propertiesの選択肢）
                # Ontology内の既存インスタンス（例：#RC_v, #東京都）を取得して選択肢にする
                
                # 都道府県
                locs = mgr.get_individuals_of_type("都道府県")
                loc_map = {i.name: i for i in locs}
                sel_loc = st.selectbox("場所にある", [""] + list(loc_map.keys()))
                
                # 構造種別 (#RC_v, #S_v...)
                structs = mgr.get_individuals_of_type("構造種別値")
                struct_map = {i.label.first() if i.label else i.name : i for i in structs}
                sel_struct = st.selectbox("構造種別を持つ", [""] + list(struct_map.keys()))
                
                # 用途 (#オフィス用途, #病院用途...)
                uses = mgr.get_individuals_of_type("用途値")
                use_map = {i.name.replace("用途", "") : i for i in uses}
                sel_use = st.selectbox("用途を持つ", [""] + list(use_map.keys()))
                
                # 耐震技術 (#免震構造_v...)
                techs = mgr.get_individuals_of_type("耐震技術値")
                tech_map = {i.label.first() if i.label else i.name : i for i in techs}
                sel_tech = st.selectbox("耐震技術を持つ", [""] + list(tech_map.keys()))

            submit = st.form_submit_button("Ontologyに追加")
            
            if submit and name:
                attrs = {
                    "名称": name,
                    "建築年": int(year),
                    "高さ_m": float(height),
                    "階数": int(floors),
                    "場所にある": loc_map.get(sel_loc),
                    "構造種別を持つ": struct_map.get(sel_struct),
                    "用途を持つ": use_map.get(sel_use),
                    "耐震技術を持つ": tech_map.get(sel_tech)
                }
                
                new_b = mgr.add_building(name, attrs)
                if new_b:
                    mgr.save_ontology()
                    st.success(f"「{name}」を追加しました。推論タブでクラス分類を確認してください。")
                    st.session_state["last_added"] = new_b
                else:
                    st.error("追加に失敗しました。")

# --- TAB 2: 推論・検証 ---
# --- 共通: 建築物リストの取得 ---
buildings = mgr.get_individuals_of_type("建築物")
b_names = [b.name for b in buildings] if buildings else []

# セッションステートの初期化
if "selected_building" not in st.session_state:
    st.session_state["selected_building"] = b_names[0] if b_names else None

# コールバック関数
def update_selected_building():
    st.session_state["selected_building"] = st.session_state["building_selector"]

# --- TAB 2: 推論・検証 ---
import datetime
import json

# --- TAB 2: 推論・検証 ---
with tab_reason:
    st.header("SWRL推論の実行")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        **適用される主なルール:**
        - **高さ >= 60m** → `高層建築物`
        - **建築年 >= 2000** → `2000基準`
        - **建築年 < 1981** → `旧耐震基準`
        """)
        if st.button("推論を実行 (Pellet Reasoner)"):
            with st.spinner("推論中..."):
                res = mgr.run_reasoner()
                st.success(res)
        
        st.divider()
        if st.button("推論結果を保存 (JSON)"):
            results = []
            # 全建築物を走査して結果を収集
            all_buildings = mgr.get_individuals_of_type("建築物")
            for b in all_buildings:
                b_classes = [c.name for c in b.is_a if hasattr(c, "name")]
                results.append({
                    "name": b.name,
                    "classes": b_classes
                })
            
            # 保存処理
            os.makedirs("inference_results", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results/result_{timestamp}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            st.success(f"推論結果を保存しました:\n`{filename}`")
    
    with col2:
        st.subheader("推論結果の確認")
        
        # 選択ボックス (同期用キーを使用)
        current_index = 0
        if st.session_state["selected_building"] in b_names:
            current_index = b_names.index(st.session_state["selected_building"])
            
        target_b_name = st.selectbox(
            "建築物を選択", 
            b_names, 
            index=current_index,
            key="building_selector_reason",
            on_change=lambda: st.session_state.update({"selected_building": st.session_state.building_selector_reason})
        )
        
        # 選択された建築物の詳細表示
        if target_b_name:
            target_b = mgr.ontology.search_one(iri=f"*{target_b_name}")
            if target_b:
                st.write(f"**名称:** {target_b.名称 if target_b.名称 else ''}")
                
                # 所属クラスの表示
                classes = [c.name for c in target_b.is_a if hasattr(c, "name")]
                
                st.write("**現在の分類 (Classes):**")
                
                # 検証対象クラスの選択
                target_classes = st.multiselect(
                    "検証対象クラスを選択",
                    ['高層建築物', '免震建築物', '制震建築物', '新耐震基準', '旧耐震基準'],
                    default=['高層建築物', '免震建築物']
                )

                # 自動分類された重要なクラスをハイライト表示
                if target_classes:
                    cols = st.columns(len(target_classes))
                    for idx, cls_name in enumerate(target_classes):
                        is_match = cls_name in classes
                        cols[idx].metric(
                            f"{cls_name}判定", 
                            "YES" if is_match else "NO", 
                            delta="適合" if is_match else None
                        )
                
                st.caption(f"全所属クラス: {', '.join(classes)}")

# --- TAB 3: 可視化 ---
with tab_visual:
    st.header("ナレッジグラフ可視化")
    
    # 選択ボックス (同期用)
    current_index_vis = 0
    if st.session_state["selected_building"] in b_names:
        current_index_vis = b_names.index(st.session_state["selected_building"])

    target_b_name_vis = st.selectbox(
        "可視化する建築物を選択", 
        b_names, 
        index=current_index_vis,
        key="building_selector_visual",
        on_change=lambda: st.session_state.update({"selected_building": st.session_state.building_selector_visual})
    )

    if target_b_name_vis:
        target_b_vis = mgr.ontology.search_one(iri=f"*{target_b_name_vis}")
        if target_b_vis:
            graph = mgr.visualize_building(target_b_vis)
            st.graphviz_chart(graph)
    else:
        st.info("建築物が登録されていません。")