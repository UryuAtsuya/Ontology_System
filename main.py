import streamlit as st
import os
import datetime
import json
from modules.ontology_manager import OntologyManager
from modules.llm_client import LLMClient
from modules.rag_engine import HybridRetriever

st.set_page_config(page_title="Seismic Ontology System", layout="wide")

# --- Sidebar: Config ---
with st.sidebar:
    st.header("System Config")
    default_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    uploaded_file = st.file_uploader("Upload OWL File", type=["owl", "rdf", "xml"])

# --- Initialization ---
if "manager" not in st.session_state:
    st.session_state["manager"] = OntologyManager()

mgr = st.session_state["manager"]

# Initialize LLM & Retriever if API Key is present
if api_key and "retriever" not in st.session_state:
    try:
        llm_client = LLMClient(api_key)
        retriever = HybridRetriever(mgr, llm_client)
        st.session_state["llm_client"] = llm_client
        st.session_state["retriever"] = retriever
    except Exception as e:
        st.error(f"Failed to initialize AI modules: {e}")

st.title("🏗️ Dynamic Seismic Ontology System")
st.markdown("SWRLルールに基づく自動分類とナレッジグラフ構築")

# --- タブ構成 ---
tab_add, tab_reason, tab_visual, tab_ai = st.tabs(["➕ 建築物登録", "🧠 推論・検証", "📊 可視化", "🤖 AI Assistant"])

# --- TAB 1: 建築物登録 ---
with tab_add:
    st.header("新しい建築物の登録")
    st.info("ここでデータを入力すると、SWRLルールにより推論タブで自動的に「高層」や「耐震基準」が判定されます。")

    if mgr.ontology:
        # --- Auto-Complete Section ---
        with st.expander("📝 自然言語から自動入力 (Auto-Complete)", expanded=False):
            raw_text = st.text_area("建築物の説明を貼り付けてください", placeholder="例: 2020年に竣工した、高さ150mの横浜にある鉄骨造のオフィスビル。")
            if st.button("Extract & Auto-fill (AI解析)"):
                if raw_text:
                    if "llm_client" in st.session_state:
                         llm = st.session_state["llm_client"]
                         
                         # Options for mapping
                         locs = mgr.get_individuals_of_type("都道府県")
                         structs = mgr.get_individuals_of_type("構造種別値")
                         uses = mgr.get_individuals_of_type("用途値")
                         techs = mgr.get_individuals_of_type("耐震技術値")
                         
                         options = {
                             "都道府県": [i.name for i in locs],
                             "構造種別": [i.label.first() if i.label else i.name for i in structs],
                             "用途": [i.name.replace("用途", "") for i in uses],
                             "耐震技術": [i.label.first() if i.label else i.name for i in techs]
                         }
                         
                         with st.spinner("Parsing..."):
                             parsed = llm.parse_building_info(raw_text, options)
                             st.session_state["parsed_data"] = parsed
                             st.success("解析完了！下のフォームに反映されました。")
                    else:
                        st.error("AIモジュールが初期化されていません。APIキーを確認してください。")

        # Get default values from parsed data
        p_data = st.session_state.get("parsed_data", {})
        
        with st.form("building_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # 基本情報の入力
                name = st.text_input("名称 (必須)", value=p_data.get("名称", ""), placeholder="例: 新宿パークタワー")
                year = st.number_input("建築年", min_value=1800, max_value=2100, value=int(p_data.get("建築年", 2024)))
                height = st.number_input("高さ (m)", min_value=0.0, value=float(p_data.get("高さ_m", 30.0)))
                floors = st.number_input("階数", min_value=1, value=int(p_data.get("階数", 5)))
            
            with col2:
                # マスタデータの取得
                locs = mgr.get_individuals_of_type("都道府県")
                loc_map = {i.name: i for i in locs}
                
                # Try to match index
                def get_idx(options, target):
                    if not target: return 0
                    try: return options.index(target) + 1
                    except: return 0
                
                sel_loc = st.selectbox("場所にある", [""] + list(loc_map.keys()), index=get_idx(list(loc_map.keys()), p_data.get("場所にある")))
                
                structs = mgr.get_individuals_of_type("構造種別値")
                struct_map = {i.label.first() if i.label else i.name : i for i in structs}
                sel_struct = st.selectbox("構造種別を持つ", [""] + list(struct_map.keys()), index=get_idx(list(struct_map.keys()), p_data.get("構造種別を持つ")))
                
                uses = mgr.get_individuals_of_type("用途値")
                use_map = {i.name.replace("用途", "") : i for i in uses}
                sel_use = st.selectbox("用途を持つ", [""] + list(use_map.keys()), index=get_idx(list(use_map.keys()), p_data.get("用途を持つ")))
                
                techs = mgr.get_individuals_of_type("耐震技術値")
                tech_map = {i.label.first() if i.label else i.name : i for i in techs}
                sel_tech = st.selectbox("耐震技術を持つ", [""] + list(tech_map.keys()), index=get_idx(list(tech_map.keys()), p_data.get("耐震技術を持つ")))

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
                    st.success(f"「{name}」を追加しました。")
                    st.session_state["last_added"] = new_b
                else:
                    st.error("追加に失敗しました。")

# --- TAB 2: 推論・検証 ---
with tab_reason:
    st.header("SWRL推論の実行")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("推論を実行 (Pellet Reasoner)"):
            with st.spinner("推論中..."):
                res = mgr.run_reasoner()
                
                if isinstance(res, dict) and res.get("status") == "Success":
                    st.success(res["message"])
                    st.session_state["last_inference_report"] = res.get("report", [])
                elif isinstance(res, str): # Fallback for old version
                    st.success(res)
                else:
                    st.error(res.get("message", "Error"))
        
        # 結果保存ボタン
        if st.button("推論詳細ログを保存"):
             # ... (simplified save logic)
             pass

    with col2:
        st.subheader("推論結果の説明 (Explainability)")
        report = st.session_state.get("last_inference_report", [])
        if report:
             for item in report:
                with st.expander(f"🏗️ {item['name']} (Classes: {', '.join(item['classes'])})"):
                    if item["explanations"]:
                        st.markdown("#### 📝 Why?")
                        for cls, rules in item["explanations"].items():
                            st.markdown(f"**Classified as `{cls}` because:**")
                            for r in rules:
                                st.code(r, language="text")
                    else:
                        st.info("No specific SWRL rules triggered.")
        else:
            st.info("推論を実行するとここに結果が表示されます。")

# --- TAB 3: 可視化 ---
# (Previous code assumed here, just keeping it minimal for simplicity in this replacement)
# Re-implementing correctly
# --- 共通: 建築物リストの取得 ---
buildings = mgr.get_individuals_of_type("建築物")
b_names = [b.name for b in buildings] if buildings else []

if "selected_building" not in st.session_state:
    st.session_state["selected_building"] = b_names[0] if b_names else None

with tab_visual:
    st.header("ナレッジグラフ可視化")
    if b_names:
        target_b_name_vis = st.selectbox("可視化する建築物", b_names, key="vis_sel")
        if target_b_name_vis:
            target_b = mgr.ontology.search_one(iri=f"*{target_b_name_vis}")
            if target_b:
                graph = mgr.visualize_building(target_b)
                st.graphviz_chart(graph)
    else:
        st.info("No buildings found.")

# --- TAB 4: AI Assistant (Vector Search) ---
with tab_ai:
    st.header("Ontology-Guided AI Chat")
    if not api_key:
        st.warning("Please enter Gemini API Key in the sidebar.")
    else:
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("質問を入力してください..."):
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.get("retriever")
                    llm = st.session_state.get("llm_client")
                    
                    if retriever and llm:
                        context_items = retriever.semantic_search(prompt)
                        context_str = retriever.format_context_for_llm(context_items)
                        # Pass context_items to enable Dynamic Few-Shot generation
                        response_text = llm.generate_response(prompt, context_str, retrieved_items=context_items)
                    else:
                        response_text = "AI modules not initialized. Please check API Key."
                    
                    st.markdown(response_text)
                    st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})