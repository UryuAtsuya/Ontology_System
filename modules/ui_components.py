import streamlit as st
import json
import os
import pandas as pd
from modules.llm_client import LLMClient
from modules.evaluation import GeminiEvaluator

def render_sidebar():
    """Renders the sidebar configuration."""
    with st.sidebar:
        st.header("System Config")
        default_key = st.secrets.get("GEMINI_API_KEY", "")
        openai_key = st.text_input("Gemini API Key", value=default_key, type="password")
        uploaded_file = st.file_uploader("Upload OWL File", type=["owl", "rdf", "xml"])
        return openai_key, uploaded_file

def render_ontology_manager_tab(om, openai_key):
    """Renders the Ontology Manager tab."""
    if om.ontology:
        # --- Section 1: Individual Inspector ---
        st.subheader("🔍 Individual Inspector")
        col_insp1, col_insp2 = st.columns([1, 1])
        
        with col_insp1:
            individuals = list(om.ontology.individuals())
            ind_names = [i.name for i in individuals]
            selected_ind_name = st.selectbox("Select Individual", [""] + ind_names)
            
            if selected_ind_name:
                ind_data = om.get_individual_data(selected_ind_name)
                st.json(ind_data)
                
                if st.button("Generate Description with AI"):
                    if not openai_key:
                        st.error("Please enter Gemini API Key.")
                    else:
                        llm = LLMClient(openai_key)
                        with st.spinner("Generating description..."):
                            prompt = f"Describe this building based on the following data: {json.dumps(ind_data)}"
                            response = llm.generate_response(prompt, "Data provided in prompt.")
                            st.write(response)

        # --- Section 2: Add / Update Instance ---
        st.subheader("✏️ Add / Update Instance")
        with st.expander("Open Editor", expanded=True):
            classes = om.get_classes()
            class_names = [c.name for c in classes]
            
            default_index = 0
            if "建築物" in class_names:
                default_index = class_names.index("建築物")
            
            selected_cls = st.selectbox("Class", class_names, index=default_index)
            
            if selected_cls:
                cls_entity = om.ontology[selected_cls]
                d_props, o_props = om.get_properties_for_class(cls_entity)
                
                with st.form("add_ind_form"):
                    new_name = st.text_input("Instance Name (ID)", value=selected_ind_name if selected_ind_name else "")
                    props_input = {}
                    
                    priority_props = ["名称", "高さ_m", "階数", "延床面積_m2", "竣工年", "所在地"]
                    
                    st.markdown("### Data Properties")
                    
                    for dp in d_props:
                        if selected_cls == "建築物" and dp.name not in priority_props:
                            continue 
                            
                        val = ""
                        if selected_ind_name and selected_ind_name == new_name:
                             curr_data = om.get_individual_data(selected_ind_name)
                             if dp.name in curr_data.get("properties", {}):
                                 val = curr_data["properties"][dp.name]
                                 if isinstance(val, list): val = ", ".join(map(str, val))
                        
                        props_input[dp.name] = st.text_input(f"{dp.name}", value=str(val))
                    
                    if selected_cls != "建築物":
                         for dp in d_props:
                            if dp.name in props_input: continue
                            val = ""
                            props_input[dp.name] = st.text_input(f"{dp.name}", value=str(val))

                    st.markdown("### Object Properties")
                    for op in o_props:
                        candidates = [i.name for i in om.ontology.individuals()]
                        props_input[op.name] = st.selectbox(f"{op.name}", [""] + candidates)

                    if st.form_submit_button("Save Instance"):
                        clean_props = {k: v for k, v in props_input.items() if v}
                        try:
                            om.add_individual(selected_cls, new_name, clean_props)
                            st.success(f"Saved {new_name}")
                            if "current_owl_file" in st.session_state:
                                om.save_ontology(st.session_state["current_owl_file"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        # --- Section 3: Download ---
        st.subheader("💾 Export Ontology")
        if "current_owl_file" in st.session_state:
            with open(st.session_state["current_owl_file"], "rb") as f:
                st.download_button(
                    label="Download Updated OWL",
                    data=f,
                    file_name="updated_ontology.owl",
                    mime="application/rdf+xml"
                )

        # --- Section 4: Visualization ---
        st.subheader("🕸️ Knowledge Graph")
        if "inference_result" in st.session_state:
            st.info("Inference Results Available (See below)")
            st.json(st.session_state["inference_result"])

        graph = om.visualize_graph()
        st.graphviz_chart(graph, use_container_width=True)
        
    else:
        st.info("Please upload an ontology file.")

def render_reasoning_engine_tab(om):
    """Renders the Reasoning Engine tab."""
    st.subheader("Automatic Inference & Consistency Check")
    if st.button("Run Pellet Reasoner"):
        with st.spinner("Reasoning..."):
            result = om.run_reasoner()
            st.session_state["inference_result"] = result
            
            if result["status"] == "success":
                st.success(result["message"])
                if result["inconsistent_classes"]:
                    st.error(f"Inconsistent Classes: {result['inconsistent_classes']}")
                
                if "output_log" in result and result["output_log"]:
                    with st.expander("Show Inference Log (Reparenting Details)", expanded=True):
                        st.text(result["output_log"])
            else:
                st.error(result["message"])
                if "output_log" in result and result["output_log"]:
                    with st.expander("Show Error Log"):
                        st.text(result["output_log"])
            
    st.subheader("SWRL Rule Editor")
    rule_text = st.text_area("Edit Rules (Format: Class(?x) ^ hasVal(?x,?y) -> NewProp(?x,?z))", height=150)
    if st.button("Apply Rules"):
        if rule_text:
            with st.spinner("Applying Rule..."):
                result = om.apply_swrl_rule(rule_text)
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result["message"])
        else:
            st.warning("Please enter a rule.")

def render_ai_assistant_tab(openai_key):
    """Renders the AI Assistant (RAG) tab."""
    st.subheader("Ontology-Guided Q&A")
    
    if not openai_key:
        st.warning("Please enter Gemini API Key in sidebar.")
    else:
        llm = LLMClient(openai_key)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the building regulations or specs..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving & Reasoning..."):
                    retriever = st.session_state.get("retriever")
                    if retriever:
                        context_items = retriever.semantic_search(prompt)
                        context_str = retriever.format_context_for_llm(context_items)
                    else:
                        context_str = "No ontology loaded."
                    
                    response_text = llm.generate_response(prompt, context_str)
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

def render_evaluation_tab(openai_key):
    """Renders the Evaluation tab."""
    st.subheader("🧪 Evaluation: Few-shot vs Zero-shot")
    st.info("Compare performance metrics and qualitative differences between Ontology-Guided (Few-shot) and Standard (Zero-shot) prompting.")

    ontology_txt_path = "Ontology.txt"
    default_ontology_content = ""
    if os.path.exists(ontology_txt_path):
        with open(ontology_txt_path, "r", encoding="utf-8") as f:
            default_ontology_content = f.read()
    
    col_conf1, col_conf2 = st.columns([1, 1])
    with col_conf1:
        model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    with col_conf2:
        runs = st.slider("Number of Runs (per mode)", min_value=1, max_value=10, value=3)

    with st.expander("📝 Edit Few-shot Context (Ontology)", expanded=False):
        fewshot_doc = st.text_area("System Instruction / Context", value=default_ontology_content, height=200)

    with st.expander("⚖️ Factual Accuracy Settings (Optional)", expanded=False):
        gold_text = st.text_area("Gold Standard / Reference Knowledge", height=150, help="Enter the correct facts/knowledge here to evaluate accuracy.")
        enable_factual = st.checkbox("Enable Factual Accuracy (NLI & Evidence Recall)", value=False, help="Requires downloading a large model (~2GB). May be slow.")

    test_query = st.text_area("Test Query", value="Tell me about base-isolated buildings in Tokyo built after 2000.", height=100)
    
    if st.button("🚀 Run Evaluation"):
        if not openai_key:
            st.error("Please enter Gemini API Key in sidebar.")
        else:
            evaluator = GeminiEvaluator(openai_key)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p, text):
                progress_bar.progress(p)
                status_text.text(text)
            
            try:
                results = evaluator.run_evaluation(
                    query=test_query,
                    fewshot_doc=fewshot_doc,
                    runs=runs,
                    model_name=model_name,
                    gold_text=gold_text if gold_text else None,
                    enable_factual=enable_factual,
                    progress_callback=update_progress
                )
                
                status_text.text("Evaluation Complete!")
                progress_bar.progress(100)
                
                st.divider()
                
                # 1. Metrics Table
                st.markdown("### 📊 Quantitative Metrics (Avg)")
                
                def get_avg_metrics(metrics_dict):
                    return {k: sum(v)/len(v) if v else 0.0 for k, v in metrics_dict.items()}
                
                avg_fs = get_avg_metrics(results["metrics_fs"])
                avg_zs = get_avg_metrics(results["metrics_zs"])
                
                if results.get("factual_fs"):
                    avg_fs["NLI Entailment"] = sum([x.nli_entail_rate for x in results["factual_fs"]]) / len(results["factual_fs"])
                    avg_fs["Evidence Recall"] = sum([x.evidence_recall for x in results["factual_fs"]]) / len(results["factual_fs"])
                if results.get("factual_zs"):
                    avg_zs["NLI Entailment"] = sum([x.nli_entail_rate for x in results["factual_zs"]]) / len(results["factual_zs"])
                    avg_zs["Evidence Recall"] = sum([x.evidence_recall for x in results["factual_zs"]]) / len(results["factual_zs"])

                df_metrics = pd.DataFrame([avg_fs, avg_zs], index=["Few-shot", "Zero-shot"])
                st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'))
                
                # 2. Diversity
                st.markdown("### 🌈 Diversity (Distinct-N)")
                div_df = pd.DataFrame([results["diversity_fs"], results["diversity_zs"]], index=["Few-shot", "Zero-shot"])
                st.table(div_df)
                
                # 3. Qualitative Diff
                st.markdown("### 🔍 Unique Words in Few-shot (Top 20)")
                st.write("Words that appear more frequently in Few-shot answers compared to Zero-shot.")
                diff_items = [{"Word": k, "Count": v} for k, v in results["diff_tokens"]]
                if diff_items:
                    st.dataframe(pd.DataFrame(diff_items).T)
                else:
                    st.info("No significant unique words found.")

                # 4. Worst Examples
                st.markdown("### ⚠️ Worst Examples (vs Reference)")
                st.caption(f"Reference Answer (Few-shot #1): {results['reference'][:100]}...")
                
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    st.markdown("**Few-shot Worst Cases**")
                    for idx, score, txt in results["worst_fs"]:
                        with st.expander(f"Score: {score:.3f}"):
                            st.text(txt)
                with col_w2:
                    st.markdown("**Zero-shot Worst Cases**")
                    for idx, score, txt in results["worst_zs"]:
                        with st.expander(f"Score: {score:.3f}"):
                            st.text(txt)
                            
                # 5. Full Data Download
                st.markdown("### 📥 Download Report")
                report_str = f"Query: {test_query}\n\n"
                report_str += "--- Few-shot Answers ---\n" + "\n\n".join(results["fewshot_answers"]) + "\n\n"
                report_str += "--- Zero-shot Answers ---\n" + "\n\n".join(results["zeroshot_answers"])
                
                st.download_button("Download Full Report", report_str, file_name="eval_report.txt")
                
                st.success("Results logged to `runs/eval_runs.sqlite`")

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")
