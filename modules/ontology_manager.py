from owlready2 import *
import graphviz
import os

class OntologyManager:
    def __init__(self, file_path="temp_Ontology.owl"):
        self.ontology = None
        # 事前にJavaのパス設定が必要な場合があります
        # owlready2.JAVA_EXE = "/usr/bin/java"
        
        if os.path.exists(file_path):
            self.load_ontology(file_path)
        else:
            print(f"Warning: Ontology file not found at {file_path}")

    def load_ontology(self, file_path):
        try:
            file_path = os.path.abspath(file_path)
            # RDF/XML形式としてロード（SWRLルールを含むため）
            self.ontology = get_ontology(f"file://{file_path}").load()
            print("Ontology loaded successfully.")
        except Exception as e:
            print(f"Error loading ontology: {e}")

    def get_building_class(self):
        """建築物クラスを取得"""
        if not self.ontology: return None
        # 名前空間の検索（#建築物 または 建築物）
        return self.ontology.search_one(iri="*建築物")

    def get_individuals_of_type(self, type_name):
        """特定の型（例：構造種別値、都道府県）を持つ個体のリストを取得"""
        if not self.ontology: return []
        target_class = self.ontology.search_one(iri=f"*{type_name}")
        if target_class:
            return target_class.instances()
        return []

    def add_building(self, name, attributes):
        """
        建築物を追加する専用関数
        attributes: {
            "建築年": int, "高さ_m": float, "階数": int, "名称": str,
            "場所にある": individual, "構造種別を持つ": individual,...
        }
        """
        with self.ontology:
            building_cls = self.get_building_class()
            if not building_cls: return None
            
            # URIに使用するID生成（スペース等は除去）
            safe_name = name.replace(" ", "_")
            new_building = building_cls(safe_name)
            
            # データプロパティとオブジェクトプロパティの設定
            for prop_name, value in attributes.items():
                if value is None or value == "": continue
                
                # プロパティを検索
                prop = self.ontology.search_one(iri=f"*{prop_name}")
                if prop:
                    # 値の設定（リスト形式か単一値か確認）
                    try:
                        # Owlready2ではプロパティへの代入はリストまたは単一値
                        if isinstance(prop, ObjectPropertyClass):
                            getattr(new_building, prop.name).append(value)
                        else:
                            setattr(new_building, prop.name, value)
                    except Exception as e:
                        print(f"Error setting property {prop_name}: {e}")
            
            return new_building

    def get_explanation_for_class(self, class_name):
        """
        指定されたクラスに分類された理由（自然言語の説明）を返す
        """
        # SWRLルールの説明マッピング (論文用デモ定義)
        # swrl_examples.md や main.py の記載に基づく
        RULE_EXPLANATIONS = {
            # クラス名: 説明
            "高層建築物": "高さが60m以上 (Height >= 60m)",
            "2000基準": "2000年以降に建築 (Year >= 2000)",
            "旧耐震基準": "1981年以前に建築 (Year < 1981)",
            "新耐震基準": "1981年以降に建築 (Year >= 1981)",
            "長周期地震動注意建築物": "高さ60m以上の鉄骨造 (Height >= 60m AND Structure = Steel)",
            "免震建築物": "免震構造を持つ (Has Seismic Isolation)",
            "制震建築物": "制震構造を持つ (Has Damping Structure)",
            "歴史的建造物": "築50年以上 (Age > 50 years)",
            "木造建築物": "構造が木造 (Structure = Wood)",
            "オフィス建築物": "用途がオフィス (Usage = Office)",
            "商業施設": "用途が商業施設 (Usage = Commercial)",
        }
        
        return RULE_EXPLANATIONS.get(class_name, None)

    def get_matching_rules(self, class_name):
        # Backward compatibility or fallback if needed
        return []

    def run_reasoner(self):
        """Pellet推論機を実行し、推論結果とその説明（根拠）を返す"""
        if not self.ontology: return {"status": "Error", "message": "Ontology not loaded"}
        
        try:
            # 推論前のクラス所属を記録（オプション、変化を見る場合）
            # pre_inference_state = {}
            # for ind in self.ontology.individuals():
            #    pre_inference_state[ind.name] = [c.name for c in ind.is_a if hasattr(c, "name")]

            with self.ontology:
                # infer_property_values=True でデータプロパティ推論も有効化
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            
            # 推論後の説明レポート作成
            explanation_report = []
            
            # 全個体をスキャンして、推論によって付与された可能性のあるクラスを特定
            # ここでは簡単のため、"is_a" に含まれる各クラスについてルールを探す
            for ind in self.ontology.individuals():
                ind_report = {
                    "name": ind.name,
                    "classes": [],
                    "explanations": {}
                }
                
                current_classes = [c.name for c in ind.is_a if hasattr(c, "name")]
                ind_report["classes"] = current_classes
                
                for cls_name in current_classes:
                    # そのクラスを導く「自然言語の説明」を取得
                    explanation = self.get_explanation_for_class(cls_name)
                    if explanation:
                        # 説明が見つかった場合、リストとして追加（既存のUIコードとの互換性のため）
                        ind_report["explanations"][cls_name] = [explanation]
                
                if ind_report["explanations"]:
                    explanation_report.append(ind_report)

            return {
                "status": "Success", 
                "message": "Reasoning Completed: Rules applied.",
                "report": explanation_report
            }

        except Exception as e:
            return {"status": "Error", "message": f"Reasoning Failed: {e}"}

    def visualize_building(self, building_ind):
        """特定の建築物に関するグラフを生成"""
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        
        # 中心ノード（建築物）
        # 推論されたクラス（例：高層建築物）もラベルに表示
        classes = [c.name for c in building_ind.is_a if hasattr(c, "name")]
        label = f"{building_ind.name}\n({', '.join(classes)})"
        
        # 高層建築物や免震建築物が付与されていたら赤色にする等の視覚化
        color = "lightblue"
        if "高層建築物" in classes: color = "salmon"
        if "免震建築物" in classes: color = "lightgreen"

        dot.node(building_ind.name, label=label, shape='box', style='filled', fillcolor=color)
        
        # プロパティの結合を描画
        for prop in building_ind.get_properties():
            values = prop[building_ind]
            for v in values:
                # オブジェクトプロパティ（リンク）の場合
                if hasattr(v, 'name'): 
                    dot.node(v.name, label=v.name, shape='ellipse', style='filled', fillcolor='lightgrey')
                    dot.edge(building_ind.name, v.name, label=prop.name)
                # データプロパティ（値）の場合
                else:
                    val_node_id = f"{prop.name}_{v}"
                    dot.node(val_node_id, label=f"{v}", shape='plaintext')
                    dot.edge(building_ind.name, val_node_id, label=prop.name)
                    
        return dot

    def save_ontology(self, path="temp_Ontology.owl"):
        if self.ontology:
            self.ontology.save(file=path, format="rdfxml")
            print(f"Ontology saved to {path}")