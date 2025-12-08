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

    def run_reasoner(self):
        """Pellet推論機を実行し、SWRLルールを適用"""
        if not self.ontology: return "Ontology not loaded"
        try:
            with self.ontology:
                # infer_property_values=True でデータプロパティ推論も有効化
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            return "Reasoning Completed: Rules applied."
        except Exception as e:
            return f"Reasoning Failed: {e}"

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