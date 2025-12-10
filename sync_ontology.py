
import os
from modules.ontology_manager import OntologyManager

def get_building_info_from_owl():
    """Load building info from temp_Ontology.owl"""
    mgr = OntologyManager("temp_Ontology.owl")
    if not mgr.ontology:
        print("Failed to load ontology.")
        return []

    buildings = []
    # Get all individuals that are instances of '建築物' or its subclasses
    # Note: mgr.get_building_class() returns the class. 
    # We can iterate over all individuals and check if they are buildings.
    building_cls = mgr.get_building_class()
    if not building_cls:
        return []

    for ind in building_cls.instances():
        # Safely get properties
        def get_prop(pname):
            props = ind.get_properties()
            for p in props:
                if p.name.endswith(pname) or p.name == pname:
                    vals = p[ind]
                    if vals:
                        # Return first value string
                        v = vals[0]
                        if hasattr(v, 'name'): return v.name
                        return str(v)
            return "-"

        # Collect attributes matching the text file columns
        # Columns: ID, 名称, 所在, 構造, 用途, 年, 階数, 高さ(m), 技術/装置・等級, 備考
        
        # ID is strict name
        b_id = ind.name
        
        # Name
        name = get_prop("名称")
        if name == "-": name = b_id # Fallback
        
        # Location
        loc = get_prop("場所にある")
        
        # Structure (e.g. S_v -> S)
        # Note: '構造種別を持つ' usually points to an individual like 'S_v'
        struct_raw = get_prop("構造種別を持つ")
        struct = struct_raw.replace("_v", "") if struct_raw != "-" else "-"
        
        # Usage
        usage_raw = get_prop("用途を持つ")
        usage = usage_raw.replace("_v", "").replace("用途", "") if usage_raw != "-" else "-"
        
        # Year
        year = get_prop("建築年")
        
        # Floors
        floors = get_prop("階数")
        
        # Height
        height = get_prop("高さ_m")
        
        # Tech/Device/Grade
        # Combine '耐震技術を持つ', '装置を有する', '重要度区分を持つ' etc.
        techs = []
        
        # Tech
        tech_props = ind.get_properties()
        for p in tech_props:
             if "耐震技術を持つ" in p.name:
                 for v in p[ind]:
                     if hasattr(v, 'name'): techs.append(v.name.replace("_v", "").replace("構造", ""))
        
        # Device
        for p in tech_props:
             if "装置を有する" in p.name:
                 for v in p[ind]:
                     if hasattr(v, 'name'): techs.append(f"装置:{v.name}")

        tech_str = ", ".join(list(set(techs))) if techs else "-"
        
        # Remarks (Rules implied classes like 2000基準, 高層)
        # We can simulate this by checking classes
        classes = [c.name for c in ind.is_a if hasattr(c, 'name')]
        remarks = []
        if "2000基準_v" in classes or "2000基準" in classes: remarks.append("2000基準")
        if "高層建築物" in classes: remarks.append("高層")
        if "旧耐震_v" in classes: remarks.append("旧耐震")
        
        remark_str = ", ".join(remarks) if remarks else "-"

        buildings.append({
            "ID": b_id,
            "名称": name,
            "所在": loc,
            "構造": struct,
            "用途": usage,
            "年": year,
            "階数": floors,
            "高さ(m)": height,
            "技術/装置・等級": tech_str,
            "備考": remark_str
        })
        
    return buildings

def sync_files():
    txt_path = "Updated_Ontology.txt"
    if not os.path.exists(txt_path):
        print(f"{txt_path} not found.")
        return

    # Read existing IDs from text file
    existing_ids = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Find data section start
    data_start_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "ID": # Header line
            data_start_idx = i
            break
            
    if data_start_idx != -1:
        # Scan subsequent lines for IDs
        # The file format seems to be tab-separated or line-separated blocks?
        # Looking at view_file output:
        # 73: ID
        # 74: 	名称
        # ...
        # 83: 	Google東京本社
        # It seems it's NOT a standard CSV table. It's a vertical listing? 
        # Wait, let's re-examine Updated_Ontology.txt format carefully.
        # Lines 73-82 are headers.
        # Lines 83-92 are one entry? 
        # 83: Google東京本社 (This looks like ID)
        # 84: Google東京本社（渋谷ストリーム内） (Name)
        # ...
        # It seems strictly line-based blocks of 10 lines per building?
        # Let's check a few blocks.
        # Block 1: 83-92 (10 lines)
        # Block 2: 93-102 (10 lines)
        # Yes, it looks like blocks of 10 lines following the header.
        
        # However, to be safe, let's rely on exact IDs found in the file.
        # The ID line is the first line of each block.
        # Or simpler: Just read the whole content and check if ID exists in it.
        pass

    content = "".join(lines)
    
    owl_buildings = get_building_info_from_owl()
    new_buildings = []
    
    for b in owl_buildings:
        if b["ID"] not in content:
            new_buildings.append(b)
            
    if not new_buildings:
        print("No new buildings to add.")
        return

    print(f"Found {len(new_buildings)} new buildings. Appending...")
    
    with open(txt_path, "a", encoding="utf-8") as f:
        for b in new_buildings:
            # Append in the vertical format matching the file
            # Assuming 10 fields order: ID, Name, Loc, Struct, Usage, Year, Floors, Height, Tech, Remarks
            f.write(f"\t{b['ID']}\n")
            f.write(f"\t{b['名称']}\n")
            f.write(f"\t{b['所在']}\n")
            f.write(f"\t{b['構造']}\n")
            f.write(f"\t{b['用途']}\n")
            f.write(f"\t{b['年']}\n")
            f.write(f"\t{b['階数']}\n")
            f.write(f"\t{b['高さ(m)']}\n")
            f.write(f"\t{b['技術/装置・等級']}\n")
            f.write(f"\t{b['備考']}\n")

    print("Sync complete.")

if __name__ == "__main__":
    sync_files()
