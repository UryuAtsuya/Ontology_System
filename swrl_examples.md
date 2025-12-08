# SWRL Rule Examples for Architectural Ontology

This document provides examples of SWRL (Semantic Web Rule Language) rules that can be applied to the architectural ontology.

## 1. Building Age Classification
**Goal**: Classify a building as "Historic" if it is older than 50 years.

**Rule**:
```
Building(?b) ^ hasAge(?b, ?age) ^ swrlb:greaterThan(?age, 50) -> HistoricBuilding(?b)
```
*Note: You need to define `HistoricBuilding` class and `hasAge` data property.*

## 2. Room Connectivity Inference
**Goal**: If Room A is connected to Room B, and Room B is connected to Room C, infer that Room A is indirectly connected to Room C (Transitive property example, though OWL supports transitive properties natively, this shows SWRL usage).

**Rule**:
```
Room(?r1) ^ isConnectedTo(?r1, ?r2) ^ isConnectedTo(?r2, ?r3) -> isIndirectlyConnectedTo(?r1, ?r3)
```

## 3. Safety Regulation Check
**Goal**: If a room has a capacity greater than 100 but has fewer than 2 exits, mark it as "Unsafe".

**Rule**:
```
Room(?r) ^ hasCapacity(?r, ?cap) ^ swrlb:greaterThan(?cap, 100) ^ hasExitCount(?r, ?exits) ^ swrlb:lessThan(?exits, 2) -> UnsafeRoom(?r)
```

## 4. Material Compatibility
**Goal**: If a wall is made of Wood and is located in a HighFireRiskZone, classify it as "HighRiskStructure".

**Rule**:
```
Wall(?w) ^ hasMaterial(?w, Wood) ^ isLocatedIn(?w, ?zone) ^ HighFireRiskZone(?zone) -> HighRiskStructure(?w)
```
