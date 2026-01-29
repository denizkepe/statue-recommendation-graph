#!/usr/bin/env python3
"""
Interactive Knowledge Graph Visualization.

Creates a state-of-the-art interactive HTML visualization of the legal knowledge graph
using PyVis with customizable node colors, sizes, and edge styling.
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import random

try:
    from pyvis.network import Network
except ImportError:
    print("Installing pyvis...")
    import subprocess
    subprocess.run(["pip", "install", "pyvis"], check=True)
    from pyvis.network import Network


# Color schemes for different node types
COLORS = {
    "case": {
        "ONAMA": "#27ae60",      # Green - Affirmed
        "BOZMA": "#e74c3c",      # Red - Reversed
        "GOREVSIZLIK": "#95a5a6", # Gray - Jurisdiction
        "GERI_CEVIRME": "#f39c12", # Orange - Returned
        "UNKNOWN": "#bdc3c7",    # Light gray
    },
    "statute": "#3498db",       # Blue
    "chamber": "#9b59b6",       # Purple
    "case_type": "#1abc9c",     # Teal
}

# Node shapes
SHAPES = {
    "case": "dot",
    "statute": "diamond",
    "chamber": "square",
    "case_type": "triangle",
}


def load_parsed_data(filepath: str) -> list:
    """Load parsed JSON data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_graph_data(cases: list, max_cases: int = 200, max_statutes: int = 50):
    """
    Build graph data from parsed cases.
    
    Args:
        cases: List of parsed case dictionaries
        max_cases: Maximum number of cases to include (for performance)
        max_statutes: Maximum number of statutes to include
    
    Returns:
        nodes: List of node dictionaries
        edges: List of edge tuples (source, target, attributes)
    """
    nodes = []
    edges = []
    
    # Filter to cases with outcomes and statutes (more interesting for viz)
    valid_cases = [c for c in cases if c.get("statute_ids") and c.get("outcome")]
    
    # Prioritize ONAMA/BOZMA cases for visualization
    priority_cases = [c for c in valid_cases if c.get("outcome") in ["ONAMA", "BOZMA"]]
    other_cases = [c for c in valid_cases if c.get("outcome") not in ["ONAMA", "BOZMA"]]
    
    # Take priority cases first
    if len(priority_cases) >= max_cases:
        selected_cases = random.sample(priority_cases, max_cases)
    else:
        remaining = max_cases - len(priority_cases)
        selected_cases = priority_cases + random.sample(other_cases, min(remaining, len(other_cases)))
    
    # Count statutes to find the most common ones
    all_statutes = []
    for c in selected_cases:
        all_statutes.extend(c.get("statute_ids", []))
    
    statute_counts = Counter(all_statutes)
    top_statutes = set([s for s, _ in statute_counts.most_common(max_statutes)])
    
    # Collect chambers and case types
    chambers = set()
    case_types = set()
    
    # Build case nodes
    for i, case in enumerate(selected_cases):
        case_id = case.get("id", f"case_{i}")
        outcome = case.get("outcome", "UNKNOWN")
        chamber = case.get("chamber", "Unknown")
        case_type = case.get("case_type_enum", "OTHER")
        
        # Truncate plaintiff arguments for tooltip
        args = case.get("plaintiff_arguments", "")[:200] + "..." if len(case.get("plaintiff_arguments", "")) > 200 else case.get("plaintiff_arguments", "")
        
        color = COLORS["case"].get(outcome, COLORS["case"]["UNKNOWN"])
        
        nodes.append({
            "id": case_id,
            "label": case_id.replace("decision_", "")[:10],
            "title": f"<b>{case_id}</b><br>Outcome: {outcome}<br>Chamber: {chamber}<br>Type: {case_type}<br><br>{args}",
            "group": "case",
            "color": color,
            "shape": SHAPES["case"],
            "size": 15,
            "outcome": outcome,
        })
        
        # Track metadata
        if chamber:
            chambers.add(chamber)
        if case_type:
            case_types.add(case_type)
        
        # Create edges to statutes
        for statute in case.get("statute_ids", []):
            if statute in top_statutes:
                edges.append((case_id, f"statute_{statute}", {"color": "#3498db", "width": 1}))
        
        # Create edge to chamber
        if chamber:
            edges.append((case_id, f"chamber_{chamber}", {"color": "#9b59b6", "width": 0.5, "dashes": True}))
        
        # Create edge to case type
        if case_type:
            edges.append((case_id, f"type_{case_type}", {"color": "#1abc9c", "width": 0.5, "dashes": True}))
    
    # Build statute nodes
    for statute in top_statutes:
        count = statute_counts[statute]
        size = min(8 + count * 0.5, 40)  # Scale by usage
        
        nodes.append({
            "id": f"statute_{statute}",
            "label": statute,
            "title": f"<b>Statute: {statute}</b><br>Citations: {count}",
            "group": "statute",
            "color": COLORS["statute"],
            "shape": SHAPES["statute"],
            "size": size,
        })
    
    # Build chamber nodes
    for chamber in chambers:
        nodes.append({
            "id": f"chamber_{chamber}",
            "label": chamber.replace(" Hukuk Dairesi", "").replace(". ", "."),
            "title": f"<b>Chamber: {chamber}</b>",
            "group": "chamber",
            "color": COLORS["chamber"],
            "shape": SHAPES["chamber"],
            "size": 25,
        })
    
    # Build case type nodes
    for case_type in case_types:
        nodes.append({
            "id": f"type_{case_type}",
            "label": case_type,
            "title": f"<b>Case Type: {case_type}</b>",
            "group": "case_type",
            "color": COLORS["case_type"],
            "shape": SHAPES["case_type"],
            "size": 30,
        })
    
    # Add co-citation edges between statutes
    case_statute_map = defaultdict(set)
    for case in selected_cases:
        case_id = case.get("id", "")
        for statute in case.get("statute_ids", []):
            if statute in top_statutes:
                case_statute_map[statute].add(case_id)
    
    # Find commonly co-cited statutes
    statutes_list = list(top_statutes)
    for i, s1 in enumerate(statutes_list):
        for s2 in statutes_list[i+1:]:
            common = len(case_statute_map[s1] & case_statute_map[s2])
            if common >= 3:  # At least 3 cases cite both
                edges.append((
                    f"statute_{s1}", 
                    f"statute_{s2}", 
                    {"color": "#2980b9", "width": min(common / 5, 3), "title": f"Co-cited in {common} cases"}
                ))
    
    return nodes, edges


def create_visualization(
    nodes: list,
    edges: list,
    output_file: str = "results/knowledge_graph.html",
    title: str = "Yargıtay Legal Knowledge Graph",
):
    """
    Create interactive PyVis visualization.
    """
    # Create network
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",  # Dark background
        font_color="white",
        directed=False,
        notebook=False,
    )
    
    # Physics settings for nice layout
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 12,
          "face": "Helvetica"
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "color": {
          "inherit": false
        }
      },
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      }
    }
    """)
    
    # Add nodes
    for node in nodes:
        net.add_node(
            node["id"],
            label=node["label"],
            title=node["title"],
            color=node["color"],
            shape=node["shape"],
            size=node["size"],
            group=node["group"],
        )
    
    # Add edges
    for src, dst, attrs in edges:
        net.add_edge(
            src, dst,
            color=attrs.get("color", "#555"),
            width=attrs.get("width", 1),
            dashes=attrs.get("dashes", False),
            title=attrs.get("title", ""),
        )
    
    # Add legend as HTML
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); 
                padding: 15px; border-radius: 10px; z-index: 1000; font-family: Helvetica;">
        <h3 style="color: white; margin-top: 0;">Legend</h3>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: #27ae60; border-radius: 50%; margin-right: 8px;"></span>
            Case: ONAMA (Affirmed)
        </div>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: #e74c3c; border-radius: 50%; margin-right: 8px;"></span>
            Case: BOZMA (Reversed)
        </div>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: #95a5a6; border-radius: 50%; margin-right: 8px;"></span>
            Case: GOREVSIZLIK
        </div>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: #3498db; transform: rotate(45deg); margin-right: 8px;"></span>
            Statute (Law Article)
        </div>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: #9b59b6; margin-right: 8px;"></span>
            Chamber (Court Division)
        </div>
        <div style="color: white; margin: 5px 0;">
            <span style="display: inline-block; width: 0; height: 0; 
                        border-left: 8px solid transparent; border-right: 8px solid transparent;
                        border-bottom: 15px solid #1abc9c; margin-right: 8px;"></span>
            Case Type
        </div>
        <hr style="border-color: #555;">
        <div style="color: #aaa; font-size: 11px;">
            📌 Drag nodes to rearrange<br>
            🔍 Scroll to zoom<br>
            🖱️ Hover for details
        </div>
    </div>
    """
    
    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8); 
                padding: 15px; border-radius: 10px; z-index: 1000; font-family: Helvetica;">
        <h2 style="color: white; margin: 0;">{title}</h2>
        <p style="color: #aaa; margin: 5px 0 0 0; font-size: 12px;">
            {len([n for n in nodes if n['group'] == 'case'])} cases • 
            {len([n for n in nodes if n['group'] == 'statute'])} statutes • 
            {len([n for n in nodes if n['group'] == 'chamber'])} chambers
        </p>
    </div>
    """
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save and inject custom HTML
    net.save_graph(output_file)
    
    # Read the file and inject legend
    with open(output_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Inject after body tag
    html = html.replace('<body>', f'<body>{legend_html}{title_html}')
    
    # Add custom styles
    custom_css = """
    <style>
        body { margin: 0; overflow: hidden; }
        #mynetwork { border: none !important; }
    </style>
    """
    html = html.replace('</head>', f'{custom_css}</head>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Visualize legal knowledge graph")
    parser.add_argument("--input", "-i", default="data/parsed_all.json",
                       help="Input parsed JSON file")
    parser.add_argument("--output", "-o", default="results/knowledge_graph.html",
                       help="Output HTML file")
    parser.add_argument("--max-cases", type=int, default=200,
                       help="Maximum number of cases to visualize")
    parser.add_argument("--max-statutes", type=int, default=50,
                       help="Maximum number of statutes to include")
    parser.add_argument("--title", default="Yargıtay Legal Knowledge Graph",
                       help="Graph title")
    
    args = parser.parse_args()
    
    print(f"📂 Loading data from {args.input}...")
    cases = load_parsed_data(args.input)
    print(f"   Found {len(cases)} cases")
    
    print(f"🔨 Building graph (max {args.max_cases} cases, {args.max_statutes} statutes)...")
    nodes, edges = build_graph_data(cases, args.max_cases, args.max_statutes)
    print(f"   Created {len(nodes)} nodes and {len(edges)} edges")
    
    print(f"🎨 Creating visualization...")
    output_path = create_visualization(nodes, edges, args.output, args.title)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"   Open in browser: file://{Path(output_path).absolute()}")


if __name__ == "__main__":
    main()
