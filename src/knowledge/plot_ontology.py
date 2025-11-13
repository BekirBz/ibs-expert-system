# Draw ontology/knowledge graph: food classes -> triggers (bipartite graph)

from pathlib import Path
import csv
import networkx as nx
from src.utils.paths import DATA_INTERIM, REPORT_FIG

def main():
    # read selected classes and their triggers
    mp = {}
    with open(DATA_INTERIM / "trigger_mapping.csv") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cls = row["class"]
            mp[cls] = {k:int(row[k]) for k in ["gluten","lactose","caffeine","highfat","fodmap"]}

    G = nx.Graph()
    triggers = ["gluten","lactose","caffeine","highfat","fodmap"]

    # add nodes
    for c in mp.keys():
        G.add_node(c, bipartite=0)
    for t in triggers:
        G.add_node(t, bipartite=1)

    # add edges
    for c, vec in mp.items():
        for t, v in vec.items():
            if v == 1:
                G.add_edge(c, t)

    # layout (bipartite)
    top = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    bottom = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
    pos = {}
    pos.update((n, (0, i)) for i, n in enumerate(top))
    pos.update((n, (1, i)) for i, n in enumerate(bottom))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, max(6, len(top)*0.35)))
    nx.draw(G, pos,
            with_labels=True,
            node_size=800,
            font_size=8)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIG / "rq2_ontology_graph.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", REPORT_FIG / "rq2_ontology_graph.png")

if __name__ == "__main__":
    main()