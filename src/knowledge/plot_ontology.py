# Draw ontology/knowledge graph: food classes -> triggers (bipartite graph)

from pathlib import Path
import csv
import networkx as nx
import matplotlib.pyplot as plt

from src.utils.paths import DATA_INTERIM, REPORT_FIG

TRIGGERS = ["gluten", "lactose", "caffeine", "highfat", "fodmap"]


def build_graph_from_csv(csv_path: Path) -> tuple[nx.Graph, list, list]:
    """
    Build a bipartite graph from trigger_mapping.csv.

    Left side:  food classes
    Right side: trigger nodes (gluten, lactose, etc.)
    """
    mp = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cls = row["class"]
            mp[cls] = {k: int(row[k]) for k in TRIGGERS}

    G = nx.Graph()

    # add nodes
    for c in mp.keys():
        G.add_node(c, bipartite=0)
    for t in TRIGGERS:
        G.add_node(t, bipartite=1)

    # add edges
    for c, vec in mp.items():
        for t, v in vec.items():
            if v == 1:
                G.add_edge(c, t)

    top = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    bottom = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
    return G, top, bottom


def draw_ontology_graph(G: nx.Graph, top: list, bottom: list, save_path: Path, dpi: int = 180) -> None:
    """
    Draw the ontology graph with a simple bipartite layout and save to disk.
    """
    # bipartite layout: foods on the left, triggers on the right
    pos = {}
    pos.update((n, (0, i)) for i, n in enumerate(top))
    pos.update((n, (1, i)) for i, n in enumerate(bottom))

    fig = plt.figure(figsize=(8, max(6, len(top) * 0.35)))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        font_size=8,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved ontology graph:", save_path)


def plot_ontology_graph(out_dir: Path | None = None, suffix: str = "png", dpi: int = 180) -> None:
    """
    Wrapper used by export_results.py

    Parameters
    ----------
    out_dir : Path or None
        Target directory for the figure. If None, uses REPORT_FIG.
    suffix : str
        File extension, e.g. "png" or "pdf".
    dpi : int
        Output resolution (dots per inch).
    """
    csv_path = DATA_INTERIM / "trigger_mapping.csv"
    if out_dir is None:
        out_dir = REPORT_FIG

    out_dir.mkdir(parents=True, exist_ok=True)
    G, top, bottom = build_graph_from_csv(csv_path)
    out_path = out_dir / f"rq2_ontology_graph.{suffix}"
    draw_ontology_graph(G, top, bottom, out_path, dpi=dpi)


def main():
    """
    CLI entry point, used by:  python -m src.knowledge.plot_ontology
    Generates the default PNG ontology figure under reports/figures.
    """
    plot_ontology_graph(out_dir=REPORT_FIG, suffix="png", dpi=180)


if __name__ == "__main__":
    main()