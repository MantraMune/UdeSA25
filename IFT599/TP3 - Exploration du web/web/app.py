from flask import Flask, render_template, request, flash
import networkx as nx
import time, re, requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import tldextract

app = Flask(__name__)
app.secret_key = "TP3"

HEADERS = {"User-Agent": "Mozilla/5.0 (TP3 Recherche Web; +https://www.usherbrooke.ca)"}
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = .1
MAX_SEEDS = 15
MAX_PAGES = 30
MAX_OUTLINKS_PER_PAGE = 10
CRAWL_DEPTH = 3
K = 10


# ------------------------------------------------------------
# ------------------ UTILITAIRES CRAWLER ----------------------
# ------------------------------------------------------------

def is_probable_html(resp: requests.Response) -> bool:
    return "text/html" in resp.headers.get("Content-Type", "")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="", query=parsed.query)
    return normalized.geturl()


def looks_like_webpage(url: str) -> bool:
    return not re.search(r"\.(pdf|png|jpg|jpeg|gif|svg|zip|rar|tar|mp4|mp3)(\?|$)", url, re.I)


def extract_links(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        if looks_like_webpage(absolute):
            links.add(normalize_url(absolute))
    return list(links)


def domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([part for part in [ext.domain, ext.suffix] if part])


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code == 200 and is_probable_html(resp):
            return resp.text
        return None
    except requests.RequestException:
        return None


# ------------------------------------------------------------
# -------------------- CONSTRUCTION DU GRAPHE ----------------
# ------------------------------------------------------------

def crawl_and_build_graph(seed_urls, max_pages=MAX_PAGES, max_depth=CRAWL_DEPTH):
    G = nx.DiGraph()
    visited = set()
    frontier = [(url, 0) for url in seed_urls[:MAX_SEEDS]]

    while frontier and len(visited) < max_pages:
        url, depth = frontier.pop(0)

        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        html = fetch(url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if html is None:
            continue

        outlinks = extract_links(url, html)[:MAX_OUTLINKS_PER_PAGE]

        G.add_node(url, domain=domain(url))

        for l in outlinks:
            G.add_node(l, domain=domain(l))
            G.add_edge(url, l)
            if l not in visited:
                frontier.append((l, depth + 1))

    return G


# ------------------------------------------------------------
# -------------------- ALGORITHMES DE RANG -------------------
# ------------------------------------------------------------

def compute_pagerank(G):
    return nx.pagerank(G, alpha=0.85)

def compute_hits_authority(G):
    hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    return authorities

def compute_hits_hubs(G):
    hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    return hubs


# ------------------------------------------------------------
# ----------------------- ROUTE PRINCIPALE -------------------
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        seeds_raw = request.form.get("seeds", "").strip()
        query = request.form.get("query", "").strip()
        critere = request.form.get("critere", "PageRank")

        if not seeds_raw or not query:
            flash("Veuillez renseigner au moins un seed et une requête.", "danger")
            return render_template("index.html")

        seeds = [s.strip() for s in seeds_raw.splitlines() if s.strip()]

        # 1. Construire le graphe
        G = crawl_and_build_graph(seeds)

        if len(G.nodes) == 0:
            flash("Impossible de crawler les pages fournies.", "danger")
            return render_template("index.html")

        scores = {}
        times = {}

        # --- Calculer tous les critères si "ALL" ---
        criteres_to_compute = []
        if critere == "ALL":
            criteres_to_compute = ["PageRank", "HITS-autorite", "HITS-hubs"]
        else:
            criteres_to_compute = [critere]

        for c in criteres_to_compute:

            start = time.time()

            if c == "PageRank":
                data = compute_pagerank(G)

            elif c == "HITS-autorite":
                data = compute_hits_authority(G)

            elif c == "HITS-hubs":
                data = compute_hits_hubs(G)

            else:
                continue

            end = time.time()

            scores[c] = sorted(data.items(), key=lambda x: x[1], reverse=True)[:K]
            times[c] = round(end - start, 4)

        results = {
            "query": query,
            "scores": scores,
            "times": times,
            "nb_nodes": len(G.nodes),
            "nb_edges": len(G.edges),
            "critere": critere
        }

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
