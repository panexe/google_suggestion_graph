import requests
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
import argparse

# Idea from David Forster, Applied Data Science 
# https://medium.com/applied-data-science/the-google-vs-trick-618c8fd5359f

def getSuggestions(search_term, geo_code = "de", language="de"):
    url = f'http://suggestqueries.google.com/complete/search?output=toolbar&gl={geo_code}&hl={language}&q={search_term}'
    result = requests.get(url)
    if(result.ok):
        text = result.text
        tree = ET.fromstring(text)
        suggestions = []
        for child in tree:
            for c in child:
                suggestions.append(c.attrib["data"])
        return suggestions

def cleanSuggestions(suggestions, search_query, max_result_words=2, n_results=5, filter_first_word=False):
    n_query_words = len(search_query.split(" "))
    query_split = search_query.split(" ")
    ret = []
    result_counter = 0
    
    for s in suggestions:
        elems = s.split(" ")
        start_pos = n_query_words
        if start_pos > len(elems):
            start_pos = len(elems - 1)
        elems = elems[start_pos:]
        
        # drop criteria
        if len(elems) > max_result_words:
            continue
        elif ''.join(elems) == '':
            continue
        elif (query_split[0] in " ".join(elems)) and filter_first_word:
            continue
        elif any(r in " ".join(elems) for r in ret ):
            continue
        else: 
            ret.append(" ".join(elems))
            result_counter += 1 
            if result_counter >= n_results:
                break
            
    return ret 

def top5(search_query, iterations=20, max_result_words=2, n_results=5, filter_first_word=True, compare_str=" vs", geo_code="de", language="de"):
    res = []
    already_searched = [search_query]
    suggestions = getSuggestions(search_query+compare_str, geo_code=geo_code, language=language)
    t5 = cleanSuggestions(suggestions, search_query+compare_str, max_result_words=max_result_words, n_results=n_results, filter_first_word=filter_first_word)
    for i,t in enumerate(t5):
        res.append({"search_query":search_query, "target":t, "weight":len(t5)-i, "original_search_query":search_query})

    if len(res) == 0:
        print("Error, no mathing results for this setting")
        exit()

    for i in range(iterations):
        current_term = res[i]["target"]
        if current_term in already_searched:
            i = i - 1
            continue
        else:
            already_searched.append(current_term)
        t5_ = cleanSuggestions(getSuggestions(current_term+compare_str),current_term+compare_str, max_result_words=max_result_words, n_results=n_results, filter_first_word=filter_first_word)
        for i,t in enumerate(t5_):
            res.append({"search_query":current_term, "target":t, "weight":len(t5)-i, "original_search_query":search_query})
    
    return res

def createGraph(query, filename, iterations=20, max_result_words=2, n_results=5, filter_first_word=True, compare_str=" vs", geo_code="de", language="de", font_color="orange"):
    t5 =top5(query, 
            iterations, 
            max_result_words, 
            n_results, 
            filter_first_word, 
            compare_str, 
            geo_code, 
            language)

    
    edge_list = []
    for n in t5 : 
        edge_list.append((n["search_query"], n["target"], n["weight"]))
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(edge_list)
    
    UG = G.to_undirected()
    for node in G:
        for ngbr in nx.neighbors(G, node):
            if node in nx.neighbors(G, ngbr):
                UG.edges[node, ngbr]["weight"] = (11-G.edges[node, ngbr]["weight"] + G.edges[ngbr, node]["weight"])
    
    # get size of each bubble
    targets = [t5[0]["original_search_query"]]
    for e in t5:
        targets.append(e["target"])
    hist = Counter(targets)
    
    for c in hist:
        UG.nodes[c]["size"] = hist[c] 
        
    line_width = [0.1* UG[u][v]["weight"] for u,v in UG.edges()]
    node_size = [800* nx.get_node_attributes(UG, "size")[v] for v in UG ]
    pos = nx.spring_layout(UG, k=0.8, iterations=100, scale=40)
    node_color = [G.degree(v) for v in G]

    plt.figure(3,figsize=(16,16)) 
    nx.draw_networkx(UG,pos,width=line_width,node_size=node_size,node_color=node_color,font_size=14,font_color=font_color, with_labels = True)
    plt.savefig(filename)


parser = argparse.ArgumentParser(description='Create ego-graph from google search suggestions')

parser.add_argument("query", help="the term you search for e.g. \"[tensorflow] vs\"", type=str)
parser.add_argument("filename", help="name of the file the result is saved in", type=str)
parser.add_argument('--max_result_words', help="Amount of words the target can contain", default=2, type=int)
parser.add_argument('--n_results', help="Amount of targets for each query", default=5)
parser.add_argument('--filter_first_word', help="Filters targets containing the query", default=True)
parser.add_argument('--compare_str', help="This is appended to the query e.g. \"tensorflow[ vs]\"", default=" vs")
parser.add_argument('--geocode', help="Geocode for google api", default="de")
parser.add_argument('--lang', help="Language for google api", default="de")
parser.add_argument('--font-color', help="Color of the labels", default="orange")
parser.add_argument('--iterations', help="Number of searches for resulting targets", default=20)
parser.add_argument('--no-compare', action="store_true", help="Just a normal map of topics no vs.")

results = parser.parse_args()

query = results.query 
filename = results.filename
iterations = results.iterations
max_result_words = results.max_result_words
n_results = results.n_results
filter_first_word = results.filter_first_word
compare_str = results.compare_str
if results.no_compare:
    compare_str = ""
geo_code = results.geocode
language = results.lang
font_color = results.font_color

createGraph(query,
            filename, 
            iterations,
            max_result_words, 
            n_results, 
            filter_first_word, 
            compare_str,
            geo_code, 
            language, 
            font_color)

