import networkx as nx
import numpy as np
import json
import pandas as pd
#from openpifpaf.plugins.openpifpaf_wholebody.constants import WHOLEBODY_KEYPOINTS, WHOLEBODY_SKELETON, WHOLEBODY_SCORE_WEIGHTS, WHOLEBODY_STANDING_POSE, draw_skeletons as draw_skel_raw
from constants import WHOLEBODY_KEYPOINTS, WHOLEBODY_SKELETON, WHOLEBODY_SCORE_WEIGHTS, WHOLEBODY_STANDING_POSE


def harmonic_centrality_local(G_w, distance="euclidean_dist"):
    weights = []
    for node in G_w.nodes:
        neighborhood = [node] + [n for n in G_w.neighbors(node)]
        centr = nx.harmonic_centrality(G_w.subgraph(neighborhood), distance="euclidean_dist")
        weights.append(centr[node]/(len(neighborhood)-1))
    w = np.array(weights)
    w = w/np.sum(w) * len(kps)
    return w

def harmonic_centrality_local_radius(G_w, radius, distance="euclidean_dist"):
    weights = []
    for node in G_w.nodes:
        # print(node)
        subgraph = nx.generators.ego.ego_graph(G_w, n = node, radius=radius)
        centr = nx.harmonic_centrality(subgraph, distance="euclidean_dist")
        weights.append(centr[node]/(len(subgraph.nodes())-1))
# =============================================================================
#         if radius==3:
#             WHOLEBODY_STANDING_POSE[:,2] = 0.0
#             WHOLEBODY_STANDING_POSE[subgraph.nodes(),2] = 1.0
#             draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_local), prefix="./Ego_graphs/Node_"+str(node)+"_weighted")
#             draw_skel_raw(WHOLEBODY_STANDING_POSE, prefix="Node_"+str(node)+"_viz")
# =============================================================================
    w = np.array(weights)
    w = w/np.sum(w) * len(kps)    
    return w

def closeness_centrality_local(G_w, distance="euclidean_dist"):
    weights = []
    for node in G_w.nodes:
        neighborhood = [node] + [n for n in G_w.neighbors(node)]
        centr = nx.closeness_centrality(G_w.subgraph(neighborhood), distance="euclidean_dist")
        weights.append(centr[node]/(len(neighborhood)-1))
    w = np.array(weights)
    w = w/np.sum(w) * len(kps)
    return w

def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0]) + 0.5

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)

def draw_skeletons(pose, weights, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    #from openpifpaf import show  # pylint: disable=import-outside-toplevel
    from painters import KeypointPainter

    scale = 1.0/25
    
    KeypointPainter.show_joint_scales = True
    keypoint_painter = KeypointPainter(line_width=1,
                                       monocolor_connections=False)

    ann = Annotation(keypoints=WHOLEBODY_KEYPOINTS,
                     skeleton=WHOLEBODY_SKELETON,
                     score_weights=WHOLEBODY_SCORE_WEIGHTS)
    ann.set(pose, np.array(weights) * scale)
    draw_ann(ann, filename='./'+prefix+'_skeleton_wholebody.png', keypoint_painter=keypoint_painter)

def get_normalized_weights(centrality_measure):
    l = []
    for i in range(len(kps)):
        l.append(centrality_measure[i])
    w = np.array(l)
    w = w/np.sum(w) * len(kps)
    return w

def inverse_normalize(weights):
    w = 1/weights
    w = w/np.sum(w) * len(kps)
    return w.tolist()
    
name = "wb"

skel = [(bone[0]-1, bone[1]-1) for bone in WHOLEBODY_SKELETON] # Python fromat --> start at zero
kps = WHOLEBODY_KEYPOINTS

G = nx.Graph()

G.add_nodes_from(range(len(kps)))
G.add_edges_from(skel)

with open("Edge_weights_train_wb.json", 'r') as f:
    edge_weights = json.load(f)
    
G_w = nx.Graph()
G_w.add_nodes_from(range(len(kps)))

for bone_id, bone in enumerate(skel):
    G_w.add_edge(bone[0], bone[1], euclidean_dist = edge_weights[bone_id])
    G_w.add_edge(bone[0], bone[1], euclidean_dist_inverse = 1/edge_weights[bone_id])
 
G_synthetic = nx.Graph()
G_synthetic.add_nodes_from(range(len(kps)))

for bone_id, bone in enumerate(skel):
    dist_bone = np.linalg.norm(WHOLEBODY_STANDING_POSE[bone[0],:2]-WHOLEBODY_STANDING_POSE[bone[1],:2])
    G_synthetic.add_edge(bone[0], bone[1], euclidean_dist = dist_bone)
    #G_w.add_edge(bone[0], bone[1], euclidean_dist_inverse = 1/edge_weights[bone_id])

w_bt = get_normalized_weights(nx.betweenness_centrality(G))
w_co = get_normalized_weights(nx.degree_centrality(G))
w_cl = get_normalized_weights(nx.closeness_centrality(G))
w_cl_euclid = get_normalized_weights(nx.closeness_centrality(G_w, distance="euclidean_dist"))
w_harm_cl = get_normalized_weights(nx.harmonic_centrality(G))
w_harm_cl_euclid = get_normalized_weights(nx.harmonic_centrality(G_w, distance="euclidean_dist"))
# w_eigenvector_euclid = get_normalized_weights(nx.eigenvector_centrality(G_w, max_iter=1000, tol=1.0e-6, nstart=None, weight="euclidean_dist_inverse"))
w_harm_euclid_local = get_normalized_weights(harmonic_centrality_local(G_w, distance="euclidean_dist"))
w_closeness_euclid_local = get_normalized_weights(closeness_centrality_local(G_w, distance="euclidean_dist"))

w_harm_euclid_radius_1 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=1, distance="euclidean_dist"))
w_harm_euclid_radius_2 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=2, distance="euclidean_dist"))
w_harm_euclid_radius_3 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=3, distance="euclidean_dist"))

w_harm_euclid_radius_3_synthetic = get_normalized_weights(harmonic_centrality_local_radius(G_synthetic, radius=3, distance="euclidean_dist"))
w_harm_cl_euclid_synthetic = get_normalized_weights(nx.harmonic_centrality(G_synthetic, distance="euclidean_dist"))

hand_crafted = np.array(23*[3.0] + 110*[1])
hand_crafted = hand_crafted/np.sum(hand_crafted) * len(hand_crafted)

results = {"keypoints": kps,
           #"centrality_closeness": w_cl.tolist(),
           "centrality_closeness_inverse": inverse_normalize(w_cl),
           #"centrality_closeness_euclid": w_cl_euclid.tolist(),
           "centrality_closeness_euclid_inverse": inverse_normalize(w_cl_euclid),
           "centrality_harmonic_inverse": inverse_normalize(w_harm_cl),
           "centrality_harmonic_euclid_inverse": inverse_normalize(w_harm_cl_euclid),
           "w_harm_cl_euclid_synthetic": inverse_normalize(w_harm_cl_euclid_synthetic),
           "w_harm_euclid_local": inverse_normalize(w_harm_euclid_local),
           "w_closeness_euclid_local": inverse_normalize(w_closeness_euclid_local),
           "w_harm_euclid_radius_1": inverse_normalize(w_harm_euclid_radius_1),
           "w_harm_euclid_radius_2": inverse_normalize(w_harm_euclid_radius_2),
           "w_harm_euclid_radius_3": inverse_normalize(w_harm_euclid_radius_3),
           "w_harm_euclid_radius_3_synthetic": inverse_normalize(w_harm_euclid_radius_3_synthetic),
           "hand_crafted": list(hand_crafted),
           }

WHOLEBODY_STANDING_POSE[:,2] = 1.0
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_cl_euclid), prefix="centrality_harmonic_euclid_global_inverse")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_local), prefix="w_harm_euclid_local")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_closeness_euclid_local), prefix="w_closeness_euclid_local")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_3), prefix="w_harm_euclid_radius_3")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_1), prefix="w_harm_euclid_radius_1")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_2), prefix="w_harm_euclid_radius_2")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_cl_euclid_synthetic), prefix="w_harm_cl_euclid_synthetic")
draw_skeletons(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_3_synthetic), prefix="w_harm_euclid_radius_3_synthetic")

with open("Weights_"+name+".json", 'w') as f:
        json.dump(results, f)

df = pd.read_json("Weights_"+name+".json")
export_csv = df.to_csv("Weights_"+name+".csv", index = None, header=True)