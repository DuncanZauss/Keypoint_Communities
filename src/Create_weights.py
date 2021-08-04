from constants import WHOLEBODY_KEYPOINTS, WHOLEBODY_SKELETON, WHOLEBODY_SCORE_WEIGHTS, \
    WHOLEBODY_STANDING_POSE
from constants import SKELETON_CONNECT, CAR_KEYPOINTS, CAR_SKELETON, CAR_SCORE_WEIGHTS, \
    CAR_POSE, HFLIP_ids
import networkx as nx
import numpy as np
import json
import pandas as pd
import os

def get_normalized_weights(centrality_measure, kps):
    list_centralities = []
    for i in range(len(kps)):
        list_centralities.append(centrality_measure[i])
    w = np.array(list_centralities)
    w = w/np.sum(w) * len(kps)
    return w


def inverse_normalize(weights, kps):
    w = 1/weights
    w = w/np.sum(w) * len(kps)
    return w.tolist()


def harmonic_centrality_local_radius(G_w, radius, kps, distance="euclidean_dist"):
    weights = []
    for node in G_w.nodes:
        # print(node)
        subgraph = nx.generators.ego.ego_graph(G_w, n=node, radius=radius)
        centr = nx.harmonic_centrality(subgraph, distance="euclidean_dist")
        weights.append(centr[node]/(len(subgraph.nodes())-1))
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


def draw_skeletons_wb(pose, weights, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from painters import KeypointPainter

    scale = 1.0/25
    KeypointPainter.show_joint_scales = True
    keypoint_painter = KeypointPainter(line_width=1,
                                       monocolor_connections=False)

    ann = Annotation(keypoints=WHOLEBODY_KEYPOINTS,
                     skeleton=WHOLEBODY_SKELETON,
                     score_weights=WHOLEBODY_SCORE_WEIGHTS)
    ann.set(pose, np.array(weights) * scale)
    draw_ann(ann, filename='./'+prefix+'_skeleton_wholebody.png',
             keypoint_painter=keypoint_painter)


def draw_skeletons_apollo(pose, weights, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from painters import KeypointPainter

    scale = 1.0/25
    KeypointPainter.show_joint_scales = True
    keypoint_painter = KeypointPainter(line_width=1,
                                       monocolor_connections=False)

    ann = Annotation(keypoints=CAR_KEYPOINTS,
                     skeleton=CAR_SKELETON,
                     score_weights=CAR_SCORE_WEIGHTS)
    ann.set(pose, np.array(weights) * scale)
    draw_ann(ann, filename='./'+prefix+'_skeleton_apollocar.png',
             keypoint_painter=keypoint_painter)


def rotate(pose, angle=45, axis=2):
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    pose_copy = np.copy(pose)
    pose_copy[:, 2] = pose_copy[:, 2]
    if axis == 0:
        rot_mat = np.array([[1, 0, 0],
                            [0, cos, -sin],
                            [0, sin, cos]])
    elif axis == 1:
        rot_mat = np.array([[cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
    elif axis == 2:
        rot_mat = np.array([[cos, -sin, 0],
                            [sin, cos, 0],
                            [0, 0, 1]])
    else:
        raise Exception("Axis must be 0,1 or 2 (corresponding to x,y,z).")
    rotated_pose = np.transpose(np.matmul(rot_mat, np.transpose(pose_copy)))
    rotated_pose[:, 2] = rotated_pose[:, 2] + 4
    return rotated_pose


def create_weights_wholebody():
    name = "wb"

    skel = [(bone[0]-1, bone[1]-1) for bone in WHOLEBODY_SKELETON]
    kps = WHOLEBODY_KEYPOINTS

    G = nx.Graph()

    G.add_nodes_from(range(len(kps)))
    G.add_edges_from(skel)

    with open("Edge_weights_train_wb.json", 'r') as f:
        edge_weights = json.load(f)

    G_w = nx.Graph()
    G_w.add_nodes_from(range(len(kps)))

    for bone_id, bone in enumerate(skel):
        G_w.add_edge(bone[0], bone[1], euclidean_dist=edge_weights[bone_id])
        G_w.add_edge(bone[0], bone[1], euclidean_dist_inverse=1/edge_weights[bone_id])

    G_synthetic = nx.Graph()
    G_synthetic.add_nodes_from(range(len(kps)))

    for bone_id, bone in enumerate(skel):
        dist_bone = np.linalg.norm(WHOLEBODY_STANDING_POSE[bone[0], :2] -
                                   WHOLEBODY_STANDING_POSE[bone[1], :2])
        G_synthetic.add_edge(bone[0], bone[1], euclidean_dist=dist_bone)

    w_cl = get_normalized_weights(nx.closeness_centrality(G), kps=kps)
    w_cl_euclid = get_normalized_weights(nx.closeness_centrality(G_w, distance="euclidean_dist"),
                                         kps=kps)
    w_harm_cl = get_normalized_weights(nx.harmonic_centrality(G), kps=kps)
    w_harm_cl_euclid = get_normalized_weights(
        nx.harmonic_centrality(G_w, distance="euclidean_dist"), kps=kps)

    w_harm_euclid_radius_1 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=1, kps=kps,
                                         distance="euclidean_dist"), kps=kps)
    w_harm_euclid_radius_2 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=2, kps=kps,
                                         distance="euclidean_dist"), kps=kps)
    w_harm_euclid_radius_3 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=3, kps=kps,
                                         distance="euclidean_dist"), kps=kps)

    w_harm_euclid_radius_3_synthetic = get_normalized_weights(
        harmonic_centrality_local_radius(G_synthetic, radius=3, kps=kps,
                                         distance="euclidean_dist"),
        kps=kps)
    w_harm_cl_euclid_synthetic = get_normalized_weights(
        nx.harmonic_centrality(G_synthetic, distance="euclidean_dist"), kps=kps)

    hand_crafted = np.array(23*[3.0] + 110*[1.0])
    hand_crafted = hand_crafted/np.sum(hand_crafted) * len(hand_crafted)  # normalize

    results = {"keypoints": kps,
               "centrality_closeness_inverse": inverse_normalize(w_cl, kps=kps),
               "centrality_closeness_euclid_inverse": inverse_normalize(w_cl_euclid, kps=kps),
               "centrality_harmonic_inverse": inverse_normalize(w_harm_cl, kps=kps),
               "centrality_harmonic_euclid_inverse": inverse_normalize(w_harm_cl_euclid, kps=kps),
               "w_harm_cl_euclid_synthetic": inverse_normalize(w_harm_cl_euclid_synthetic,
                                                               kps=kps),
               "w_harm_euclid_radius_1": inverse_normalize(w_harm_euclid_radius_1, kps=kps),
               "w_harm_euclid_radius_2": inverse_normalize(w_harm_euclid_radius_2, kps=kps),
               "w_harm_euclid_radius_3": inverse_normalize(w_harm_euclid_radius_3, kps=kps),
               "w_harm_euclid_radius_3_synthetic": inverse_normalize(
                   w_harm_euclid_radius_3_synthetic, kps=kps),
               "hand_crafted": list(hand_crafted),
               }

    if not os.path.isdir("docs_wb"):
        os.makedirs("docs_wb")

    WHOLEBODY_STANDING_POSE[:, 2] = 1.0
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_cl_euclid, kps=kps),
                      prefix="docs_wb/centrality_harmonic_euclid_global_inverse")
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_3, kps=kps),
                      prefix="docs_wb/w_harm_euclid_radius_3")
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_1, kps=kps),
                      prefix="docs_wb/w_harm_euclid_radius_1")
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_2, kps=kps),
                      prefix="docs_wb/w_harm_euclid_radius_2")
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_cl_euclid_synthetic,
                                                                 kps=kps),
                      prefix="docs_wb/w_harm_cl_euclid_synthetic")
    draw_skeletons_wb(WHOLEBODY_STANDING_POSE, inverse_normalize(w_harm_euclid_radius_3_synthetic,
                                                                 kps=kps),
                      prefix="docs_wb/w_harm_euclid_radius_3_synthetic")

    with open("Weights_"+name+".json", 'w') as f:
        json.dump(results, f)

    df = pd.read_json("Weights_"+name+".json")
    df.to_csv("Weights_"+name+".csv", index=None, header=True)
    print("Compututed weights written to: Weights_"+name+".csv")


def create_weights_apollo():
    name = "apollocar"

    skel = [(bone[0]-1, bone[1]-1) for bone in CAR_SKELETON]
    kps = CAR_KEYPOINTS

    G = nx.Graph()

    G.add_nodes_from(range(len(kps)))
    G.add_edges_from(skel)

    with open("Edge_weights_train_apollocar.json", 'r') as f:
        edge_weights = json.load(f)

    G_w = nx.Graph()
    G_w.add_nodes_from(range(len(kps)))

    for bone_id, bone in enumerate(skel):
        G_w.add_edge(bone[0], bone[1], euclidean_dist=edge_weights[bone_id])
        G_w.add_edge(bone[0], bone[1], euclidean_dist_inverse=1/edge_weights[bone_id])

    G_synthetic = nx.Graph()
    G_synthetic.add_nodes_from(range(len(kps)))

    for bone_id, bone in enumerate(skel):
        dist_bone = np.linalg.norm(CAR_POSE[bone[0]]-CAR_POSE[bone[1]])
        G_synthetic.add_edge(bone[0], bone[1], euclidean_dist=dist_bone)

    w_cl = get_normalized_weights(nx.closeness_centrality(G), kps=kps)
    w_cl_euclid = get_normalized_weights(
        nx.closeness_centrality(G_w, distance="euclidean_dist"), kps=kps)
    w_harm_cl = get_normalized_weights(nx.harmonic_centrality(G), kps=kps)
    w_harm_cl_euclid = get_normalized_weights(
        nx.harmonic_centrality(G_w, distance="euclidean_dist"), kps=kps)

    w_harm_euclid_radius_1 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=1, kps=kps,
                                         distance="euclidean_dist"), kps=kps)
    w_harm_euclid_radius_2 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=2, kps=kps,
                                         distance="euclidean_dist"), kps=kps)
    w_harm_euclid_radius_3 = get_normalized_weights(
        harmonic_centrality_local_radius(G_w, radius=3, kps=kps,
                                         distance="euclidean_dist"), kps=kps)

    w_harm_euclid_radius_3_synthetic = get_normalized_weights(
        harmonic_centrality_local_radius(G_synthetic, radius=3, kps=kps,
                                         distance="euclidean_dist"), kps=kps)
    w_harm_cl_euclid_synthetic = get_normalized_weights(
        nx.harmonic_centrality(G_synthetic, distance="euclidean_dist"), kps=kps)

    results = {"keypoints": kps,
               "centrality_closeness_inverse": inverse_normalize(w_cl, kps=kps),
               "centrality_closeness_euclid_inverse": inverse_normalize(w_cl_euclid, kps=kps),
               "centrality_harmonic_inverse": inverse_normalize(w_harm_cl, kps=kps),
               "centrality_harmonic_euclid_inverse": inverse_normalize(w_harm_cl_euclid, kps=kps),
               "w_harm_cl_euclid_synthetic": inverse_normalize(w_harm_cl_euclid_synthetic,
                                                               kps=kps),
               "w_harm_euclid_radius_1": inverse_normalize(w_harm_euclid_radius_1, kps=kps),
               "w_harm_euclid_radius_2": inverse_normalize(w_harm_euclid_radius_2, kps=kps),
               "w_harm_euclid_radius_3": inverse_normalize(w_harm_euclid_radius_3, kps=kps),
               "w_harm_euclid_radius_3_synthetic": inverse_normalize(
                   w_harm_euclid_radius_3_synthetic, kps=kps)
               }

    rot = rotate(CAR_POSE, angle=-70, axis=1)
    top_view = rotate(rot, angle=25, axis=0)

    if not os.path.isdir("docs_apollocar"):
        os.makedirs("docs_apollocar")

    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_cl_euclid, kps=kps),
                          prefix="docs_apollocar/w_harm_cl_euclid")
    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_euclid_radius_3, kps=kps),
                          prefix="docs_apollocar/w_harm_euclid_radius_3")
    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_cl_euclid_synthetic, kps=kps),
                          prefix="docs_apollocar/w_harm_cl_euclid_synthetic")
    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_euclid_radius_3_synthetic, kps=kps),
                          prefix="docs_apollocar/w_harm_euclid_radius_3_synthetic")

    connect_kps = [item for sublist in SKELETON_CONNECT for item in sublist]
    for i in HFLIP_ids:
        if i not in connect_kps:
            top_view[i, 2] = 0.5

    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_cl_euclid, kps=kps),
                          prefix="docs_apollocar/Dotted_w_harm_cl_euclid")
    draw_skeletons_apollo(top_view, inverse_normalize(w_harm_euclid_radius_3, kps=kps),
                          prefix="docs_apollocar/Dotted_w_harm_euclid_radius_3")

    with open("Weights_"+name+".json", 'w') as f:
        json.dump(results, f)

    df = pd.read_json("Weights_"+name+".json")
    df.to_csv("Weights_"+name+".csv", index=None, header=True)
    print("Compututed weights written to: Weights_"+name+".csv")


if __name__ == '__main__':
    create_weights_wholebody()
    create_weights_apollo()
