from constants import SKELETON_CONNECT, CAR_KEYPOINTS, CAR_SKELETON, CAR_SCORE_WEIGHTS, CAR_POSE, HFLIP_ids
import networkx as nx
import numpy as np
import json
import pandas as pd

from openpifpaf.configurable import Configurable

import math

import logging

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None

LOG = logging.getLogger(__name__)

# =============================================================================
# def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
#     from openpifpaf import show  # pylint: disable=import-outside-toplevel
# 
#     bbox = ann.bbox()
#     xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
#     ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
#     if aspect == 'equal':
#         fig_w = 5.0
#     else:
#         fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])
# 
#     with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
#         ax.set_axis_off()
#         ax.set_xlim(*xlim)
#         ax.set_ylim(*ylim)
# 
#         if aspect is not None:
#             ax.set_aspect(aspect)
# 
#         keypoint_painter.annotation(ax, ann)
# 
# def draw_skeletons(pose, weights, prefix=""):
#     from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
#     from openpifpaf import show  # pylint: disable=import-outside-toplevel
# 
#     scale = np.sqrt(
#         (np.max(pose[:, 0]) - np.min(pose[:, 0]))
#         * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
#     ) / 50
# 
#     show.KeypointPainter.show_joint_scales = True
#     keypoint_painter = show.KeypointPainter(line_width=2)
# 
#     ann = Annotation(keypoints=WHOLEBODY_KEYPOINTS,
#                      skeleton=WHOLEBODY_SKELETON,
#                      score_weights=WHOLEBODY_SCORE_WEIGHTS)
#     ann.set(pose, np.array(weights) * scale)
#     draw_ann(ann, filename='./'+prefix+'_skeleton_wholebody.png', keypoint_painter=keypoint_painter)
# =============================================================================

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

def harmonic_centrality_local_radius(G_w, radius, distance="euclidean_dist"):
    weights = []
    for node in G_w.nodes:
        subgraph = nx.generators.ego.ego_graph(G_w, n = node, radius=radius)
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

def draw_skeletons(pose, weights, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    #from openpifpaf import show  # pylint: disable=import-outside-toplevel
    #from painters import KeypointPainter
# =============================================================================
#     scale = np.sqrt(
#         (np.max(pose[:, 0]) - np.min(pose[:, 0]))
#         * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
#     ) / 50
# =============================================================================
    scale = 1.0/25
    KeypointPainter.show_joint_scales = True
    keypoint_painter = KeypointPainter(line_width=1,
                                       monocolor_connections=False)

    ann = Annotation(keypoints=CAR_KEYPOINTS,
                     skeleton=CAR_SKELETON,
                     score_weights=CAR_SCORE_WEIGHTS)
    ann.set(pose, np.array(weights) * scale)
    draw_ann(ann, filename='./'+prefix+'_skeleton_apollocar.png', keypoint_painter=keypoint_painter)

class KeypointPainter(Configurable):
    """Paint poses.

    The constructor can take any class attribute as parameter and
    overwrite the global default for that instance.

    Example to create a KeypointPainter with thick lines:
    >>> kp = KeypointPainter(line_width=48)
    """

    show_box = False
    show_joint_confidences = False
    show_joint_scales = False
    show_decoding_order = False
    show_frontier_order = False
    show_only_decoded_connections = False

    textbox_alpha = 0.5
    text_color = 'white'
    monocolor_connections = False
    line_width = None
    marker_size = None
    solid_threshold = 0.5
    font_size = 8
    

    def __init__(self, *,
                 xy_scale=1.0,
                 highlight=None,
                 highlight_invisible=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible

        # set defaults for line_width and marker_size depending on monocolor
        if self.line_width is None:
            self.line_width = 2 if self.monocolor_connections else 6
        if self.marker_size is None:
            if self.monocolor_connections:
                #self.marker_size = max(self.line_width + 1, int(self.line_width * 3.0))
                self.marker_size = max(1, int(self.line_width * 2.0))
            else:
                self.marker_size = max(0.75, int(self.line_width * 0.25))

        LOG.debug('color connections = %s, lw = %d, marker = %d',
                  self.monocolor_connections, self.line_width, self.marker_size)

    def _draw_skeleton(self, ax, x, y, v, *,
                       skeleton, skeleton_mask=None, color=None, alpha=1.0,
                       color_map=None, min_w = None, fac_col=None, caf_weights=None,
                       **kwargs):
        if not np.any(v > 0):
            return

        if skeleton_mask is None:
            skeleton_mask = [True for _ in skeleton]
        assert len(skeleton) == len(skeleton_mask)
        
        
        # connections
        lines, line_colors, line_styles = [], [], []
       
        for ci, ((j1i, j2i), mask) in enumerate(zip(np.array(skeleton) - 1, skeleton_mask)):
            if not mask:
                continue
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                #line_colors.append(c)
                col = color_map(int((caf_weights[ci]-min_w)*fac_col))
                line_colors.append(col)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append((0, (1, 3)))
        
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.line_width),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
            alpha=alpha,
        ))

# =============================================================================
#         # joints
#         ax.scatter(
#             x[v > 0.0], y[v > 0.0], s=self.marker_size**2, marker='.',
#             color=color if self.monocolor_connections else 'red',
#             edgecolor='k' if self.highlight_invisible else None,
#             zorder=2,
#             alpha=alpha,
#         )
# =============================================================================
  

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, colors, scales, alpha=1.0):
        for x, y, v, color, scale in zip(xs, ys, vs, colors, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Circle(
                    (x, y),  scale*1.1, # 0.075,
                    fill=True, color=color, alpha=1.0))


    def annotation(self, ax, ann, *, color=None, text=None, subtext=None, alpha=1.0):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        text_is_score = False
        if text is None and hasattr(ann, 'id_'):
            text = '{}'.format(ann.id_)
        if text is None and getattr(ann, 'score', None):
            text = '{:.0%}'.format(ann.score)
            text_is_score = True
        if subtext is None and not text_is_score and getattr(ann, 'score', None):
            subtext = '{:.0%}'.format(ann.score)

        kps = ann.data
        assert kps.shape[1] == 3
        x = kps[:, 0] * self.xy_scale
        y = kps[:, 1] * self.xy_scale
        v = kps[:, 2]

        skeleton_mask = None
        
        normalized_scales = list(ann.joint_scales/np.sum(ann.joint_scales)*len(ann.joint_scales))
        caf_weights = []
        for bone in ann.skeleton:
            caf_weights.append(max(ann.joint_scales[bone[0] - 1],
                                   ann.joint_scales[bone[1] - 1]))
        w_np=np.array(caf_weights)
        caf_weights = list(w_np/np.sum(w_np)*len(caf_weights))
        max_w, min_w = max(caf_weights + normalized_scales), min(caf_weights + normalized_scales)
        min_w = math.floor(min_w*2)/2
        max_w = math.ceil(max_w)
        
        fac_col = 255/(max_w - min_w)
        color_map = matplotlib.cm.get_cmap('viridis_r')
        colors = []
        
        self._draw_skeleton(ax, x, y, v, color=color,
                            skeleton=ann.skeleton, skeleton_mask=skeleton_mask, 
                            alpha=alpha, color_map=color_map, min_w = min_w, fac_col=fac_col, caf_weights=caf_weights)
        
        for scale in normalized_scales:
            col = color_map(int((scale-min_w)*fac_col))
            colors.append(col)
        if self.show_joint_scales and ann.joint_scales is not None:
            self._draw_scales(ax, x, y, v, colors, ann.joint_scales, alpha=alpha)
        
        norm = matplotlib.colors.Normalize(vmin=min_w, vmax=max_w)

# =============================================================================
#         cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                         norm=norm,
#                                 orientation='horizontal'
# =============================================================================
        matplotlib.pyplot.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=norm, cmap=color_map),)

def rotate(pose, angle=45, axis=2):
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    pose_copy = np.copy(pose)
    pose_copy[:, 2] = pose_copy[:, 2] # COOS at human center
    if axis==0:
        rot_mat = np.array([[1, 0, 0],
                            [0, cos, -sin],
                            [0, sin, cos]])
    elif axis==1:
        rot_mat = np.array([[cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
    elif axis==2:
        rot_mat = np.array([[cos, -sin, 0],
                            [sin, cos, 0],
                            [0, 0, 1]])
    else:
        raise Exception("Axis must be 0,1 or 2 (corresponding to x,y,z).")
    rotated_pose = np.transpose(np.matmul(rot_mat, np.transpose(pose_copy)))
    rotated_pose[:,2] = rotated_pose[:,2] + 4 #assure that all z's are bigger than 1
    return rotated_pose

name = "apollocar"

skel = [(bone[0]-1, bone[1]-1) for bone in CAR_SKELETON] # Python fromat --> start at zero
kps = CAR_KEYPOINTS

G = nx.Graph()

G.add_nodes_from(range(len(kps)))
G.add_edges_from(skel)

with open("Edge_weights_train_apollocar.json", 'r') as f:
    edge_weights = json.load(f)
    
G_w = nx.Graph()
G_w.add_nodes_from(range(len(kps)))

for bone_id, bone in enumerate(skel):
    G_w.add_edge(bone[0], bone[1], euclidean_dist = edge_weights[bone_id])
    G_w.add_edge(bone[0], bone[1], euclidean_dist_inverse = 1/edge_weights[bone_id])

G_synthetic = nx.Graph()
G_synthetic.add_nodes_from(range(len(kps)))

for bone_id, bone in enumerate(skel):
    dist_bone = np.linalg.norm(CAR_POSE[bone[0]]-CAR_POSE[bone[1]])
    G_synthetic.add_edge(bone[0], bone[1], euclidean_dist = dist_bone)    

w_bt = get_normalized_weights(nx.betweenness_centrality(G))
w_co = get_normalized_weights(nx.degree_centrality(G))
w_cl = get_normalized_weights(nx.closeness_centrality(G))
w_cl_euclid = get_normalized_weights(nx.closeness_centrality(G_w, distance="euclidean_dist"))
w_harm_cl = get_normalized_weights(nx.harmonic_centrality(G))
w_harm_cl_euclid = get_normalized_weights(nx.harmonic_centrality(G_w, distance="euclidean_dist"))
# w_eigenvector_euclid = get_normalized_weights(nx.eigenvector_centrality(G_w, max_iter=1000, tol=1.0e-6, nstart=None, weight="euclidean_dist_inverse"))

w_harm_euclid_radius_1 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=1, distance="euclidean_dist"))
w_harm_euclid_radius_2 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=2, distance="euclidean_dist"))
w_harm_euclid_radius_3 = get_normalized_weights(harmonic_centrality_local_radius(G_w, radius=3, distance="euclidean_dist"))

w_harm_euclid_radius_3_synthetic = get_normalized_weights(harmonic_centrality_local_radius(G_synthetic, radius=3, distance="euclidean_dist"))
w_harm_cl_euclid_synthetic = get_normalized_weights(nx.harmonic_centrality(G_synthetic, distance="euclidean_dist"))

results = {"keypoints": kps,
           "centrality_closeness_inverse": inverse_normalize(w_cl),
           "centrality_closeness_euclid_inverse": inverse_normalize(w_cl_euclid),
           "centrality_harmonic_inverse": inverse_normalize(w_harm_cl),
           "centrality_harmonic_euclid_inverse": inverse_normalize(w_harm_cl_euclid),
           "w_harm_cl_euclid_synthetic": inverse_normalize(w_harm_cl_euclid_synthetic),
           "w_harm_euclid_radius_1": inverse_normalize(w_harm_euclid_radius_1),
           "w_harm_euclid_radius_2": inverse_normalize(w_harm_euclid_radius_2),
           "w_harm_euclid_radius_3": inverse_normalize(w_harm_euclid_radius_3),
           "w_harm_euclid_radius_3_synthetic": inverse_normalize(w_harm_euclid_radius_3_synthetic)
           }

rot = rotate(CAR_POSE, angle=-70, axis=1)
top_view = rotate(rot, angle=25, axis=0)

draw_skeletons(top_view, inverse_normalize(w_harm_cl_euclid), prefix="w_harm_cl_euclid")
draw_skeletons(top_view, inverse_normalize(w_harm_euclid_radius_3), prefix="w_harm_euclid_radius_3")
draw_skeletons(top_view, inverse_normalize(w_harm_cl_euclid_synthetic), prefix="w_harm_cl_euclid_synthetic")
draw_skeletons(top_view, inverse_normalize(w_harm_euclid_radius_3_synthetic), prefix="w_harm_euclid_radius_3_synthetic")

connect_kps = [item for sublist in SKELETON_CONNECT for item in sublist]
for i in HFLIP_ids:
    if i not in connect_kps:
        top_view[i, 2]=0.5

draw_skeletons(top_view, inverse_normalize(w_harm_cl_euclid), prefix="Dotted_w_harm_cl_euclid")
draw_skeletons(top_view, inverse_normalize(w_harm_euclid_radius_3), prefix="Dotted_w_harm_euclid_radius_3")

with open("Weights_"+name+".json", 'w') as f:
        json.dump(results, f)

df = pd.read_json("Weights_"+name+".json")
export_csv = df.to_csv("Weights_"+name+".csv", index = None, header=True)