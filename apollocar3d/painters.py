import logging

import numpy as np

from openpifpaf.configurable import Configurable

import math

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)


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
                    line_styles.append('dashed')
        
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
        print("cifcaf")
        print(normalized_scales)
        print(caf_weights)
        max_w, min_w = max(caf_weights + normalized_scales), min(caf_weights + normalized_scales)
        min_w = float(math.bottom(min_w))
        max_w = math.ceil(max_w * 2) / 2
        
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
        