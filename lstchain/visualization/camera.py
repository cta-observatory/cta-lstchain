def display_source(display, source_pos_xy, **kwargs):
    ax = display.axes
    kwargs['marker'] = 'x' if 'marker' not in kwargs else kwargs['marker']
    kwargs['color'] = 'red' if 'color' not in kwargs else kwargs['color']
    ax.scatter(src_pos.x, src_pos.y, **kwargs)
    return ax