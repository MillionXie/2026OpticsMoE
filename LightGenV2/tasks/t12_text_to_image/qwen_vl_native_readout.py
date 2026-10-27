"""Layout migration for reading the same Qwen's native visual-merger output."""


def native_layout(layout):
    if layout.get('semantic_prefix', 0) or len(layout['positions']) != 64:
        raise ValueError('Migrate only the verified original 64-token fused layout')
    start, suffix = layout['instruction_bounds']
    return {**layout, 'qwen_image_positions': list(layout['positions']),
            'positions': list(range(64)), 'semantic_prefix': 64,
            'instruction_bounds': (start + 64, suffix),
            'visual_source': 'Same Qwen native visual merger before language'}


def valid_native_migration(config, state, visual_cache_present):
    return (config.get('allow_native_readout_migration') is True and visual_cache_present
            and config.get('semantic_prefix') == 64 and not state['config'].get('semantic_prefix', 0))
