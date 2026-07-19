# Configuration notes

- `student.language_stack_mode=electronic` keeps the original frozen decoder.
- `student.language_stack_mode=optical_moe` replaces decoder layers with the five-stage language MoE.
- `vision_adapter.tap_stages=[1,3,4]` supplies Qwen's three native DeepStack branches; stage 5 supplies the final vision output.
- `max_visual_tokens=120` and `max_language_tokens=120` are strict limits.
- Vision and language have independent routers, phase masks, global masks, and adapters.
- `loss.router_balance_weight=0.03` applies separately to each active optical router; importance loss remains disabled by default.
