# Architecture

## Teacher

`RGB image + MOS prompt -> Qwen processor/chat template -> patch embedding -> electronic vision blocks 0..23 -> native DeepStack mergers at 5/11/17 plus final merger -> multimodal token injection -> electronic language layers 0..27 with native DeepStack additions after 0/1/2 -> final RMSNorm -> last valid prompt token [2048] -> LayerNorm + Linear regression head`

## Student A: vision optical, language electronic

`RGB + prompt -> frozen patch embedding -> vision MoE9x5 with stage taps -> frozen Qwen vision mergers -> native multimodal injection -> frozen electronic language decoder -> frozen final RMSNorm -> answer hidden -> regression head`

Gradients pass through the frozen electronic decoder into the vision MoE; the student forward is not wrapped in `no_grad`.

## Student B: vision and language optical

`RGB + prompt -> frozen patch embedding -> vision MoE9x5 with stage taps -> frozen vision mergers -> multimodal embeddings -> language MoE9x5 with three in-order DeepStack additions -> frozen final RMSNorm -> answer hidden -> regression head`

Vision hidden is `[sum(T),1024]`; language hidden is `[B,S,2048]`. Each side projects hidden channels to 120, maps valid token rows directly into a zero-padded 120×120 field, executes the verified homogeneous optical MoE, pools the 480×480 detector to 120×120 with non-affine LayerNorm and ReLU, reads only valid rows, and restores the original hidden size. The vision output remains 1024 before frozen mergers; language output is restored to 2048.

Both `T` and `S` must be at most 120. Overflow raises an error instead of cropping, pooling, truncating, or resizing tokens.
