// Simple shader for displaying local color texture only (for LOCAL_ONLY mode)
// Uses same UV flip and colorspace as Fusion for consistency

varying vec2 vUv;
uniform sampler2D localColorSampler;

void main() {
  // Same X-axis flip as Fusion (when fusionFlipX=true: currentUv.x = 1.0 - vUv.x)
  vec2 localUv = vec2(1.0 - vUv.x, vUv.y);
  vec4 localColor = texture2D(localColorSampler, localUv);

  // Use same colorspace conversion as Fusion
  gl_FragColor = linearToOutputTexel(localColor);
}
