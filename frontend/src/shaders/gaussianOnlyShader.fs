// Simple shader for displaying WebSocket color texture only (based on fusionColorShader)
// Uses same UV convention and colorspace as Fusion for consistency

varying vec2 vUv;
uniform sampler2D wsColorSampler;

void main() {
  // Same final UV as fusionColorShader (after both flips cancel out X-axis)
  // fusionFlipX=true + wsFlipX=true results in: vec2(vUv.x, 1.0 - vUv.y)
  vec2 wsUv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 wsColor = texture2D(wsColorSampler, wsUv);

  // Use same colorspace conversion as Fusion
  gl_FragColor = linearToOutputTexel(wsColor);
}
