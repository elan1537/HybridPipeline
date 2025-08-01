uniform sampler2D localColorSampler;
uniform sampler2D localDepthSampler;
uniform sampler2D wsColorSampler;
uniform sampler2D wsDepthSampler;
uniform bool flipX;
uniform float contrast;
uniform float brightness;
varying vec2 vUv;

void main() {
    vec2 localUv = vUv;
    vec2 wsUv = vec2(1.0 - vUv.x, 1.0 - vUv.y);
    
    vec4 localColor = texture2D(localColorSampler, localUv);
    float localDepth = texture2D(localDepthSampler, localUv).r;

    float d_8bit = floor(localDepth * 255.0 + 0.5) / 255.0;

    vec4 wsColor = texture2D(wsColorSampler, wsUv);
    float wsDepth = texture2D(wsDepthSampler, wsUv).r;

    bool useLocal = wsDepth > d_8bit;

    vec4 finalColor;

    if (flipX) {
        finalColor = useLocal ? localColor : wsColor;
    } else {
        vec2 flipUv = vec2(1.0 - vUv.x, vUv.y);
        vec4 localColorFlipped = texture2D(localColorSampler, flipUv);
        vec4 wsColorFlipped = texture2D(wsColorSampler, vec2(1.0 - flipUv.x, 1.0 - flipUv.y));
        finalColor = useLocal ? localColorFlipped : wsColorFlipped;
    }
    
    vec3 adjustedColor = (finalColor.rgb - 0.5) * contrast + 0.5;
    adjustedColor = adjustedColor * brightness;
    
    gl_FragColor = vec4(adjustedColor, finalColor.a);
}